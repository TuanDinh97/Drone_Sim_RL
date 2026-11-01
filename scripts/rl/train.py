"""Train a quadsim task with any registered algorithm (resumable, checkpointed, fully logged).

Examples:
    python scripts/rl/train.py --task Quadsim-Hover-v0 --headless                       # rsl_rl PPO (default)
    python scripts/rl/train.py --task Quadsim-Hover-v0 --headless --algo sac --max_iterations 20000
    python scripts/rl/train.py --task Quadsim-Track-v0 --headless --num_envs 4096 --max_iterations 1500 --seed 1
    python scripts/rl/train.py --task Quadsim-Track-v0 --headless --resume logs/rsl_rl/quadsim_track/<run>
    python scripts/rl/train.py --task Quadsim-Hover-v0 --headless --plugins examples/custom_vehicle_plugin.py \
        env.dynamics.preset=my_quad

Run directory: logs/rsl_rl/<experiment>/<timestamp>_<run_name>/ (PPO) or logs/<algo>/<experiment>/... containing
model_<it>.pt checkpoints, TensorBoard events, params/env.yaml + params/agent.yaml (exact configuration), git commit
and command line. Hydra-style overrides are accepted as trailing ``env.<field>=<value>`` / ``agent.<field>=<value>``.
Without ``--num_envs`` the task default is used, lowered automatically if the free GPU memory would not fit it.
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--algo", default="ppo", help="registered algorithm: ppo | sac | td3 | <plugin>")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="")
parser.add_argument("--resume", default=None, help="run directory (or checkpoint .pt) to resume from")
parser.add_argument("--logger", default="tensorboard", choices=["tensorboard", "wandb"])
parser.add_argument("--plugins", nargs="*", default=[], help="modules / .py files that register components")
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.registry import ALGORITHMS, import_plugins  # noqa: E402
from quadsim.rl import algorithms as _algos  # noqa: E402,F401  (registers ppo / sac / td3)
from quadsim.rl.overrides import apply_overrides  # noqa: E402
from quadsim.utils.hardware import auto_num_envs  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    import_plugins(args.plugins)
    algo = ALGORITHMS.get(args.algo)()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = algo.default_cfg(args.task)
    apply_overrides(env_cfg, agent_cfg, overrides)
    if args.num_envs is None and not any(o.startswith("env.scene.num_envs") for o in overrides):
        env_cfg.scene.num_envs = auto_num_envs(env_cfg)
    agent_cfg.seed = args.seed
    env_cfg.seed = args.seed
    agent_cfg.device = args.device
    if hasattr(agent_cfg, "logger"):
        agent_cfg.logger = args.logger
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    root = os.path.abspath(algo.log_dir_root(agent_cfg))
    run = time.strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args.run_name}" if args.run_name else "")
    log_dir = os.path.join(root, run)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    os.system(f"git rev-parse HEAD > {log_dir}/params/git_commit.txt 2>/dev/null; "
              f"git diff --stat > {log_dir}/params/git_diff_stat.txt 2>/dev/null")

    env = gym.make(args.task, cfg=env_cfg)
    print_dict({"task": args.task, "algo": args.algo, "num_envs": env.unwrapped.num_envs, "log_dir": log_dir,
                "vehicle": env.unwrapped.params.name, "control": env_cfg.control.mode, "vision": env_cfg.vision.mode})
    t0 = time.time()
    algo.train(env, agent_cfg, log_dir, args.resume)
    print(f"[train] finished in {time.time() - t0:.1f} s -> {log_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # a failed simulator script must not hang in Kit's shutdown (and keep GPU memory)
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()
