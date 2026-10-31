"""Train a quadsim task with rsl_rl PPO (resumable, checkpointed, fully logged).

Examples:
    python scripts/rl/train.py --task Quadsim-Hover-v0 --headless
    python scripts/rl/train.py --task Quadsim-Track-v0 --headless --num_envs 4096 --max_iterations 1500 --seed 1
    python scripts/rl/train.py --task Quadsim-Track-v0 --headless --resume logs/rsl_rl/quadsim_track/<run>
    python scripts/rl/train.py --task Quadsim-Nav-Depth-v0 --headless --num_envs 1024

Run directory: logs/rsl_rl/<experiment>/<timestamp>_<run_name>/ containing model_<it>.pt checkpoints,
TensorBoard events, params/env.yaml + params/agent.yaml (exact configuration), git commit and command line.
Hydra-style overrides are accepted as trailing ``env.<field>=<value>`` / ``agent.<field>=<value>`` arguments.
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="")
parser.add_argument("--resume", default=None, help="run directory (or checkpoint .pt) to resume from")
parser.add_argument("--logger", default="tensorboard", choices=["tensorboard", "wandb"])
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.rl.overrides import apply_overrides  # noqa: E402
from quadsim.rl.policy_io import agent_cfg_for, build_runner, latest_checkpoint  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = agent_cfg_for(args.task)
    apply_overrides(env_cfg, agent_cfg, overrides)
    agent_cfg.seed = args.seed
    env_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.logger = args.logger
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
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
    runner, _ = build_runner(env, agent_cfg, log_dir, agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if args.resume:
        ckpt = args.resume if args.resume.endswith(".pt") else latest_checkpoint(args.resume)
        print(f"[train] resuming from {ckpt}")
        runner.load(ckpt)
    print_dict({"task": args.task, "num_envs": env.unwrapped.num_envs, "log_dir": log_dir,
                "control": env_cfg.control.mode, "vision": env_cfg.vision.mode})
    t0 = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[train] finished in {time.time() - t0:.1f} s -> {log_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
