"""Export a trained policy to TorchScript + ONNX with its interface contract, and verify it.

Usage: python scripts/rl/export.py --headless --task Quadsim-Hover-v0 --checkpoint logs/rsl_rl/quadsim_hover/<run>
Output: <run>/exported/{policy.onnx, policy.pt, contract.json}
The network comes from the algorithm's ``actor_module`` (PPO, SAC/TD3, or a plugin algorithm that provides one);
``--plugins`` and ``env.<field>=<value>`` overrides work as in train.py / evaluate.py.
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--algo", default=None, help="algorithm of --checkpoint (default: inferred)")
parser.add_argument("--plugins", nargs="*", default=[], help="modules / .py files that register components")
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.deploy.export import contract_from_env, export_policy  # noqa: E402
from quadsim.registry import ALGORITHMS, import_plugins  # noqa: E402
from quadsim.rl.algorithms import algorithm_of_checkpoint, load_any_policy  # noqa: E402
from quadsim.rl.overrides import apply_overrides, split_overrides  # noqa: E402
from quadsim.rl.policy_io import latest_checkpoint  # noqa: E402

overrides = split_overrides(args, overrides, sys.argv[1:])


def main():
    import_plugins(args.plugins)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    apply_overrides(cfg, None, overrides)
    env = gym.make(args.task, cfg=cfg)
    ckpt = latest_checkpoint(args.checkpoint) if os.path.isdir(args.checkpoint) else args.checkpoint
    algo = args.algo or algorithm_of_checkpoint(ckpt)
    pol = load_any_policy(ckpt, env, task=args.task, algo=algo)
    module = ALGORITHMS.get(algo)().actor_module(pol)
    if module is None:
        raise ValueError(f"algorithm '{algo}' provides no exportable network (Algorithm.actor_module)")
    obs, _ = env.reset()
    for _ in range(50):  # real observations as export examples
        with torch.no_grad():
            obs, *_ = env.step(pol(obs))
    u = env.unwrapped
    contract = contract_from_env(u, args.task)
    contract["checkpoint"] = os.path.realpath(ckpt)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(ckpt)), "exported")
    img = obs.get("image")
    rep = export_policy(module, contract, out, obs["policy"][:32].clone(), None if img is None else img[:32].clone())
    print("[EXPORT] " + json.dumps(rep, indent=2), flush=True)
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
