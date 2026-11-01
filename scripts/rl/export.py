"""Export a trained policy to TorchScript + ONNX with its interface contract, and verify it.

Usage: python scripts/rl/export.py --headless --task Quadsim-Hover-v0 --checkpoint logs/rsl_rl/quadsim_hover/<run>
Output: <run>/exported/{policy.onnx, policy.pt, contract.json}
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", default=None)
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.deploy.export import contract_from_env, export_policy  # noqa: E402
from quadsim.rl.policy_io import latest_checkpoint, load_policy  # noqa: E402


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg)
    ckpt = args.checkpoint if args.checkpoint.endswith(".pt") else latest_checkpoint(args.checkpoint)
    pol = load_policy(ckpt, env, task=args.task)
    obs, _ = env.reset()
    for _ in range(50):  # real observations as export examples
        with torch.no_grad():
            obs, *_ = env.step(pol(obs))
    u = env.unwrapped
    contract = contract_from_env(u, args.task)
    contract["checkpoint"] = os.path.realpath(ckpt)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(ckpt)), "exported")
    img = obs.get("image")
    rep = export_policy(pol.policy, contract, out, obs["policy"][:32].clone(), None if img is None else img[:32].clone())
    print("[EXPORT] " + json.dumps(rep, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
