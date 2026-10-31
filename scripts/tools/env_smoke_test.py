"""Instantiate each quadsim task, step it with noisy hover actions and check observation sanity.

Usage: python scripts/tools/env_smoke_test.py --headless [--tasks A B ...] [--enable_cameras]
Writes logs/validation/env_smoke_test.json.
"""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--tasks", nargs="*", default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def run(task):
    import omni.usd

    omni.usd.get_context().new_stage()
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    t0 = time.time()
    n_reset = 0
    finite = True
    rew_sum = 0.0
    for _ in range(args.steps):
        a = u.hover_action.expand(u.num_envs, -1) + 0.1 * torch.randn(u.num_envs, u.action_dim, device=u.device)
        obs, rew, term, trunc, _ = env.step(a)
        n_reset += int((term | trunc).sum())
        rew_sum += rew.mean().item()
        for k, v in obs.items():
            finite &= bool(torch.isfinite(v).all())
    dt = time.time() - t0
    shapes = {k: list(v.shape) for k, v in obs.items()}
    res = {"obs_shapes": shapes, "finite": finite, "resets": n_reset, "mean_reward": rew_sum / args.steps,
           "env_steps_per_s": args.steps * u.num_envs / dt,
           "obs_space": str(u.single_observation_space["policy"].shape), "pass": finite}
    env.close()
    return res


def main():
    tasks = args.tasks or [t for t in gym.registry if t.startswith("Quadsim-") and
                           ("Render" not in t and "RGB" not in t or args.enable_cameras)]
    out = {}
    for t in tasks:
        try:
            out[t] = run(t)
        except Exception as e:  # report and continue
            import traceback

            traceback.print_exc()
            out[t] = {"pass": False, "error": repr(e)}
        print(f"[SMOKE] {t}: {out[t]}", flush=True)
    os.makedirs("logs/validation", exist_ok=True)
    with open("logs/validation/env_smoke_test.json", "w") as f:
        json.dump(out, f, indent=2)
    return all(v["pass"] for v in out.values())


if __name__ == "__main__":
    ok = main()
    app.close()
