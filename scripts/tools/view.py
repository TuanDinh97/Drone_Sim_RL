"""Interactive 3-D visualization (GUI) or headless video of a task flown by a policy or a baseline.

GUI (opens the Isaac Sim window, chase camera on env 0, target markers):
    python scripts/tools/view.py --task Quadsim-Track-v0 --controller geometric
Headless MP4 (+ PNG snapshots) for documentation:
    python scripts/tools/view.py --headless --enable_cameras --video --task Quadsim-Nav-Depth-Render-v0 \
        --controller potential_field --steps 500
Policy:
    python scripts/tools/view.py --task Quadsim-Hover-v0 --checkpoint logs/rsl_rl/quadsim_hover/<run>
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Quadsim-Track-v0")
parser.add_argument("--controller", default="cascade")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--video", action="store_true")
parser.add_argument("--out", default="outputs/videos")
parser.add_argument("--follow", action="store_true", default=True)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab.envs import ViewerCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.rl.overrides import apply_overrides  # noqa: E402
from quadsim.rl.policy_io import agent_cfg_for, latest_checkpoint, load_policy  # noqa: E402
from quadsim.tasks.baselines import BASELINES  # noqa: E402


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    apply_overrides(cfg, agent_cfg_for(args.task), overrides)
    cfg.seed = args.seed
    cfg.debug_vis = True
    cfg.viewer = ViewerCfg(eye=(-2.5, -2.5, 1.6), lookat=(0.0, 0.0, 0.0), resolution=(1280, 720),
                           origin_type="asset_root" if args.follow else "env", asset_name="robot", env_index=0)
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        os.makedirs(args.out, exist_ok=True)
        name = f"{args.task}_{'policy' if args.checkpoint else args.controller}"
        env = gym.wrappers.RecordVideo(env, args.out, name_prefix=name, step_trigger=lambda s: s == 0,
                                       video_length=args.steps, disable_logger=True)
    u = env.unwrapped
    obs, _ = env.reset()
    if args.checkpoint:
        ckpt = args.checkpoint if args.checkpoint.endswith(".pt") else latest_checkpoint(args.checkpoint)
        pol = load_policy(ckpt, env, task=args.task)
        obs, _ = env.reset()
        act = lambda o: pol(o)  # noqa: E731
    else:
        ctrl = BASELINES[args.controller](u)
        act = lambda o: ctrl()  # noqa: E731
    step = 0
    while app.is_running() and step < args.steps:
        with torch.no_grad():
            obs, _, _, _, _ = env.step(act(obs))
        step += 1
    env.close()


if __name__ == "__main__":
    main()
    app.close()
