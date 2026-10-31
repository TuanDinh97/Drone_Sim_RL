"""Record synchronized datasets (ground truth, estimates, IMU, baro, actions, rotor speeds, wind, images).

Examples:
    python scripts/data/record.py --headless --task Quadsim-Track-v0 --controller geometric --steps 500
    python scripts/data/record.py --headless --enable_cameras --task Quadsim-Nav-Depth-Render-v0 \
        --controller potential_field --steps 300
    python scripts/data/record.py --headless --task Quadsim-Hover-v0 --checkpoint logs/rsl_rl/.../model_600.pt

Afterwards the file is re-opened and verified (round trip): shapes, monotonic time, last sample equality.
"""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Quadsim-Track-v0")
parser.add_argument("--controller", default="cascade", help="cascade | geometric | potential_field | policy")
parser.add_argument("--checkpoint", default=None, help="rsl_rl checkpoint (implies --controller policy)")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--record_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--out", default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_noise", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.data.recorder import EpisodeRecorder, load  # noqa: E402
from quadsim.tasks.baselines import BASELINES  # noqa: E402


def main():
    torch.manual_seed(args.seed)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    if args.no_noise:
        cfg.sensors.noise_enabled = False
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    if args.checkpoint:
        from quadsim.rl.policy_io import load_policy

        policy = load_policy(args.checkpoint, env, task=args.task)
        ctrl_name = "policy"
    else:
        ctrl = BASELINES[args.controller](u)
        ctrl_name = args.controller
        policy = None
    os.makedirs("datasets", exist_ok=True)
    out = args.out or f"datasets/{args.task}_{ctrl_name}_{time.strftime('%Y%m%d_%H%M%S')}.h5"
    obs, _ = env.reset()
    rec = EpisodeRecorder(out, u, env_ids=range(min(args.record_envs, u.num_envs)),
                          meta={"task": args.task, "controller": ctrl_name, "seed": args.seed})
    t0 = time.time()
    for _ in range(args.steps):
        with torch.no_grad():
            a = policy(obs) if policy is not None else ctrl()
        obs, rew, term, trunc, _ = env.step(a)
        rec.record(rew, term, trunc)
    last = {"pos": (u.robot.data.root_pos_w - u.scene.env_origins)[: rec.k].cpu().numpy(),
            "action": u.actions[: rec.k].cpu().numpy()}
    rec.close()
    wall = time.time() - t0
    # ---- round trip ----
    data = load(out)
    T = args.steps
    checks = {
        "shapes": data["true/pos"].shape == (T, rec.k, 3) and data["action"].shape[:2] == (T, rec.k),
        "time_monotonic": bool(np.all(np.diff(data["time"]) > 0)),
        "dt_consistent": bool(np.allclose(np.diff(data["time"]), data["attrs"]["dt"])),
        "last_pos_equal": bool(np.allclose(data["true/pos"][-1], last["pos"], atol=1e-6)),
        "last_action_equal": bool(np.allclose(data["action"][-1], last["action"], atol=1e-6)),
        "has_config": len(data["attrs"]["config"]) > 100,
    }
    if "image" in data:
        checks["image_shape"] = list(data["image"].shape)
    res = {"file": out, "size_MB": os.path.getsize(out) / 2**20, "samples": T, "envs": rec.k,
           "wall_s": wall, "checks": checks,
           "pass": all(v for k, v in checks.items() if isinstance(v, bool))}
    print("[RECORD] " + json.dumps(res, indent=2), flush=True)
    os.makedirs("logs/validation", exist_ok=True)
    with open(f"logs/validation/record_{args.task}.json", "w") as f:
        json.dump(res, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
