"""Throughput and VRAM benchmark of the quadsim environments (environment stepping only, no learning).

Usage: python scripts/tools/benchmark.py --headless --task Quadsim-Hover-v0 --num_envs 256 1024 4096
       python scripts/tools/benchmark.py --headless --enable_cameras --task Quadsim-Nav-Depth-Render-v0 --num_envs 64 128
Each configuration runs in a fresh stage; results are appended to logs/benchmark/benchmark.jsonl.
Run with an otherwise idle GPU for meaningful numbers.
"""

import argparse
import json
import os
import subprocess
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Quadsim-Hover-v0")
parser.add_argument("--num_envs", type=int, nargs="+", default=[256, 1024, 4096])
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--warmup", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def process_vram_mb() -> float:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                       "--format=csv,noheader,nounits"]).decode()
        for line in out.strip().splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            if int(pid) == os.getpid():
                return float(mem)
    except Exception:
        pass
    return float("nan")


def bench(n):
    import omni.usd

    omni.usd.get_context().new_stage()
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=n)
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    env.reset()
    a = u.hover_action.expand(n, -1).clone()
    with torch.no_grad():
        for _ in range(args.warmup):
            env.step(a + 0.05 * torch.randn_like(a))
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.steps):
            env.step(a + 0.05 * torch.randn_like(a))
        torch.cuda.synchronize()
    dt = time.time() - t0
    res = {"task": args.task, "num_envs": n, "env_steps_per_s": n * args.steps / dt,
           "physics_steps_per_s": n * args.steps * u.cfg.decimation / dt,
           "ms_per_env_step": 1e3 * dt / args.steps, "process_vram_MB": process_vram_mb(),
           "vision": u.cfg.vision.mode, "vision_source": u.cfg.vision.source,
           "gpu": torch.cuda.get_device_name(0)}
    env.close()
    return res


def main():
    os.makedirs("logs/benchmark", exist_ok=True)
    for n in args.num_envs:
        try:
            r = bench(n)
        except Exception as e:  # e.g. out of memory
            r = {"task": args.task, "num_envs": n, "error": repr(e)[:300]}
        print("[BENCH] " + json.dumps(r), flush=True)
        with open("logs/benchmark/benchmark.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
    app.close()
