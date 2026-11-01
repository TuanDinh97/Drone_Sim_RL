"""Hardware detection and resource-aware defaults (number of parallel environments).

The memory model is calibrated on the RTX 5060 8 GB measurements in ``docs/results/benchmark.jsonl`` (process VRAM
while stepping the env) plus a training allowance per workload class. It only ever *lowers* a task's default
``num_envs`` (task defaults are the tuned values) unless ``allow_increase`` is set.
"""

from __future__ import annotations

import os
import subprocess

# workload class -> (simulator base GB, GB per env, training allowance GB, cap)
_MODEL = {
    "state": (2.4, 0.0002, 0.5, 8192),
    "depth_raycast": (2.3, 0.0011, 1.0, 4096),
    "render": (3.6, 0.011, 1.0, 1024),
}


def detect() -> dict:
    """GPU name, total / free VRAM [GB] (first visible GPU), CPU count and RAM [GB]."""
    info = {"gpu": None, "vram_total_gb": 0.0, "vram_free_gb": 0.0, "cpus": os.cpu_count() or 1, "ram_gb": 0.0}
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                                       "--format=csv,noheader,nounits"], text=True, timeout=10)
        name, total, used = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        info.update(gpu=name, vram_total_gb=float(total) / 1024, vram_free_gb=(float(total) - float(used)) / 1024)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            info["ram_gb"] = int(f.readline().split()[1]) / 1024 ** 2
    except Exception:
        pass
    return info


def workload_class(env_cfg) -> str:
    v = getattr(env_cfg, "vision", None)
    if v is None or v.mode == "none":
        return "state"
    return "depth_raycast" if v.source == "raycast" else "render"


def recommend_num_envs(kind: str, vram_free_gb: float, default: int, allow_increase: bool = False,
                       safety: float = 0.85) -> int:
    base, per_env, train, cap = _MODEL[kind]
    budget = vram_free_gb * safety - train - base
    if budget <= 0:
        return 64
    n = int(budget / per_env) // 64 * 64
    n = max(64, min(n, cap))
    return n if allow_increase else min(n, default)


def auto_num_envs(env_cfg, allow_increase: bool = False, verbose: bool = True) -> int:
    """Pick ``num_envs`` for this machine (lowers the task default when the free VRAM would not fit it)."""
    hw = detect()
    default = env_cfg.scene.num_envs
    if not hw["gpu"]:
        return default
    kind = workload_class(env_cfg)
    n = recommend_num_envs(kind, hw["vram_free_gb"], default, allow_increase)
    if verbose:
        print(f"[hardware] {hw['gpu']}: {hw['vram_free_gb']:.1f}/{hw['vram_total_gb']:.1f} GB free, workload "
              f"'{kind}' -> num_envs {n} (task default {default})", flush=True)
    return n
