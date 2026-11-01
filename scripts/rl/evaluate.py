"""Evaluate RL policies and classical baselines on a task under identical conditions.

Runs complete episodes (no truncated first episodes), collects per-episode metrics from the env and
reports mean / std / 95 % CI. Every environment contributes the same number of episodes, its first
k = ceil(episodes / num_envs): stopping at a global episode count would drop the episodes still running at that
moment - the long ones - and bias the sample toward short episodes. Several controllers can be compared in one
call; each uses the same seed, so they face the same sequence of initial states and randomizations.

Examples:
    python scripts/rl/evaluate.py --headless --task Quadsim-Hover-v0 --controllers cascade \
        --checkpoint logs/rsl_rl/quadsim_hover/<run> --episodes 1000
    python scripts/rl/evaluate.py --headless --task Quadsim-Track-v0 --controllers cascade geometric policy ...
    # robustness: evaluate under stronger perturbations than training
    python scripts/rl/evaluate.py ... env.wind.enabled=True env.randomization.mass_scale="(0.8,1.25)"
"""

import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--controllers", nargs="+", default=["policy"],
                    help="policy (needs --checkpoint) and/or registered baselines (quadsim.registry.BASELINES)")
parser.add_argument("--checkpoint", default=None, help="run dir or .pt (needed for 'policy')")
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=500)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--tag", default="")
parser.add_argument("--out_dir", default="logs/eval")
parser.add_argument("--algo", default=None, help="algorithm of --checkpoint (default: inferred from the checkpoint)")
parser.add_argument("--plugins", nargs="*", default=[], help="modules / .py files that register components")
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.rl.overrides import apply_overrides, split_overrides  # noqa: E402
from quadsim.registry import BASELINES, import_plugins  # noqa: E402
from quadsim.rl.algorithms import load_any_policy  # noqa: E402
from quadsim.rl.policy_io import agent_cfg_for, latest_checkpoint  # noqa: E402
from quadsim.tasks import baselines as _baselines  # noqa: E402,F401  (registers the classical controllers)

overrides = split_overrides(args, overrides, sys.argv[1:])  # `--controllers cascade env.a=b`: argparse took env.a=b


def summarize(records):
    keys = [k for k in records[0] if k != "env_id"]
    out = {}
    for k in keys:
        v = torch.cat([r[k] for r in records]).numpy().astype(np.float64)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            out[k] = {"mean": float("nan"), "n": 0}
            continue
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out[k] = {"mean": float(v.mean()), "std": sd, "ci95": float(1.96 * sd / math.sqrt(len(v))),
                  "median": float(np.median(v)), "n": int(len(v))}
        if np.isin(v, (0.0, 1.0)).all():  # rate: Wilson score interval (meaningful at 0 % and 100 %)
            n, p, z = len(v), float(v.mean()), 1.96
            c = (p + z * z / (2 * n)) / (1 + z * z / n)
            h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            out[k]["wilson95"] = [max(0.0, c - h), min(1.0, c + h)]
    return out


def run_controller(name, env):
    u = env.unwrapped
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    u.seed(args.seed)
    obs, _ = env.reset()
    if name == "policy":
        ckpt = latest_checkpoint(args.checkpoint) if os.path.isdir(args.checkpoint) else args.checkpoint
        pol = load_any_policy(ckpt, env, task=args.task, algo=args.algo)
        obs, _ = env.reset()  # runner construction resets the env through the wrapper
        act = lambda o: pol(o)  # noqa: E731
    else:
        ctrl = BASELINES.get(name)(u)
        act = lambda o: ctrl()  # noqa: E731
    u.seed(args.seed)
    obs, _ = env.reset()
    u.episode_metric_buffer.clear()
    u.collect_episode_metrics = True
    k = max(1, math.ceil(args.episodes / u.num_envs))  # episodes per env (quota)
    counts = torch.zeros(u.num_envs, dtype=torch.long)
    recs, seen = [], 0
    t0 = time.time()
    max_steps = (k + 2) * int(u.max_episode_length)
    steps = 0
    while bool((counts < k).any()) and steps < max_steps:
        with torch.no_grad():
            obs, rew, term, trunc, _ = env.step(act(obs))
        for r in u.episode_metric_buffer[seen:]:  # keep each env's first k episodes only
            keep = counts[r["env_id"]] < k
            counts[r["env_id"]] += 1
            if keep.any():
                recs.append({key: v[keep] for key, v in r.items()})
        seen = len(u.episode_metric_buffer)
        steps += 1
    u.collect_episode_metrics = False
    u.episode_metric_buffer.clear()
    res = summarize(recs)
    res["_meta"] = {"episodes": int(counts.clamp(max=k).sum()), "episodes_per_env": k, "wall_s": time.time() - t0,
                    "steps": steps}
    return res


def main():
    import_plugins(args.plugins)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    try:
        agent_cfg = agent_cfg_for(args.task)
    except Exception:  # task registered by a plugin without an rsl_rl config
        agent_cfg = None
    apply_overrides(env_cfg, agent_cfg, overrides)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    results = {"task": args.task, "checkpoint": args.checkpoint, "algo": args.algo, "plugins": args.plugins,
               "seed": args.seed, "num_envs": args.num_envs, "overrides": overrides,
               "episodes_requested": args.episodes, "controllers": {}}
    for c in args.controllers:
        print(f"[eval] {c} ...", flush=True)
        results["controllers"][c] = run_controller(c, env)
        brief = {k: round(v["mean"], 4) for k, v in results["controllers"][c].items() if k != "_meta"}
        print(f"[eval] {c}: {brief}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = os.path.join(args.out_dir, f"{args.task}{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {out}")
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
