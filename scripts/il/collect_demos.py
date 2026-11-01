"""Collect expert demonstrations (episode-structured HDF5, quadsim.il.dataset) from RL policies or classical controllers.

Examples:
    # RL expert (any registered algorithm's checkpoint), successful episodes only
    python scripts/il/collect_demos.py --headless --task Quadsim-Hover-v0 --expert policy \
        --checkpoint logs/rsl_rl/quadsim_hover/<run> --episodes 500 --out datasets/hover_ppo.h5
    # classical (privileged scripted) expert with RGB images and language instructions
    python scripts/il/collect_demos.py --headless --enable_cameras --task Quadsim-LangNav-RGB-v0 --expert cascade \
        --episodes 2000 --out datasets/langnav_rgb.h5
    # DAgger round: the learner acts, the expert labels every visited state (keep all episodes)
    python scripts/il/collect_demos.py ... --executor logs/il/<run>/model.pt --beta 0.2 --keep all
"""

import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--expert", default="cascade", help="'policy' (needs --checkpoint) or a registered baseline")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--algo", default=None)
parser.add_argument("--executor", default=None, help="checkpoint of a learner that acts instead of the expert (DAgger)")
parser.add_argument("--beta", type=float, default=0.0, help="probability that the expert action is executed (DAgger)")
parser.add_argument("--episodes", type=int, default=500,
                    help="episodes to *finish* (quota of ceil(episodes / num_envs) per env: no episode-length bias)")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--keep", default="success", choices=["success", "all"])
parser.add_argument("--no_images", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--plugins", nargs="*", default=[])
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.il.dataset import DemoWriter, summarize  # noqa: E402
from quadsim.registry import BASELINES, import_plugins  # noqa: E402
from quadsim.rl.algorithms import load_any_policy  # noqa: E402
from quadsim.rl.overrides import apply_overrides  # noqa: E402


def main():
    import_plugins(args.plugins)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    apply_overrides(cfg, None, overrides)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    torch.manual_seed(args.seed)
    if args.expert == "policy":
        expert = load_any_policy(args.checkpoint, env, task=args.task, algo=args.algo)
        expert_name = f"policy:{os.path.realpath(args.checkpoint)}"
    else:
        ctrl = BASELINES.get(args.expert)(u)
        expert, expert_name = (lambda o: ctrl()), args.expert
    executor = load_any_policy(args.executor, env, task=args.task) if args.executor else None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    obs, _ = env.reset()
    writer = DemoWriter(args.out, env, expert_name, u.obs_layout(), record_images=not args.no_images,
                        meta={"task": args.task, "seed": args.seed, "overrides": overrides,
                              "executor": args.executor or "", "beta": args.beta,
                              "tokenizer": u.tokenizer.to_dict() if hasattr(u, "tokenizer") else {}})
    u.episode_metric_buffer.clear()
    u.collect_episode_metrics = True
    keep = (lambda m: m.get("success", 0.0) > 0.5) if args.keep == "success" else (lambda m: True)
    # every env finishes the same number of episodes: stopping at a global count would drop the episodes still in
    # progress at the end, which are the long ones (biasing the data toward short, easy episodes)
    quota = int(math.ceil(args.episodes / u.num_envs))
    finished = [0] * u.num_envs
    t0, steps, n_rec = time.time(), 0, 0
    max_steps = (quota + 2) * int(u.max_episode_length)
    while min(finished) < quota and steps < max_steps:
        with torch.no_grad():
            a_exp = expert(obs)
            a = a_exp
            if executor is not None:
                a_l = executor(obs)
                use_exp = (torch.rand(u.num_envs, 1, device=u.device) < args.beta)
                a = torch.where(use_exp, a_exp, a_l)
        writer.add(obs, a_exp)
        obs, rew, term, trunc, _ = env.step(a)
        writer.add_reward(rew)
        steps += 1
        done = (term | trunc).nonzero().squeeze(-1).tolist()
        if done:
            metrics = {}
            for r in u.episode_metric_buffer[n_rec:]:
                for i, e in enumerate(r["env_id"].tolist()):
                    metrics[e] = {k: float(v[i]) for k, v in r.items() if k != "env_id"}
            n_rec = len(u.episode_metric_buffer)
            in_quota = [e for e in done if finished[e] < quota]
            for e in done:
                finished[e] += 1
            writer.end_episodes(in_quota, metrics, keep)
            for e in set(done) - set(in_quota):
                writer.buf[e] = writer._empty()  # beyond the quota: discard
            writer.refresh_instructions(done)
            if steps % 50 == 0:
                print(f"[demos] {writer.n_saved} saved, {writer.n_seen}/{quota * u.num_envs} finished, "
                      f"{time.time() - t0:.0f} s", flush=True)
    writer.close()
    s = summarize(args.out)
    s["wall_s"] = time.time() - t0
    print("[demos] " + json.dumps(s, indent=2), flush=True)
    with open(args.out + ".json", "w") as f:
        json.dump(s, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
