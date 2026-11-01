"""Aggregate the multi-seed replications (scripts/rl/multiseed.sh) into docs/results/framework/multiseed.json.

Usage: python scripts/tools/collect_multiseed.py
Every result lists its per-seed values (seed 42, or BC seed 0, is the published single-seed run), their mean, sample
standard deviation (n-1), min and max over seeds. Missing sources are listed under "missing" (nothing is invented).
"""

import json
import math
import os

OUT = "docs/results/framework/multiseed.json"
missing = []


def stats(per_seed):
    v = [x for x in per_seed.values() if x is not None and x == x]
    if not v:
        return None
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else None  # undefined for one seed
    return {"per_seed": per_seed, "n_seeds": len(v), "mean": m, "std": sd, "min": min(v), "max": max(v)}


def bench(suite, tag, label):
    p = f"logs/benchmark_suites/{suite}/{tag}/summary.json"
    if not os.path.exists(p):
        missing.append(p)
        return None
    r = json.load(open(p))["results"].get(label)
    if r is None:
        missing.append(f"{p}: {label}")
    return r


def metric(c, key):
    v = (c or {}).get(key)
    return v["mean"] if isinstance(v, dict) else None


def ev(path):
    if not os.path.exists(path):
        missing.append(path)
        return None
    return next(iter(json.load(open(path))["controllers"].values()))


def suite_seeds(suite, runs, conds, keys):
    """runs: {seed: (tag, label)} -> {condition: {metric: stats}} (+ sources)."""
    res = {s: bench(suite, *tl) for s, tl in runs.items()}
    out = {c: {k: stats({s: metric((r or {}).get("per_condition", {}).get(c), k) for s, r in res.items()})
               for k in keys} for c in conds}
    out["_sources"] = {s: f"logs/benchmark_suites/{suite}/{t}/summary.json:{lab}" for s, (t, lab) in runs.items()}
    return out


def il_seeds(tags, tasks, keys):
    out = {}
    for tag in tags:
        files = {0: tag, 1: f"{tag}_s1", 2: f"{tag}_s2"}
        out[tag] = {name: {k: stats({s: metric(ev(f"logs/eval/il/{task}_{t}.json"), k) for s, t in files.items()})
                           for k in keys} for name, task in tasks.items()}
    return out


result = {
    "note": "seed 42 (RL) / seed 0 (BC) = the first published runs; seeds 1, 2 from scripts/rl/multiseed.sh with "
            "identical configurations and code. All seeds evaluated with the per-environment episode quota "
            "(review fix). BC seeds vary initialization and train/val split on the same demonstrations (DAgger "
            "data were collected once, with the seed-0 learners). std = sample std over seeds (n - 1).",
    "race": {
        "v0_ccw_circuits": suite_seeds("race", {42: ("final", "ppo"), 1: ("seeds", "ppo_s1"), 2: ("seeds", "ppo_s2")},
                                       ("random_tracks", "ellipse", "figure8", "wind"),
                                       ("success", "crash", "lap_time", "mean_speed")),
        "v1_both_directions": suite_seeds("race", {42: ("final", "ppo_v1"), 1: ("seeds", "ppo_v1_s1"),
                                                   2: ("seeds", "ppo_v1_s2")},
                                          ("random_tracks", "ellipse", "figure8", "wind"),
                                          ("success", "crash", "lap_time", "mean_speed")),
    },
    "swarm": {"ppo": suite_seeds("swarm", {42: ("final", "ppo"), 1: ("seeds", "ppo_s1"), 2: ("seeds", "ppo_s2")},
                                 ("swap4", "wind", "latency"),
                                 ("success", "agent_collision", "min_separation", "time_to_goal"))},
    "safe": {"budget_s": 0.1, "ppo_lagrangian": {
        k: stats({s: metric(ev(f"logs/eval/safe/Quadsim-Waypoint-SpeedCost-v0_safe_lagrangian_s{s}.json"), k)
                  for s in (42, 1, 2)}) for k in ("success", "cost", "time_to_finish", "crash")}},
    "langnav_symbolic": il_seeds(("langnav_sym_bc_v2_unkavg", "langnav_sym_bc_v2", "langnav_sym_dagger2",
                                  "langnav_sym_dagger2_clip"),
                                 {"train_phrasing": "Quadsim-LangNav-v0",
                                  "heldout_phrasing": "Quadsim-LangNav-Heldout-v0"}, ("success", "wrong_landmark")),
    "langnav_rgb": il_seeds(("langnav_rgb_bc", "langnav_rgb_dagger1", "langnav_rgb_dagger1_clip"),
                            {"train_phrasing": "Quadsim-LangNav-RGB-v0",
                             "heldout_phrasing": "Quadsim-LangNav-RGB-Heldout-v0"}, ("success", "wrong_landmark")),
}
c = result["safe"]["ppo_lagrangian"]["cost"]
if c:
    result["safe"]["seeds_within_budget"] = sum(v <= 0.1 for v in c["per_seed"].values() if v is not None)
result["missing"] = missing = list(dict.fromkeys(missing))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)


def line(name, st, pct=True):
    if not st:
        return f"| {name} | – |"
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3g}")
    seeds = " / ".join("–" if v is None else f(v) for v in st["per_seed"].values())
    spread = f"{f(st['mean'])} ± {f(st['std'])}" if st["std"] is not None else f"{f(st['mean'])}"
    return f"| {name} | {seeds} | {spread} (n={st['n_seeds']}) |"


rows = ["| result | per seed | mean ± std |", "|---|---|---|"]
for var, r in result["race"].items():
    for cond in ("random_tracks", "ellipse", "figure8", "wind"):
        rows.append(line(f"race {var} {cond} success %", r[cond]["success"]))
    rows.append(line(f"race {var} lap time s", r["random_tracks"]["lap_time"], False))
for cond in ("swap4", "wind", "latency"):
    rows.append(line(f"swarm {cond} success %", result["swarm"]["ppo"][cond]["success"]))
    rows.append(line(f"swarm {cond} collision %", result["swarm"]["ppo"][cond]["agent_collision"]))
for k in ("cost", "success"):
    rows.append(line(f"safe PPO-Lagrangian {k}", result["safe"]["ppo_lagrangian"][k], k == "success"))
for fam in ("langnav_symbolic", "langnav_rgb"):
    for tag, r in result[fam].items():
        for ph in ("train_phrasing", "heldout_phrasing"):
            rows.append(line(f"{tag} {ph} success %", r[ph]["success"]))
print("\n".join(rows))
print(f"[multiseed] wrote {OUT}; missing: {missing}")
