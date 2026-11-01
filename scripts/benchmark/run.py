"""Run a benchmark suite (quadsim.benchmark.suites) for several controllers and write a comparison table.

Examples:
    python scripts/benchmark/run.py --suite hover --policy ppo=logs/rsl_rl/quadsim_hover/<run> \
        --policy sac=logs/sac/quadsim_hover/<run> --tag algos
    python scripts/benchmark/run.py --suite race --policy ppo=logs/rsl_rl/quadsim_race/<run> --baselines cascade
    python scripts/benchmark/run.py --list
Each condition runs scripts/rl/evaluate.py in a fresh process (same seed for every controller). Output:
logs/benchmark_suites/<suite>/<tag>/{<label>__<condition>.json, summary.json, summary.md}; the summary covers every
label evaluated under the same tag (runs with different labels accumulate).
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "quadsim"))
from quadsim.benchmark.suites import BENCHMARKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--suite")
ap.add_argument("--policy", action="append", default=[], help="label=checkpoint[@algo] (repeatable)")
ap.add_argument("--baselines", nargs="*", default=None, help="classical controllers (default: the suite's)")
ap.add_argument("--conditions", nargs="*", default=None)
ap.add_argument("--episodes", type=int, default=None)
ap.add_argument("--tag", default="default")
ap.add_argument("--plugins", nargs="*", default=[])
ap.add_argument("--list", action="store_true")
ap.add_argument("--summarize_only", action="store_true")
args = ap.parse_args()


def _run(suite, cond, label, controller, ckpt=None, algo=None):
    out_dir = os.path.join("logs", "benchmark_suites", suite.name, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{label}__{cond.name}"
    task = cond.task or suite.task
    cmd = [sys.executable, "scripts/rl/evaluate.py", "--headless", "--task", task, "--controllers", controller,
           "--episodes", str(args.episodes or suite.episodes), "--num_envs", str(suite.num_envs),
           "--seed", str(suite.seed), "--out_dir", out_dir, "--tag", tag, *suite.extra_args, *cond.overrides]
    if ckpt:
        cmd += ["--checkpoint", ckpt] + (["--algo", algo] if algo else [])
    if args.plugins:
        cmd += ["--plugins", *args.plugins]
    print(f"[bench] {suite.name}/{cond.name}: {label}", flush=True)
    log = os.path.join(out_dir, f"{tag}.log")
    res = os.path.join(out_dir, f"{task}_{tag}.json")
    if os.path.exists(res):  # never report a stale result if this evaluation fails
        os.remove(res)
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
    if not os.path.exists(res):
        print(f"[bench]   FAILED (see {log})", flush=True)
        return None
    d = json.load(open(res))
    return next(iter(d["controllers"].values()))


def summarize(suite):
    out_dir = os.path.join("logs", "benchmark_suites", suite.name, args.tag)
    rows = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "*__*.json"))):
        dj = json.load(open(f))
        label, cname = os.path.basename(f)[:-5][len(dj["task"]) + 1:].rsplit("__", 1)
        d = next(iter(dj["controllers"].values()))
        rows.setdefault(label, {})[cname] = {k: v for k, v in d.items() if isinstance(v, dict) and "mean" in v}
    conds = [c for c in suite.conditions if any(c.name in r for r in rows.values())]
    summary = {"suite": suite.name, "task": suite.task, "primary": suite.primary,
               "higher_is_better": suite.higher_is_better, "conditions": [vars(c) for c in conds], "results": {}}
    for label, r in rows.items():
        ent = {"per_condition": r}
        pid = [r[c.name][suite.primary]["mean"] for c in conds if c.split == "id" and c.name in r
               and suite.primary in r[c.name]]
        pood = [r[c.name][suite.primary]["mean"] for c in conds if c.split == "ood" and c.name in r
                and suite.primary in r[c.name]]
        if pid and pood:
            ent["id"] = sum(pid) / len(pid)
            ent["ood_mean"] = sum(pood) / len(pood)
            ent["generalization_gap"] = ent["id"] - ent["ood_mean"]
        summary["results"][label] = ent
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    lines = [f"# Benchmark `{suite.name}` ({suite.task})", "", suite.description, "",
             f"Primary metric: `{suite.primary}` ({'higher' if suite.higher_is_better else 'lower'} is better). "
             "± = 95 % CI over episodes.", ""]
    for m in suite.metrics:
        lines += [f"## {m}", "", "| controller | " + " | ".join(f"{c.name} ({c.split})" for c in conds) + " |",
                  "|---|" + "---|" * len(conds)]
        for label, r in rows.items():
            cells = []
            for c in conds:
                v = r.get(c.name, {}).get(m)
                cells.append("–" if v is None or v["mean"] != v["mean"] else
                             f"{v['mean']:.3g} ± {v.get('ci95', 0):.2g}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Generalization gap (primary metric, id − mean(ood))", "", "| controller | id | ood mean | gap |",
              "|---|---|---|---|"]
    for label, ent in summary["results"].items():
        if "id" in ent:
            lines.append(f"| {label} | {ent['id']:.3g} | {ent['ood_mean']:.3g} | {ent['generalization_gap']:+.3g} |")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    if args.plugins:
        from quadsim.registry import import_plugins

        import_plugins(args.plugins)
    if args.list:
        for n in BENCHMARKS:
            s = BENCHMARKS.get(n)
            print(f"{n:10s} {s.task:28s} conditions: {[c.name + '/' + c.split for c in s.conditions]}")
        return
    suite = BENCHMARKS.get(args.suite)
    if not args.summarize_only:
        conds = [c for c in suite.conditions if args.conditions is None or c.name in args.conditions]
        ctrls = [(p.split("=", 1)[0], "policy", *(p.split("=", 1)[1].split("@") + [None])[:2]) for p in args.policy]
        ctrls += [(b, b, None, None) for b in (suite.baselines if args.baselines is None else args.baselines)]
        for c in conds:
            for label, ctrl, ckpt, algo in ctrls:
                _run(suite, c, label, ctrl, ckpt, algo)
    summarize(suite)


if __name__ == "__main__":
    main()
