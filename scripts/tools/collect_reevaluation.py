"""Before/after record of the re-evaluation with the fixed evaluator (``scripts/validation/reevaluate.py``).

Usage: python scripts/tools/collect_reevaluation.py
Reads logs/validation/reevaluate.json (old vs new means of every re-run evaluation) and the regression evaluations
logs/eval/regress_{before,evalfix,review} (reference set: evaluator fix alone, then all review fixes); writes
docs/results/framework/reevaluation.json.
"""

import glob
import json
import os

OUT = "docs/results/framework/reevaluation.json"
KEYS = ("success", "crash", "time_out", "final_pos_err", "track_rmse", "cost", "agent_collision", "time_to_finish",
        "time_to_goal", "lap_time", "wrong_landmark", "episode_length_s")


def _pairs(a_dir, b_dir):
    """Metric-by-metric comparison of two evaluation directories (same file names)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(a_dir, "*.json"))):
        g = os.path.join(b_dir, os.path.basename(f))
        if not os.path.exists(g):
            continue
        a, b = json.load(open(f))["controllers"], json.load(open(g))["controllers"]
        for c in a:
            for k, v in a[c].items():
                if isinstance(v, dict) and "mean" in v and c in b and k in b[c] and v["mean"] == v["mean"]:
                    rows.append({"file": os.path.basename(f), "controller": c, "metric": k, "a": v["mean"],
                                 "b": b[c][k]["mean"], "diff": b[c][k]["mean"] - v["mean"]})
    return rows


def main():
    rep = json.load(open("logs/validation/reevaluate.json")) if os.path.exists("logs/validation/reevaluate.json") \
        else {}
    per_file, changes = {}, []
    for f, e in sorted(rep.items()):
        if not e.get("pass"):
            per_file[f] = {"pass": False}
            continue
        ent = {"pass": True, "episodes_old": e.get("n_old"), "episodes_new": e.get("n_new"), "metrics": {}}
        for c, old in e["old"].items():
            new = e["new"].get(c, {})
            m = {k: {"old": old[k], "new": new.get(k)} for k in KEYS if k in old and old[k] == old[k]}
            ent["metrics"][c] = m
            for k in ("success", "crash"):
                if k in m and m[k]["new"] is not None:
                    changes.append({"file": f, "controller": c, "metric": k, "old": m[k]["old"], "new": m[k]["new"],
                                    "diff": m[k]["new"] - m[k]["old"]})
        per_file[f] = ent
    changes.sort(key=lambda r: -abs(r["diff"]))
    succ = [abs(r["diff"]) for r in changes if r["metric"] == "success"]
    out = {
        "note": "old = published value (evaluator stopped at a global episode count); new = per-environment episode "
                "quota and the review fixes (hover reset frame, cascade integrator reset, race track resampling, "
                "LangNav success/collision exclusivity, rendered-camera re-render). The regression section separates "
                "the evaluator's effect from the code fixes on the reference set.",
        "files": len(rep), "passed": sum(bool(e.get("pass")) for e in rep.values()),
        "success_abs_change_points": {"n": len(succ), "max": 100 * max(succ, default=0.0),
                                      "median": 100 * sorted(succ)[len(succ) // 2] if succ else 0.0,
                                      "over_2_points": sum(s > 0.02 for s in succ)},
        "largest_changes": changes[:25],
        "regression": {},
        "per_file": per_file,
    }
    for name, (a, b) in {"evaluator_only": ("logs/eval/regress_before", "logs/eval/regress_evalfix"),
                         "code_fixes": ("logs/eval/regress_evalfix", "logs/eval/regress_review")}.items():
        rows = _pairs(a, b)
        if rows:
            out["regression"][name] = {"metrics": len(rows), "bit_identical": sum(r["diff"] == 0 for r in rows),
                                       "changed": [r for r in rows if r["diff"] != 0]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[reeval] {out['passed']}/{out['files']} files; |delta success| max {out['success_abs_change_points']['max']:.1f}"
          f" points, median {out['success_abs_change_points']['median']:.2f}; regression: "
          + ", ".join(f"{k} {v['bit_identical']}/{v['metrics']} identical" for k, v in out["regression"].items()))


if __name__ == "__main__":
    main()
