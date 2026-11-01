"""Compare two directories of evaluation JSONs (scripts/rl/evaluate.py) metric by metric.

Usage: python scripts/validation/compare_evals.py <ref_dir> <new_dir> [--tol 0]
Exit status 0 and "PASS" when every controller metric mean agrees within ``--tol`` (0 = bit-identical).
"""

import argparse
import glob
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("new")
ap.add_argument("--tol", type=float, default=0.0)
args = ap.parse_args()

worst, n, missing = 0.0, 0, []
for f in sorted(glob.glob(os.path.join(args.ref, "*.json"))):
    g = os.path.join(args.new, os.path.basename(f))
    if not os.path.exists(g):
        missing.append(os.path.basename(f))
        continue
    a, b = json.load(open(f))["controllers"], json.load(open(g))["controllers"]
    for c in a:
        for k, v in a[c].items():
            if k.startswith("_") or not isinstance(v, dict) or "mean" not in v:
                continue
            w = b.get(c, {}).get(k, {}).get("mean")
            d = abs(v["mean"] - w) if w is not None and v["mean"] == v["mean"] else 0.0
            n += 1
            worst = max(worst, d)
            flag = "" if d <= args.tol else "   <-- differs"
            print(f"{os.path.basename(f):45s} {c:16s} {k:22s} {v['mean']:.10g} {w!s:>22}{flag}")
ok = worst <= args.tol and not missing and n > 0
print(f"{'PASS' if ok else 'FAIL'}: {n} metrics, max |diff| = {worst:.3g}, missing = {missing}")
sys.exit(0 if ok else 1)
