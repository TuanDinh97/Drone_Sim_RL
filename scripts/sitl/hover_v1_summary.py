"""Multi-seed summary: hover policies trained on the v0 vs the v1 (sim2real) distribution.

Inputs (written by the evaluation batch):
  logs/eval/hover_v1/Quadsim-Hover-v{0,1}_<run>_<cond>.json     scripts/rl/evaluate.py, cond = v0env | v1env | harsh
  logs/sitl/hover_v1/quadsim_<run>_g{1.0,1.6}.json              scripts/sitl/quadsim_scenario.py, thrust-map gain g
  logs/sitl/hover_v1/sitl_<run>_{linear,curve}_r*.npz            PX4 SITL hover/step runs (optional)
Per family (v0 = ``*_ctbr_s*``, v1 = ``*_v1_s*``): mean +- std over seeds -> docs/results/hover_v1.json.

Usage: PYTHONPATH=source/quadsim python3 scripts/sitl/hover_v1_summary.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from compare import evaluate_run  # noqa: E402

E, S = "logs/eval/hover_v1", "logs/sitl/hover_v1"
OSC = 0.1  # collective-action std above which a hover window counts as a limit cycle (quiet runs: <= 0.03)


def family(run: str) -> str:
    m = re.search(r"_(v[12])_s\d+$", run)
    return m.group(1) if m else "v0"


def ms(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    return {"mean": float(x.mean()) if len(x) else float("nan"), "std": float(x.std()) if len(x) else 0.0,
            "n": int(len(x)), "values": x.tolist()}


def main():
    runs = sorted({re.sub(r"^Quadsim-Hover-v\d_(.*)_(v0env|v1env|harsh)\.json$", r"\1", os.path.basename(f))
                   for f in glob.glob(f"{E}/Quadsim-Hover-v*_*.json")})
    per_run = {}
    for r in runs:
        d = {"family": family(r)}
        for cond in ("v0env", "v1env", "harsh"):
            f = glob.glob(f"{E}/Quadsim-Hover-v*_{r}_{cond}.json")
            if f:
                p = json.load(open(f[0]))["controllers"]["policy"]
                d[cond] = {k: p[k]["mean"] for k in ("success", "crash", "final_pos_err", "final_yaw_err") if k in p}
        for g in ("1.0", "1.6"):
            f = f"{S}/quadsim_{r}_g{g}.json"
            if os.path.exists(f):
                sm = json.load(open(f))["summary"]
                d[f"scen_g{g}"] = {"hover_est_rms": sm["hover_est"]["rms_3d"]["mean"],
                                   "collective_std": sm["hover_collective"]["rms"]["mean"],
                                   "heave_rms": sm["hover_heave"]["rms"]["mean"],
                                   "step_z_settling": sm["step_z_est"]["settling_s"]["mean"],
                                   "step_x_settling": sm["step_x_est"]["settling_s"]["mean"]}
        for mp in ("linear", "curve"):
            fs = sorted(glob.glob(f"{S}/sitl_{r}_{mp}_r*.npz"))
            if fs:
                ev = [evaluate_run(f) for f in fs]
                n_lc = int(sum(e["hover_collective"]["rms"] > OSC for e in ev))
                d[f"sitl_{mp}"] = {"runs": len(ev), "limit_cycle_runs": n_lc,
                                   "collective_std": float(np.mean([e["hover_collective"]["rms"] for e in ev])),
                                   "hover_est_rms": float(np.mean([e["hover_est"]["rms_3d"] for e in ev])),
                                   "step_x_settling": float(np.nanmean([e["step_x_est"]["settling_s"] for e in ev])),
                                   "step_z_settling": float(np.nanmean([e["step_z_est"]["settling_s"] for e in ev]))}
        per_run[r] = d
    fam = {}
    for f in ("v0", "v1", "v2"):
        rs = [d for d in per_run.values() if d["family"] == f]
        if not rs:
            continue
        out = {"seeds": len(rs)}
        for sec in ("v0env", "v1env", "harsh", "scen_g1.0", "scen_g1.6", "sitl_linear", "sitl_curve"):
            keys = sorted({k for d in rs if sec in d for k in d[sec]})
            out[sec] = {k: ms([d[sec].get(k) for d in rs if sec in d]) for k in keys}
        fam[f] = out
    res = {"per_run": per_run, "family": fam, "limit_cycle_threshold_collective_std": OSC}
    with open("docs/results/hover_v1.json", "w") as fh:
        json.dump(res, fh, indent=2)
    for f, out in fam.items():
        print(f"== {f} ({out['seeds']} seeds)")
        for sec, kv in out.items():
            if isinstance(kv, dict):
                print("  ", sec, {k: (round(v["mean"], 4), round(v["std"], 4)) for k, v in kv.items()})


if __name__ == "__main__":
    main()
