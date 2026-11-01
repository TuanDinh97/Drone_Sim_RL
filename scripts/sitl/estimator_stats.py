"""Identify the error statistics of PX4 EKF2 against the simulator ground truth from SITL recordings.

Hover runs (stationary): per-flight position bias, slow (< 0.5 Hz) Gauss-Markov-like position error (std,
correlation time), fast position error, velocity error (std, correlation time), roll/pitch error. Tracking runs (in
motion): time shift between EKF2 position and truth (``estimate_delay``) -- exposes the GNSS delay model.
These numbers parametrize the EKF-like odometry model of the v1 training distribution (quadsim.tasks.apply_sim2real_v1).

Usage: PYTHONPATH=source/quadsim python3 scripts/sitl/estimator_stats.py --hover "logs/sitl/curve_r*.npz" \
    --motion "circle GPS delay 110 ms=logs/sitl/track_circle_gpsdelay110_r1.npz" \
    --motion "circle GPS delay 0=logs/sitl/track_circle_r1.npz" --out docs/results/sitl_estimator.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.signal import butter, filtfilt

from quadsim.deploy.closed_loop_metrics import estimate_delay, resample, roll_pitch
from quadsim.deploy.interface import ned_to_enu, q_ned_frd_to_enu_flu


def corr_time(x, dt):
    """1/e decay time of the autocorrelation."""
    x = x - x.mean()
    c = np.correlate(x, x, "full")[len(x) - 1:]
    c = c / c[0]
    k = np.argmax(c < np.exp(-1)) if np.any(c < np.exp(-1)) else len(c)
    return float(k * dt)


def streams(path):
    d = np.load(path)
    s = np.genfromtxt(path[:-4] + ".bridge_steps.csv", delimiter=",", names=True)
    t0 = s["t"][0]
    g, o, a = d["gt_pos"], d["odom"], d["gt_att"]
    qn = lambda q: q / np.linalg.norm(q, axis=1, keepdims=True)  # noqa: E731
    return dict(tg=g[:, 1] - t0, pg=ned_to_enu(g[:, 2:5]), vg=ned_to_enu(g[:, 5:8]), ta=a[:, 1] - t0,
                qg=q_ned_frd_to_enu_flu(qn(a[:, 2:6])), to=o[:, 2] - t0, po=ned_to_enu(o[:, 3:6]),
                vo=ned_to_enu(o[:, 10:13]), qo=q_ned_frd_to_enu_flu(qn(o[:, 6:10])))


def hover_stats(path, t0=5.0, t1=30.0, fs=50.0):
    x = streams(path)
    m = (x["tg"] >= t0) & (x["tg"] <= t1)
    t = x["tg"][m]
    ep = resample(x["to"], x["po"], t) - x["pg"][m]
    # truth velocity: central difference of the ground-truth position (PX4's gz vx/vy/vz is a backward difference)
    vt = np.gradient(x["pg"][m], t, axis=0)
    ev = resample(x["to"], x["vo"], t) - vt
    ea = resample(x["to"], roll_pitch(x["qo"]), t) - resample(x["ta"], roll_pitch(x["qg"]), t)
    b, a = butter(2, 0.5 / (fs / 2))
    slow = filtfilt(b, a, ep - ep.mean(0), axis=0)
    return {"pos_bias": ep.mean(0).tolist(), "pos_slow_std": slow.std(0).tolist(),
            "pos_slow_corr_s": [corr_time(slow[:, i], 1 / fs) for i in range(3)],
            "pos_fast_std": (ep - ep.mean(0) - slow).std(0).tolist(), "vel_std": ev.std(0).tolist(),
            "vel_corr_s": [corr_time(ev[:, i], 1 / fs) for i in range(3)], "att_std_rad": ea.std(0).tolist()}


def motion_stats(path):
    x = streams(path)
    out = {}
    for i, ax in enumerate("xyz"):
        out[f"pos_lag_{ax}_s"] = estimate_delay(x["tg"], x["pg"][:, i], x["to"], x["po"][:, i], 0.3)["lag_s"]
    m = (x["tg"] >= 2.0) & (x["tg"] <= x["to"].max())
    e = resample(x["to"], x["po"], x["tg"][m]) - x["pg"][m]
    out["pos_err_rms"] = float(np.sqrt((e ** 2).sum(1).mean()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hover", nargs="+", required=True)
    p.add_argument("--motion", action="append", default=[], help="'label=path'")
    p.add_argument("--out", default="docs/results/sitl_estimator.json")
    args = p.parse_args()
    files = sorted(f for pat in args.hover for f in glob.glob(pat))
    per = {f: hover_stats(f) for f in files}
    keys = list(next(iter(per.values())).keys())
    summary = {k: np.mean([v[k] for v in per.values()], 0).tolist() for k in keys}
    summary["pos_bias_abs_mean"] = np.mean([np.abs(v["pos_bias"]) for v in per.values()], 0).tolist()
    motion = {lab: motion_stats(path) for lab, path in (m.rsplit("=", 1) for m in args.motion)}
    res = {"hover_runs": per, "hover_mean": summary, "motion": motion}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({"hover_mean": {k: np.round(v, 4).tolist() for k, v in summary.items()}, "motion": motion},
                     indent=1))


if __name__ == "__main__":
    main()
