"""Evaluate PX4 SITL runs (scripts/sitl/run_sitl.sh) and build the quadsim-vs-SITL comparison table.

Each SITL run is evaluated with :func:`quadsim.deploy.closed_loop_metrics.scenario_metrics`, the code used for
quadsim (scripts/sitl/quadsim_scenario.py):
  true  = simulator ground truth (gz, exported over uXRCE-DDS),      est = EKF2 VehicleOdometry
  cmd   = rate commands of the bridge's step log (policy action x max body rate)
plus pipeline timing that only exists in SITL (odometry age, policy period, real-time factor).

Usage (system Python of ROS 2 or the quadsim env; needs numpy + scipy, optionally pyulog):
    PYTHONPATH=source/quadsim python3 scripts/sitl/compare.py --sitl logs/sitl/default_r*.npz \
        --quadsim logs/sitl/quadsim_nominal.json --out docs/results/sitl_comparison.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from quadsim.deploy.closed_loop_metrics import body_rates_from_quat, resample, scenario_metrics
from quadsim.deploy.interface import frd_to_flu, ned_to_enu, q_ned_frd_to_enu_flu

MAX_RATE = np.array([6.0, 6.0, 3.0])  # contract: max_body_rate_rad_s


def load_run(path: str):
    base = path[: -len(".npz")]
    d = dict(np.load(path))
    meta = json.load(open(base + ".json"))
    steps = np.genfromtxt(base + ".bridge_steps.csv", delimiter=",", names=True)
    return base, d, meta, steps


def _q_wxyz_norm(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def evaluate_run(path: str, hover_window=(10.0, 30.0)) -> dict:
    base, d, meta, s = load_run(path)
    if meta.get("error"):
        raise RuntimeError(f"{path}: scenario error {meta['error']}")
    t0 = s["t"][0]  # first policy command = hand-over
    goal = np.stack([s["goal_x"], s["goal_y"], s["goal_z"]], -1)
    g0 = goal[0]
    sched = {"goal0": g0.tolist(), "hover_window": hover_window}
    changed = np.nonzero(np.any(np.abs(goal - g0) > 1e-6, -1))[0]
    if len(changed):
        i1 = changed[0]
        i2 = np.nonzero(np.abs(goal[:, 2] - g0[2]) > 1e-6)[0][0]
        sched.update(t_step_x=float(s["t"][i1] - t0), t_step_z=float(s["t"][i2] - t0), t_end=float(s["t"][-1] - t0))
    # ground truth: three topics with identical stamps (gz pose callback); align on the position stamps
    # (PX4's gz ground-truth angular velocity is a backward difference of Euler angles: body rates are recomputed from
    # the ground-truth attitude by central differences instead)
    gp, ga = d["gt_pos"], d["gt_att"]
    tg = gp[:, 1] - t0
    ta = ga[:, 1] - t0
    q_flu = q_ned_frd_to_enu_flu(_q_wxyz_norm(ga[:, 2:6]))
    q_true = _q_wxyz_norm(resample(ta, q_flu, tg))
    true = {"t": tg, "pos": ned_to_enu(gp[:, 2:5]), "q": q_true,
            "rate": resample(ta, body_rates_from_quat(ta, q_flu), tg)}
    o = d["odom"]  # cols: rx, stamp, stamp_sample, pos3, q4, vel3, angvel3, pose_frame, vel_frame, reset
    to = o[:, 2] - t0
    est = {"t": to, "pos": ned_to_enu(o[:, 3:6]), "q": q_ned_frd_to_enu_flu(_q_wxyz_norm(o[:, 6:10])),
           "rate": frd_to_flu(o[:, 13:16])}
    cmd = {"t": s["t"] - t0, "rate": np.stack([s["a1"], s["a2"], s["a3"]], -1).clip(-1, 1) * MAX_RATE,
           "collective": s["a0"]}
    win = (to >= 0) & (to <= cmd["t"][-1])
    for k in ("t", "pos", "q", "rate"):
        est[k] = est[k][win]
    out = scenario_metrics(sched, true, est, cmd)
    # ---- pipeline timing (SITL only)
    age_step = (s["t"] - s["odom_stamp"]) * 1e3
    age_rx = (o[win, 0] - o[win, 2]) * 1e3
    period = np.diff(s["t"]) * 1e3
    rx, st = gp[:, 0] - t0, gp[:, 1] - t0
    w = (rx >= 0) & (rx <= cmd["t"][-1])
    out["timing"] = {
        "odom_age_at_step_ms": {"median": float(np.median(age_step)), "p95": float(np.percentile(age_step, 95)),
                                "max": float(age_step.max())},
        "odom_age_at_receipt_ms": {"median": float(np.median(age_rx)), "p95": float(np.percentile(age_rx, 95))},
        "odom_rate_hz": float(win.sum() / (to[win][-1] - to[win][0])),
        "policy_period_ms": {"median": float(np.median(period)), "p99": float(np.percentile(period, 99)),
                             "max": float(period.max())},
        "policy_steps": int(len(s)),
        "real_time_factor": float(np.polyfit(rx[w], st[w], 1)[0]),
    }
    out["thrust_scale"] = meta["thrust_scale"]
    out["collective_action_mean"] = float(np.mean(s["a0"]))
    out["offboard_duration_s"] = float(cmd["t"][-1])
    out["left_offboard"] = "left_offboard" in meta["events"]
    if os.path.exists(base + ".ulg"):
        out["px4_log"] = ulog_checks(base + ".ulg")
    return out


def ulog_checks(path: str) -> dict:
    """PX4-side view (PX4 clock): regularity of the received rate set-points while in OFFBOARD."""
    try:
        from pyulog import ULog
    except ImportError:
        return {"skipped": "pyulog not installed"}
    u = ULog(path, ["vehicle_status", "vehicle_rates_setpoint"])
    get = {x.name: x.data for x in u.data_list}
    vs = get["vehicle_status"]
    t_off = vs["timestamp"][vs["nav_state"] == 14] * 1e-6
    tr = get["vehicle_rates_setpoint"]["timestamp"] * 1e-6
    m = (tr > t_off[0] + 0.1) & (tr < t_off[-1])
    gaps = np.diff(tr[m]) * 1e3
    return {"rates_setpoints_in_offboard": int(m.sum()), "gap_ms_median": float(np.median(gaps)),
            "gap_ms_max": float(gaps.max()), "gaps_over_30ms": int((gaps > 30).sum())}


def leaves(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from leaves(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield f"{prefix}{k}", float(v)


def aggregate_runs(per_run: list[dict]) -> dict:
    keys = dict(leaves(per_run[0])).keys()
    vals = [dict(leaves(r)) for r in per_run]
    return {k: {"mean": float(np.nanmean([v.get(k, np.nan) for v in vals])),
                "std": float(np.nanstd([v.get(k, np.nan) for v in vals])), "n": len(vals)} for k in keys}


def quadsim_flat(summary: dict) -> dict:
    """quadsim_scenario summary (per-env stats) -> {metric: {mean, std, n}} like :func:`aggregate_runs`."""
    out = {}

    def rec(d, prefix):
        for k, v in d.items():
            if isinstance(v, dict) and "mean" in v and "std" in v:
                out[prefix + k] = {"mean": v["mean"], "std": v["std"], "n": v.get("n_finite", v.get("n"))}
            elif isinstance(v, dict):
                rec(v, prefix + k + ".")
    rec(summary, "")
    return out


ROWS = [  # (label, metric key, scale, unit)
    ("hover RMS 3-D error, ground truth", "hover_true.rms_3d", 100, "cm"),
    ("hover RMS horizontal / vertical, ground truth", ("hover_true.rms_xy", "hover_true.rms_z"), 100, "cm"),
    ("hover RMS 3-D error w.r.t. onboard estimate", "hover_est.rms_3d", 100, "cm"),
    ("estimate vs ground truth position RMS (hover)", "est_pos_err_rms", 100, "cm"),
    ("x step (truth): rise time 10-90 %", "step_x_true.rise_10_90_s", 1, "s"),
    ("x step (truth): overshoot", "step_x_true.overshoot_pct", 1, "%"),
    ("x step (truth): settling time (5 cm)", "step_x_true.settling_s", 1, "s"),
    ("z step (truth): rise time 10-90 %", "step_z_true.rise_10_90_s", 1, "s"),
    ("z step (truth): overshoot", "step_z_true.overshoot_pct", 1, "%"),
    ("z step (truth): settling time (5 cm)", "step_z_true.settling_s", 1, "s"),
    ("x step w.r.t. estimate: rise [s] / overshoot [%] / settling [s]",
     ("step_x_est.rise_10_90_s", "step_x_est.overshoot_pct", "step_x_est.settling_s"), 1, ""),
    ("z step w.r.t. estimate: rise [s] / overshoot [%] / settling [s]",
     ("step_z_est.rise_10_90_s", "step_z_est.overshoot_pct", "step_z_est.settling_s"), 1, ""),
    ("rate cmd -> true rate: delay (roll / pitch)", ("rate_fopdt_roll_true.delay_s", "rate_fopdt_pitch_true.delay_s"),
     1000, "ms"),
    ("rate cmd -> true rate: time constant (roll / pitch)",
     ("rate_fopdt_roll_true.tau_s", "rate_fopdt_pitch_true.tau_s"), 1000, "ms"),
    ("rate cmd -> true rate: gain (roll / pitch)", ("rate_fopdt_roll_true.gain", "rate_fopdt_pitch_true.gain"), 1, ""),
    ("rate cmd -> true rate: VAF (roll / pitch)", ("rate_fopdt_roll_true.vaf", "rate_fopdt_pitch_true.vaf"), 1, ""),
    ("hover collective action a0: std / dominant frequency [Hz]", ("hover_collective.rms", "hover_collective.peak_hz"),
     1, ""),
    ("hover vertical velocity RMS, truth", "hover_heave.rms", 100, "cm/s"),
    ("hover body-rate RMS, truth (roll / pitch)", ("hover_rate.roll.rms", "hover_rate.pitch.rms"), 1, "rad/s"),
    ("hover body-rate dominant frequency (roll / pitch)", ("hover_rate.roll.peak_hz", "hover_rate.pitch.peak_hz"), 1,
     "Hz"),
    ("estimate attitude lag vs truth (roll / pitch)", ("est_att_lag.roll.lag_s", "est_att_lag.pitch.lag_s"), 1000,
     "ms"),
    ("odometry age at policy step: median / p95", ("timing.odom_age_at_step_ms.median",
                                                   "timing.odom_age_at_step_ms.p95"), 1, "ms"),
]


def fmt(stats: dict, key, scale):
    keys = key if isinstance(key, tuple) else (key,)
    parts = []
    for k in keys:
        if k not in stats:
            return "–"
        m, s = stats[k]["mean"] * scale, stats[k]["std"] * scale
        parts.append(f"{m:.3g} ± {s:.2g}" if s > 0 else f"{m:.3g}")
    return " / ".join(parts)


def table(columns: list[tuple[str, dict]]) -> str:
    lines = ["| metric | " + " | ".join(c for c, _ in columns) + " |", "|---|" + "---|" * len(columns)]
    for label, key, scale, unit in ROWS:
        cells = " | ".join(fmt(st, key, scale) for _, st in columns)
        lines.append(f"| {label}{f' [{unit}]' if unit else ''} | {cells} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sitl", nargs="*", default=[], help="SITL run .npz files; 'label=glob' groups runs")
    p.add_argument("--quadsim", nargs="*", default=[], help="quadsim_scenario JSONs; 'label=path'")
    p.add_argument("--out", default="logs/sitl/sitl_comparison.json")
    args = p.parse_args()
    groups = {}
    for item in args.sitl:
        label, pat = item.rsplit("=", 1) if "=" in item else ("PX4 SITL", item)
        if not glob.glob(pat):
            raise FileNotFoundError(f"no SITL runs match {pat!r}")
        groups.setdefault(label, []).extend(sorted(glob.glob(pat)))
    res = {"sitl": {}, "quadsim": {}}
    columns = []
    for item in args.quadsim:
        label, path = item.rsplit("=", 1) if "=" in item else (os.path.basename(item), item)
        q = json.load(open(path))
        res["quadsim"][label] = {"file": path, "num_envs": q["num_envs"], "crashed_or_reset": q["crashed_or_reset"],
                                 "overrides": q["overrides"], "metrics": quadsim_flat(q["summary"])}
        columns.append((f"quadsim: {label} (n={q['num_envs']} envs)", res["quadsim"][label]["metrics"]))
    for label, paths in groups.items():
        per_run = {pth: evaluate_run(pth) for pth in paths}
        agg = aggregate_runs(list(per_run.values()))
        res["sitl"][label] = {"runs": per_run, "aggregate": agg}
        columns.append((f"{label} (n={len(paths)} runs)", agg))
        for pth, r in per_run.items():
            print(f"[sitl] {pth}: hover_true {r['hover_true']['rms_3d']:.4f} m, step_x settle "
                  f"{r.get('step_x_true', {}).get('settling_s', float('nan')):.2f} s, timing {r['timing']}")
    md = table(columns)
    res["table_md"] = md
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(md)
    print(f"[compare] wrote {args.out}")


if __name__ == "__main__":
    main()
