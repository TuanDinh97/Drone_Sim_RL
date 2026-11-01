"""Evaluate PX4 SITL tracking / waypoint / navigation runs against their quadsim counterparts.

SITL runs come from scripts/sitl/run_sitl.sh (``--task track|waypoint|nav``), quadsim runs from
scripts/sitl/quadsim_task_scenario.py; both are scored with ``quadsim.deploy.closed_loop_metrics`` /
``quadsim.deploy.virtual_sensors`` (truth = simulator ground truth, est = the estimate the policy used).

Usage: PYTHONPATH=source/quadsim python3 scripts/sitl/compare_tasks.py \
    --track "circle=logs/sitl/track_circle_*r*.npz:logs/sitl/quadsim_track_circle_v0.json" \
    --waypoint "logs/sitl/waypoint_r*.npz:logs/sitl/quadsim_waypoint_v0.json" \
    --nav "logs/sitl/nav_l*_r1.npz:logs/sitl/quadsim_nav_range_v2.json" --out docs/results/sitl_tasks.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from quadsim.deploy.closed_loop_metrics import resample, tracking_metrics, waypoint_metrics
from quadsim.deploy.interface import SinusoidReference, ned_to_enu
from quadsim.deploy.virtual_sensors import clearance


def load(path):
    base = path[: -len(".npz")]
    s = np.genfromtxt(base + ".bridge_steps.csv", delimiter=",", names=True)
    return dict(np.load(path)), json.load(open(base + ".json")), s


def streams(d, s):
    t0 = s["t"][0]
    g, o = d["gt_pos"], d["odom"]
    return (t0, g[:, 1] - t0, ned_to_enu(g[:, 2:5]), o[:, 2] - t0, ned_to_enu(o[:, 3:6]))


def track_run(path):
    d, m, s = load(path)
    tp = m["task_params"]
    ref = SinusoidReference.shape(tp["track_shape"], tp["track_radius"], tp["track_freq_hz"], tp["track_z_amp"],
                                  tp["track_start_enu"])
    t0, tg, pg, to, po = streams(d, s)
    dur = s["t"][-1] - t0
    out = {"duration_s": float(dur), "left_offboard": "left_offboard" in m["events"]}
    for src, t, p in (("true", tg, pg), ("est", to, po)):
        w = (t >= 0) & (t <= dur)
        for k, v in tracking_metrics(t[w], p[w], ref(t[w])[0], 2.0, dur).items():
            out[f"{src}_{k}"] = v
    return out


def waypoint_run(path):
    d, m, s = load(path)
    wps = np.asarray(m["task_params"]["waypoints_enu"]).reshape(-1, 3)
    t0, tg, pg, to, po = streams(d, s)
    out = {"left_offboard": "left_offboard" in m["events"]}
    for src, t, p in (("true", tg, pg), ("est", to, po)):
        w = waypoint_metrics(t, p, wps, 0.3)
        out.update({f"{src}_{k}": v for k, v in w.items() if k != "reach_times"})
        out[f"{src}_reach_times"] = w["reach_times"]
    t_end = out["est_time_to_finish"] if np.isfinite(out["est_time_to_finish"]) else 20.0
    w = (tg >= 0) & (tg <= t_end)
    out["true_min_pass_dist"] = [float(np.linalg.norm(pg[w] - wp, axis=1).min()) for wp in wps]
    out["max_speed"] = float(np.linalg.norm(np.gradient(pg[w], tg[w], axis=0), axis=1).max())
    e = resample(to, po, tg[w]) - pg[w]
    out["est_err_rms"] = float(np.sqrt((e ** 2).sum(1).mean()))
    out["est_err_max"] = float(np.linalg.norm(e, axis=1).max())
    return out


def nav_run(path):
    d, m, s = load(path)
    nav = m["nav"]
    goal, cyl = np.asarray(nav["goal_enu"]), np.asarray(nav["cylinders"])
    t0, tg, pg, to, po = streams(d, s)
    ev = m["events"]
    t_end = (ev.get("goal_reached") or ev.get("collision") or ev["handback"]) - t0  # same clock as the streams
    w = (tg >= 0) & (tg <= t_end + 0.05)
    clr = clearance(pg[w], cyl)
    success = "goal_reached" in ev
    return {"layout": nav["layout_index"], "success": success, "collision": "collision" in ev,
            "timeout": not success and "collision" not in ev, "left_offboard": "left_offboard" in ev,
            "time_to_goal": float(t_end) if success else float("nan"), "min_clearance": float(clr.min()),
            "path_length": float(np.linalg.norm(np.diff(pg[w], axis=0), axis=1).sum()),
            "min_goal_dist": float(np.linalg.norm(pg[w] - goal, axis=1).min())}


def agg(rows, keys):
    out = {}
    for k in keys:
        x = np.array([float(r[k]) for r in rows])
        f = x[np.isfinite(x)]
        out[k] = {"mean": float(f.mean()) if len(f) else float("nan"), "std": float(f.std()) if len(f) else 0.0,
                  "n": int(len(f))}
    return out


def fmt(st, k, scale=1.0, nd=3):
    v = st.get(k)
    if v is None or not np.isfinite(v["mean"]):
        return "–"
    return f"{v['mean'] * scale:.{nd}g} ± {v['std'] * scale:.2g}" if v["std"] > 0 else f"{v['mean'] * scale:.{nd}g}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", nargs="*", default=[], help="'label=sitl_glob:quadsim_json'")
    p.add_argument("--waypoint", nargs="*", default=[], help="'sitl_glob:quadsim_json'")
    p.add_argument("--nav", nargs="*", default=[], help="'label=sitl_glob:quadsim_json' (several groups)")
    p.add_argument("--out", default="docs/results/sitl_tasks.json")
    args = p.parse_args()
    res, md = {"track": {}, "waypoint": {}, "nav": {}}, []
    for item in args.track:
        label, rest = item.split("=", 1)
        pat, qj = rest.split(":")
        runs = {f: track_run(f) for f in sorted(glob.glob(pat))}
        q = json.load(open(qj))
        keys = ["true_rmse", "true_rmse_xy", "true_rmse_z", "true_max", "est_rmse", "est_max"]
        res["track"][label] = {"runs": runs, "sitl": agg(list(runs.values()), keys), "quadsim": q["summary"],
                               "quadsim_crash_rate": q["crash_rate"], "reference": q["reference"]}
        r = res["track"][label]
        md += [f"\n**Tracking — {label}** (SITL n={len(runs)} runs, quadsim n={q['num_envs']} envs; RMSE after 2 s)\n",
               "| metric | quadsim | PX4 SITL |", "|---|---|---|"]
        for k, lab in (("true_rmse", "RMSE truth [cm]"), ("true_rmse_xy", "RMSE horizontal, truth [cm]"),
                       ("true_rmse_z", "RMSE vertical, truth [cm]"), ("true_max", "max error, truth [cm]"),
                       ("est_rmse", "RMSE w.r.t. estimate [cm]")):
            md.append(f"| {lab} | {fmt(r['quadsim'], k, 100)} | {fmt(r['sitl'], k, 100)} |")
    for item in args.waypoint:
        pat, qj = item.split(":")
        runs = {f: waypoint_run(f) for f in sorted(glob.glob(pat))}
        q = json.load(open(qj))
        keys = ["est_success", "est_time_to_finish", "true_reached", "max_speed", "est_err_rms", "est_err_max"]
        res["waypoint"] = {"runs": runs, "sitl": agg(list(runs.values()), keys), "quadsim": q["summary"],
                           "quadsim_crash_rate": q["crash_rate"], "waypoints": q["waypoints"]}
        r = res["waypoint"]
        md += [f"\n**Waypoints** (5-waypoint course; SITL n={len(runs)} runs, quadsim n={q['num_envs']} envs)\n",
               "| metric | quadsim | PX4 SITL |", "|---|---|---|",
               f"| course completed (policy's own logic) | {fmt(r['quadsim'], 'true_success')} | "
               f"{fmt(r['sitl'], 'est_success')} |",
               f"| time to finish [s] | {fmt(r['quadsim'], 'true_time_to_finish')} | "
               f"{fmt(r['sitl'], 'est_time_to_finish')} |",
               f"| waypoints passed within 0.3 m on truth (of 5) | {fmt(r['quadsim'], 'true_reached')} | "
               f"{fmt(r['sitl'], 'true_reached')} |",
               f"| estimate error during the course, RMS / max [cm] | – | {fmt(r['sitl'], 'est_err_rms', 100)} / "
               f"{fmt(r['sitl'], 'est_err_max', 100)} |"]
    groups = []
    for idx, item in enumerate(args.nav):
        label, rest = item.split("=", 1) if "=" in item else (f"SITL {idx}", item)
        pat, qj = rest.rsplit(":", 1)
        runs = {f: nav_run(f) for f in sorted(glob.glob(pat))}
        q = json.load(open(qj))
        key = "nav" if idx == 0 else f"nav_{idx}"
        res[key] = {"label": label, "runs": runs, "quadsim_per_layout": q["per_layout"],
                    "quadsim_overall": q["overall"], "layouts_file": q["layouts_file"],
                    "sitl_success_rate": float(np.mean([r["success"] for r in runs.values()])),
                    "sitl_collision_rate": float(np.mean([r["collision"] for r in runs.values()]))}
        groups.append(key)
    if groups:
        q = res["nav"]
        lay = json.load(open(q["layouts_file"]))["layouts"]
        md += ["\n**Navigation (range ring, velocity interface)** — per layout: quadsim success / collision rate over "
               "its envs vs the PX4 SITL outcome (time to goal [s])\n",
               "| layout | obstacles | quadsim | " + " | ".join(res[g]["label"] for g in groups) + " |",
               "|---|---|---|" + "---|" * len(groups)]
        for li in sorted(q["quadsim_per_layout"], key=int):
            ql = q["quadsim_per_layout"][li]
            cells = []
            for g in groups:
                r = next((r for r in res[g]["runs"].values() if r["layout"] == int(li)), None)
                if r is None:
                    cells.append("–")
                else:
                    outc = f"goal ({r['time_to_goal']:.2f})" if r["success"] else \
                        ("collision" if r["collision"] else "timeout")
                    cells.append(f"{outc}, min clearance {r['min_clearance']:.2f} m")
            md.append(f"| {li} | {len(lay[int(li)]['cylinders'])} | {ql['success']:.2f} / {ql['collision']:.2f} "
                      f"({ql['time_to_goal']:.2f}) | " + " | ".join(cells) + " |")
        sr = " | ".join(f"{res[g]['sitl_success_rate']:.2f} success" for g in groups)
        md.append(f"| all | | {q['quadsim_overall']['success']:.2f} / {q['quadsim_overall']['collision']:.2f} | {sr} |")
    res["markdown"] = "\n".join(md)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(res["markdown"])
    print(f"[compare_tasks] wrote {args.out}")


if __name__ == "__main__":
    main()


def velocity_cmd_world(a, yaw, max_vel=(3.0, 3.0, 2.0)):
    """Velocity-interface action (yaw-aligned frame, ``ActionDecoder.decode``) -> world ENU velocity command."""
    v = np.clip(a[:, :3], -1, 1) * np.asarray(max_vel)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)


def velocity_loop_sitl(paths, t_max=4.0):
    """FOPDT of commanded -> EKF velocity per world axis over the first ``t_max`` s of the navigation runs."""
    from quadsim.deploy.closed_loop_metrics import fit_fopdt

    out = {}
    for ax, i in (("x", 0), ("y", 1), ("z", 2)):
        fits = []
        for f in paths:
            d, m, s = load(f)
            t0 = s["t"][0]
            q = np.stack([s["qw"], s["qx"], s["qy"], s["qz"]], -1)
            yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            cmd = velocity_cmd_world(np.stack([s["a0"], s["a1"], s["a2"]], -1), yaw)
            o = d["odom"]
            to, vo = o[:, 2] - t0, ned_to_enu(o[:, 10:13])
            ts = s["t"] - t0
            w, wo = ts <= t_max, (to >= 0) & (to <= t_max)
            fits.append(fit_fopdt(ts[w], cmd[w, i], to[wo], vo[wo, i], max_delay=0.3,
                                  taus=np.geomspace(0.02, 2.0, 60)))
        out[ax] = {k: float(np.mean([f[k] for f in fits])) for k in ("delay_s", "tau_s", "gain", "vaf")}
    return out


def velocity_loop_quadsim(npz, t_max=4.0):
    from quadsim.deploy.closed_loop_metrics import fit_fopdt

    q = np.load(npz)
    n_t, n_env = q["act"].shape[:2]
    t = np.arange(n_t) * float(q["dt"])
    w = t <= t_max
    out = {}
    for ax, i in (("x", 0), ("y", 1), ("z", 2)):
        fits = []
        for e in range(n_env):
            cmd = velocity_cmd_world(q["act"][:, e], q["yaw"][:, e], q["max_vel"])
            # action at step k is applied during (t_k, t_k+1]; velocity is sampled at the end of the step
            fits.append(fit_fopdt(t[w], cmd[w, i], t[w] + float(q["dt"]), q["vel"][w, e, i], max_delay=0.3,
                                  taus=np.geomspace(0.02, 2.0, 60)))
        out[ax] = {k: float(np.mean([f[k] for f in fits])) for k in ("delay_s", "tau_s", "gain", "vaf")}
    return out
