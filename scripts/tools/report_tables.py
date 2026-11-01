"""Generate LaTeX tables for the technical report directly from the JSON results in docs/results/.

Usage: python scripts/tools/report_tables.py   ->  report/generated/*.tex
Avoids manual transcription of numbers; every table cites its source file.
"""

import glob
import json
import os

R = "docs/results"
OUT = "report/generated"


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def w(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)


def pct(x):
    return f"{100 * x:.1f}\\,\\%"


def m2():
    rows = []
    d = {p: load(f"m2_dynamics_{p}.json") for p in ("x500", "crazyflie")}
    if not all(d.values()):
        return
    x, c = d["x500"], d["crazyflie"]
    rows.append(("PhysX mass / inertia rel. error", f"{x['T1_inertial']['mass_rel_err']:.0e} / {x['T1_inertial']['inertia_rel_err']:.0e}",
                 f"{c['T1_inertial']['mass_rel_err']:.0e} / {c['T1_inertial']['inertia_rel_err']:.0e}", "$<10^{-4}$"))
    rows.append(("Hover altitude drift over 2\\,s [mm]", f"{1e3 * x['T2_hover']['max_abs_dz_m']:.2f}",
                 f"{1e3 * c['T2_hover']['max_abs_dz_m']:.2f}", "$<5$"))
    rows.append(("Thrust step: accel. RMS err / peak", f"{100 * x['T3_thrust_step']['rel']:.3f}\\,\\%",
                 f"{100 * c['T3_thrust_step']['rel']:.3f}\\,\\%", "$<2\\,\\%$"))
    rows.append(("Yaw accel. vs $\\tau_z/J_z$", f"{100 * x['T4_yaw']['rel_err']:.4f}\\,\\%", f"{100 * c['T4_yaw']['rel_err']:.4f}\\,\\%", "$<3\\,\\%$"))
    rows.append(("Open loop 1.5\\,s, pos. err vs semi-implicit ref. [mm]", f"{1e3 * x['T5_open_loop']['pos_err_final_semi_implicit_m']:.2f}",
                 f"{1e3 * c['T5_open_loop']['pos_err_final_semi_implicit_m']:.2f}", "$<10$"))
    rows.append(("Open loop, att. err vs semi-implicit ref. [deg]", f"{x['T5_open_loop']['att_err_max_semi_implicit_deg']:.3f}",
                 f"{c['T5_open_loop']['att_err_max_semi_implicit_deg']:.3f}", "$<0.5$"))
    rows.append(("Open loop, pos. err vs RK4 ref. [mm] (PhysX discretization)", f"{1e3 * x['T5_open_loop']['pos_err_final_rk4_m']:.1f}",
                 f"{1e3 * c['T5_open_loop']['pos_err_final_rk4_m']:.1f}", "info"))
    rows.append(("Closed-loop velocity cascade, PhysX vs ref. [mm]", f"{1e3 * x['T6_closed_loop']['pos_err_final_m']:.1f}",
                 f"{1e3 * c['T6_closed_loop']['pos_err_final_m']:.1f}", "$<30$"))
    body = "\n".join(f"{a} & {b} & {cc} & {dd}\\\\" for a, b, cc, dd in rows)
    w("m2.tex", "\\begin{tabular}{lccc}\\toprule\nTest & X500 & Crazyflie & Criterion\\\\\\midrule\n" + body +
      "\n\\bottomrule\\end{tabular}\n")


def depth():
    rows = []
    for p in (0, 15):
        d = load(f"m3_depth_consistency_pitch{p}.json")
        if d:
            rows.append(f"{p}$^\\circ$ & {1e3 * d['median_abs_err_m']:.1f} & {1e2 * d['p90_abs_err_m']:.1f} & "
                        f"{pct(d['hit_miss_agreement'])} & {pct(d['self_occluded_pixel_frac'])} & {d['num_pixels_compared']}\\\\")
    w("depth.tex", "\\begin{tabular}{lccccc}\\toprule\nCamera pitch & median [mm] & p90 [cm] & hit/miss agree & self-occl. & pixels\\\\"
      "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")


def fmt(v, key, scale=1.0, unit_pct=False, digits=1):
    if key not in v or v[key].get("n", 1) == 0:
        return "--"
    m, ci = v[key]["mean"], v[key].get("ci95", 0.0)
    if unit_pct:
        return f"{100 * m:.{digits}f} $\\pm$ {100 * ci:.{digits}f}"
    return f"{scale * m:.{digits}f} $\\pm$ {scale * ci:.{digits}f}"


NAMES = {"policy": "PPO (ours)", "cascade": "Cascade PID", "geometric": "SE(3) geometric", "potential_field": "Potential field",
         "geometric_v1": "SE(3) v1 (torque-level @50\\,Hz)$^\\dagger$", "policy_noDR": "PPO w/o DR", "policy_DR": "PPO with DR"}


def eval_table(files, cols, label):
    lines = []
    for tag, fname in files:
        d = load(fname)
        if not d:
            continue
        for c, v in d["controllers"].items():
            vals = " & ".join(fmt(v, k, s, p, dg) for k, s, p, dg in cols)
            n = v["_meta"]["episodes"]
            lines.append(f"{tag} & {NAMES.get(c, c)} & {vals} & {n}\\\\")
        lines.append("\\midrule")
    if lines:
        lines = lines[:-1]
    return "\n".join(lines)


def evals():
    hover_cols = [("success", 1, True, 1), ("crash", 1, True, 1), ("final_pos_err", 100, False, 2), ("final_yaw_err", 57.2958, False, 1)]
    body = eval_table([("nominal", "eval_hover_nominal.json"), ("wind + pushes", "eval_hover_wind_push.json")], hover_cols, "hover")
    w("hover.tex", "\\begin{tabular}{llccccc}\\toprule\nCondition & Controller & success [\\%] & crash [\\%] & final pos. err [cm] & "
      "final yaw err [deg] & episodes\\\\\\midrule\n" + body + "\n\\bottomrule\\end{tabular}\n")
    tc = [("track_rmse", 100, False, 1), ("track_max_err", 100, False, 1), ("crash", 1, True, 1)]
    body = eval_table([("aggressive ($\\le$4\\,m/s)", "eval_track_nominal.json"), ("moderate ($\\le$2\\,m/s)", "eval_track_moderate.json")], tc, "track")
    w("track.tex", "\\begin{tabular}{llcccc}\\toprule\nReference set & Controller & RMSE [cm] & max err [cm] & crash [\\%] & episodes\\\\"
      "\\midrule\n" + body + "\n\\bottomrule\\end{tabular}\n")
    wc = [("success", 1, True, 1), ("waypoints_reached_frac", 1, True, 1), ("time_to_finish", 1, False, 2), ("crash", 1, True, 1)]
    body = eval_table([("nominal", "eval_waypoint_nominal.json")], wc, "wp")
    w("waypoint.tex", "\\begin{tabular}{llccccc}\\toprule\nCondition & Controller & success [\\%] & waypoints [\\%] & time [s] & crash [\\%] & episodes\\\\"
      "\\midrule\n" + body + "\n\\bottomrule\\end{tabular}\n")
    nc = [("success", 1, True, 1), ("collision", 1, True, 1), ("time_to_goal", 1, False, 2), ("min_clearance", 100, False, 0)]
    files = [(json.load(open(f)).get("label", os.path.basename(f)), os.path.basename(f))
             for f in sorted(glob.glob(f"{R}/eval_nav_*.json"))]
    body = eval_table(files, nc, "nav")
    w("nav.tex", "\\begin{tabular}{llccccc}\\toprule\nPolicy input / eval & Controller & success [\\%] & collision [\\%] & time [s] & "
      "min clear. [cm] & episodes\\\\\\midrule\n" + body + "\n\\bottomrule\\end{tabular}\n")
    files = [(f.split("eval_dr_ablation_")[1][:-5].split("_", 1)[1].replace("_", " "), os.path.basename(f))
             for f in sorted(glob.glob(f"{R}/eval_dr_ablation_*.json"))]
    body = eval_table(files, hover_cols, "dr")
    w("dr.tex", "\\begin{tabular}{llccccc}\\toprule\nEval. condition & Policy & success [\\%] & crash [\\%] & final pos. err [cm] & "
      "final yaw err [deg] & episodes\\\\\\midrule\n" + body + "\n\\bottomrule\\end{tabular}\n")


def bench():
    p = os.path.join(R, "benchmark.jsonl")
    if not os.path.exists(p):
        return
    rows = []
    for line in open(p):
        d = json.loads(line)
        if "error" in d:
            rows.append(f"{d['task']} & {d['num_envs']} & \\multicolumn{{3}}{{c}}{{failed: {d['error'][:40]}}}\\\\")
            continue
        rows.append(f"\\texttt{{{d['task'].replace('Quadsim-', '')}}} & {d['num_envs']} & {d['env_steps_per_s'] / 1e3:.1f} & "
                    f"{d['ms_per_env_step']:.2f} & {d['process_vram_MB'] / 1024:.2f}\\\\")
    w("bench.tex", "\\begin{tabular}{lrrrr}\\toprule\nTask & envs & k env-steps/s & ms / step & VRAM [GB]\\\\\\midrule\n" +
      "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")


def _ms(st, key, scale=1.0, digits=2):
    """mean +- std cell from a {key: {mean, std}} dict."""
    v = st.get(key)
    if v is None or v.get("mean") is None or v["mean"] != v["mean"]:
        return "--"
    m, sd = scale * v["mean"], scale * v.get("std", 0.0)
    return f"{m:.{digits}f} $\\pm$ {sd:.{digits}f}" if sd > 0 else f"{m:.{digits}f}"


def sitl():
    """PX4 SITL cross-check tables (docs/results/sitl_*.json)."""
    c = load("sitl_comparison.json")
    if c:
        cols = [("quadsim nominal", c["quadsim"].get("nominal", {}).get("metrics", {}))]
        for lab, short in (("PX4 SITL, linear thrust map", "SITL linear map"), ("PX4 SITL, thrust-curve map",
                           "SITL curve map"), ("PX4 SITL, linear + CA_ROTOR fix", "SITL linear + geometry fix")):
            if lab in c["sitl"]:
                cols.append((f"{short} ({c['sitl'][lab]['aggregate']['hover_true.rms_3d']['n']})",
                             c["sitl"][lab]["aggregate"]))
        rows = [("Hover RMS error, truth [cm]", "hover_true.rms_3d", 100, 2),
                ("Hover RMS error w.r.t. estimate [cm]", "hover_est.rms_3d", 100, 2),
                ("Estimate vs truth, hover RMS [cm]", "est_pos_err_rms", 100, 2),
                ("$x$ step w.r.t. estimate: rise 10--90\\,\\% [s]", "step_x_est.rise_10_90_s", 1, 2),
                ("$x$ step w.r.t. estimate: settling (5\\,cm) [s]", "step_x_est.settling_s", 1, 2),
                ("$z$ step w.r.t. estimate: rise 10--90\\,\\% [s]", "step_z_est.rise_10_90_s", 1, 2),
                ("$z$ step w.r.t. estimate: settling (5\\,cm) [s]", "step_z_est.settling_s", 1, 2),
                ("Rate cmd$\\to$rate: delay roll [ms]", "rate_fopdt_roll_true.delay_s", 1000, 0),
                ("Rate cmd$\\to$rate: delay pitch [ms]", "rate_fopdt_pitch_true.delay_s", 1000, 0),
                ("Rate cmd$\\to$rate: $\\tau$ roll [ms]", "rate_fopdt_roll_true.tau_s", 1000, 0),
                ("Rate cmd$\\to$rate: $\\tau$ pitch [ms]", "rate_fopdt_pitch_true.tau_s", 1000, 0),
                ("Hover collective $a_0$ std", "hover_collective.rms", 1, 3),
                ("Hover vertical velocity RMS, truth [cm/s]", "hover_heave.rms", 100, 1),
                ("Odometry age at policy step, median [ms]", "timing.odom_age_at_step_ms.median", 1, 1)]
        head = " & ".join(["Metric"] + [h for h, _ in cols])
        body = "\n".join(" & ".join([lab] + [_ms(st, k, sc, dg) for _, st in cols]) + "\\\\" for lab, k, sc, dg in rows)
        w("sitl_hover.tex", f"\\begin{{tabular}}{{l{'c' * len(cols)}}}\\toprule\n{head}\\\\\\midrule\n{body}\n"
          "\\bottomrule\\end{tabular}\n")
    e = load("sitl_estimator.json")
    if e:
        h = e["hover_mean"]
        f3 = lambda v, s=100, d=1: " / ".join(f"{s * x:.{d}f}" for x in v)  # noqa: E731
        rows = [("Per-flight position bias $|\\bar e_p|$ (x / y / z) [cm]", f3(h["pos_bias_abs_mean"])),
                ("Slow position error ($<$0.5\\,Hz) std [cm]", f3(h["pos_slow_std"])),
                ("Slow position error correlation time [s]", f3(h["pos_slow_corr_s"], 1, 1)),
                ("Fast position error std [mm]", f3(h["pos_fast_std"], 1000, 1)),
                ("Velocity error std [cm/s]", f3(h["vel_std"], 100, 2)),
                ("Velocity error correlation time [s]", f3(h["vel_corr_s"], 1, 2)),
                ("Roll / pitch error std [mrad]", f3(h["att_std_rad"], 1000, 2))]
        for lab, m in e["motion"].items():
            gd = "110" if "110" in lab else "0"
            rows.append((f"Circle flight, \\texttt{{EKF2\\_GPS\\_DELAY}}={gd}\\,ms: lag x / y [ms], RMS error [cm]",
                         f"{1e3 * m['pos_lag_x_s']:.0f} / {1e3 * m['pos_lag_y_s']:.0f}, {100 * m['pos_err_rms']:.1f}"))
        n = len(e["hover_runs"])
        w("sitl_estimator.tex", "\\begin{tabular}{lc}\\toprule\nQuantity & PX4 EKF2 in SITL\\\\\\midrule\n" +
          "\n".join(f"{a_} & {b_}\\\\" for a_, b_ in rows) + f"\n\\bottomrule\\end{{tabular}}\n% {n} hover runs\n")
    t = load("sitl_tasks.json")
    if t:
        rows = []
        for lab, r in t["track"].items():
            q, si = r["quadsim"], r["sitl"]
            rows.append(f"Tracking, {lab} & RMSE truth [cm] & {_ms(q, 'true_rmse', 100, 1)} & {_ms(si, 'true_rmse', 100, 1)}\\\\")
            rows.append(f" & RMSE w.r.t. estimate [cm] & {_ms(q, 'est_rmse', 100, 1)} & {_ms(si, 'est_rmse', 100, 1)}\\\\")
        if t.get("waypoint"):
            q, si = t["waypoint"]["quadsim"], t["waypoint"]["sitl"]
            rows.append(f"Waypoints (5) & course completed & {_ms(q, 'true_success', 1, 2)} & {_ms(si, 'est_success', 1, 2)}\\\\")
            rows.append(f" & time to finish [s] & {_ms(q, 'true_time_to_finish', 1, 2)} & "
                        f"{_ms(si, 'est_time_to_finish', 1, 2)}\\\\")
            rows.append(f" & estimate error during course, RMS [cm] & -- & {_ms(si, 'est_err_rms', 100, 1)}\\\\")
        w("sitl_tasks.tex", "\\begin{tabular}{llcc}\\toprule\nTask & Metric & quadsim & PX4 SITL\\\\\\midrule\n" +
          "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
        if t.get("nav"):
            lay = json.load(open(t["nav"]["layouts_file"]))["layouts"] if "layouts_file" in t["nav"] else None
            rows = []
            groups = {k: v for k, v in t.items() if k.startswith("nav")}
            per = t["nav"]["quadsim_per_layout"]
            sitl_by = {g: {r["layout"]: r for r in v["runs"].values()} for g, v in groups.items()}
            for li in sorted(per, key=int):
                ql = per[li]
                cells = []
                for g in groups:
                    r = sitl_by[g].get(int(li))
                    outc = None if r is None else ("goal" if r["success"] else "collision" if r["collision"] else "timeout")
                    cells.append(outc or "--")
                nob = len(lay[int(li)]["cylinders"]) if lay else ""
                rows.append(f"{li} & {nob} & {100 * ql['success']:.0f} / {100 * ql['collision']:.0f} & " +
                            " & ".join(cells) + "\\\\")
            heads = " & ".join(t[g].get("label", g) for g in groups)
            w("sitl_nav.tex", f"\\begin{{tabular}}{{ccc{'c' * len(groups)}}}\\toprule\nLayout & obstacles & quadsim "
              f"success / collision [\\%] & {heads}\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")


def hover_v1():
    """v0 vs v1 training distribution, mean +- std over seeds (docs/results/hover_v1.json)."""
    d = load("hover_v1.json")
    if not d:
        return
    f = d["family"]
    fams = [k for k in ("v0", "v1", "v2") if k in f]
    rows = [("quadsim, v0 distribution: success [\\%]", "v0env", "success", 100, 1),
            ("quadsim, v0 distribution: final pos. error [cm]", "v0env", "final_pos_err", 100, 2),
            ("quadsim, v1 distribution: success [\\%]", "v1env", "success", 100, 1),
            ("quadsim, v1 distribution: final pos. error [cm]", "v1env", "final_pos_err", 100, 2),
            ("quadsim, harsh OOD: success [\\%]", "harsh", "success", 100, 1),
            ("quadsim, harsh OOD: crash [\\%]", "harsh", "crash", 100, 1),
            ("scenario $g=1.0$: hover RMS w.r.t. estimate [cm]", "scen_g1.0", "hover_est_rms", 100, 2),
            ("scenario $g=1.0$: collective std", "scen_g1.0", "collective_std", 1, 3),
            ("scenario $g=1.6$: collective std", "scen_g1.6", "collective_std", 1, 3),
            ("scenario $g=1.6$: vertical velocity RMS [cm/s]", "scen_g1.6", "heave_rms", 100, 1),
            ("SITL linear map: runs with limit cycle", "sitl_linear", "limit_cycle_runs", 1, 2),
            ("SITL linear map: collective std", "sitl_linear", "collective_std", 1, 3),
            ("SITL linear map: hover RMS w.r.t. estimate [cm]", "sitl_linear", "hover_est_rms", 100, 2),
            ("SITL linear map: $x$ step settling [s]", "sitl_linear", "step_x_settling", 1, 2),
            ("SITL curve map: collective std", "sitl_curve", "collective_std", 1, 3),
            ("SITL curve map: hover RMS w.r.t. estimate [cm]", "sitl_curve", "hover_est_rms", 100, 2)]
    def total(fam, sec):  # limit-cycle runs summed over seeds
        st = f[fam].get(sec, {})
        if "limit_cycle_runs" not in st:
            return "--"
        return f"{int(sum(st['limit_cycle_runs']['values']))} / {int(sum(st['runs']['values']))}"

    lines = []
    for lab, sec, key, sc, dg in rows:
        if key == "limit_cycle_runs":
            lines.append(" & ".join([lab] + [total(k, sec) for k in fams]) + "\\\\")
            if sec == "sitl_linear":
                lines.append(" & ".join(["SITL curve map: runs with limit cycle"] + [total(k, "sitl_curve") for k in fams])
                             + "\\\\")
            continue
        cells = [_ms(f[k].get(sec, {}), key, sc, dg) for k in fams]
        if all(c == "--" for c in cells):
            continue
        lines.append(" & ".join([lab] + cells) + "\\\\")
    head = " & ".join(["Metric"] + [f"{k} ({f[k]['seeds']} seeds)" for k in fams])
    w("hover_v1.tex", f"\\begin{{tabular}}{{l{'c' * len(fams)}}}\\toprule\n{head}\\\\\\midrule\n" + "\n".join(lines) +
      "\n\\bottomrule\\end{tabular}\n")


def _m(e, key, pct_=False, digits=2, scale=1.0):
    if not e or key not in e or e[key]["mean"] is None or e[key]["mean"] != e[key]["mean"]:
        return "--"
    v, ci = e[key]["mean"] * scale, (e[key].get("ci95") or 0.0) * scale
    if pct_:
        return f"{100 * v:.1f}\\,\\%"
    return f"{v:.{digits}f}" + (f"\\,$\\pm$\\,{ci:.{digits}f}" if ci else "")


def _table(caption, label, cols, rows, spec=None):
    spec = spec or ("l" + "c" * (len(cols) - 1))
    head = " & ".join(cols)
    body = "\n".join(" & ".join(r) + "\\\\" for r in rows)
    return (f"\\begin{{table}}[!htbp]\\centering\\small\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{adjustbox}}{{max width=\\linewidth}}\\begin{{tabular}}{{{spec}}}\\toprule\n{head}\\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\\end{{adjustbox}}\n\\end{{table}}\n\n")


def framework():
    F = "docs/results/framework"
    L = lambda n: json.load(open(os.path.join(F, n))) if os.path.exists(os.path.join(F, n)) else None  # noqa: E731
    out = []
    # --- algorithms x conditions (hover suite)
    b = L("bench_hover_algos.json")
    if b:
        conds = [c["name"] for c in b["conditions"]]
        rows = []
        for lab in ("ppo", "sac", "bc_distilled", "cascade"):
            r = b["results"].get(lab)
            if not r:
                continue
            pc = r["per_condition"]
            rows.append([lab.replace("_", r"\_")] + [_m(pc.get(c), "success", True) for c in conds]
                        + [f"{100 * r['generalization_gap']:+.1f}"])
        out.append(_table("Hover benchmark suite: success per condition (1000 episodes each, same seed, fresh process "
                          "per controller and condition) and generalization gap id$-$mean(ood) in points. "
                          "Source: \\texttt{docs/results/framework/bench\\_hover\\_algos.json}.", "tab:fw_algos",
                          ["Controller"] + [c.replace("_", r"\_") + (" (id)" if i == 0 else "") for i, c in
                                            enumerate(conds)] + ["gap"], rows))
    o = L("offpolicy_hover.json")
    if o:
        rows = []
        for k, lab in (("ppo_reference", "PPO (rsl\\_rl), 4096 envs, 600 it."), ("sac", "SAC (quadsim), 1024 envs, 15k steps"),
                       ("td3", "TD3 (quadsim), 1024 envs, 15k steps"), ("td3b", "TD3, noise 0.05-0.15, reward scale 10, 8 upd./step"),
                       ("ars", "ARS (plugin example), 150 it.")):
            if k in o and o[k].get("eval"):
                e = o[k]["eval"]
                t = o[k].get("train_wall_s")
                rows.append([lab, _m(e, "success", True), _m(e, "final_pos_err", scale=100, digits=2),
                             _m(e, "final_yaw_err", digits=2), f"{t / 60:.0f}" if t else "--"])
        out.append(_table("Learning algorithms on hover (nominal condition, 1000 episodes). Position error in cm, yaw "
                          "error in rad. Source: \\texttt{offpolicy\\_hover.json}.", "tab:fw_offpolicy",
                          ["Algorithm", "Success", "Pos. err [cm]", "Yaw err [rad]", "Train [min]"], rows, "lcccc"))
    v = L("vehicles_interfaces.json")
    if v:
        rows = []
        for k, lab in (("hexa_ctbr_ppo", "hexa\\_x (6 rotors, generated frame), CTBR, PPO"),
                       ("hexa_ctbr_cascade", "hexa\\_x, cascade baseline"),
                       ("hexa_motor_ppo", "hexa\\_x, motor-level (6-D action), PPO"),
                       ("accel_interface_ppo", "x500, acceleration interface (plugin example), PPO"),
                       ("landing_ppo_reward_v1", "landing task (plugin example), PPO, first reward"),
                       ("landing_ppo", "landing task, PPO, reward v2 (time penalty, low-and-slow term)"),
                       ("landing_cascade", "landing task, cascade baseline")):
            e = v.get(k)
            if e:
                rows.append([lab, _m(e, "success", True), _m(e, "final_pos_err", scale=100, digits=2),
                             _m(e, "crash", True)])
        out.append(_table("New vehicles, control interfaces and tasks defined through the extension points (1000 "
                          "episodes; position error in cm). Source: \\texttt{vehicles\\_interfaces.json}.",
                          "tab:fw_vehicles", ["Configuration", "Success", "Pos. err [cm]", "Crash"], rows, "lccc"))
    sr = L("safe_rl.json")
    if sr:
        rows = []
        for k, lab in (("unconstrained_ppo", "PPO (unconstrained)"), ("unconstrained_cascade", "cascade"),
                       ("ppo_lagrangian", "PPO-Lagrangian (PID multiplier)")):
            e = sr.get(k)
            if e:
                rows.append([lab, _m(e, "success", True), _m(e, "cost", digits=3), _m(e, "time_to_finish", digits=2),
                             _m(e, "crash", True)])
        out.append(_table("Constrained waypoint racing: speed $\\le$\\,\\SI{2}{m/s}, budget \\SI{0.1}{s} of over-speed "
                          "flight per episode (500 episodes). Source: \\texttt{safe\\_rl.json}.", "tab:fw_safe",
                          ["Policy", "Success", "Episode cost [s]", "Time [s]", "Crash"], rows, "lcccc"))
    for suite, cap, keys in (("race", "Drone racing (2 laps, 7 gates, 5-inch racer); trained on random circuits.",
                              ("success", "lap_time", "gates_passed", "crash")),
                             ("swarm", "Four-agent antipodal swap (shared policy, centralized critic).",
                              ("success", "agent_collision", "min_separation", "time_to_goal"))):
        bs = L(f"bench_{suite}.json")
        if not bs:
            continue
        rows = []
        for lab, r in bs["results"].items():
            for c in bs["conditions"]:
                e = r["per_condition"].get(c["name"])
                if e:
                    rows.append([lab.replace("_", r"\_"), c["name"].replace("_", r"\_") + f" ({c['split']})"]
                                + [_m(e, k, k in ("success", "crash", "agent_collision")) for k in keys])
        out.append(_table(cap + f" Source: \\texttt{{bench\\_{suite}.json}}.", f"tab:fw_{suite}",
                          ["Controller", "Condition"] + [k.replace("_", r"\_") for k in keys], rows, "llcccc"))
    il = L("imitation.json")
    if il:
        rows = []
        for k, lab in (("expert_ppo", "PPO teacher"), ("bc", "BC student"), ("bc_no_last_action", "BC student, no last action")):
            e = il["hover_distillation"].get(k)
            if e:
                rows.append(["hover", lab, _m(e, "success", True), "--", _m(e, "final_pos_err", scale=100, digits=2)])
        sym = il.get("langnav_symbolic", {})
        for tag, lab in (("expert", "scripted expert (privileged)"), ("langnav_sym_bc", "BC (with last action)"),
                         ("langnav_sym_bc_nolang", "BC, instruction removed"),
                         ("langnav_sym_bc_nolast", "BC, no last action"), ("langnav_sym_bc_v2", "+ unknown words ignored"),
                         ("langnav_sym_dagger1", "+ DAgger round 1"), ("langnav_sym_dagger2", "+ DAgger round 2"),
                         ("langnav_sym_dagger2_clip", "same data, frozen CLIP text encoder")):
            r = sym.get(tag, {})
            a, h = r.get("Quadsim-LangNav-v0"), r.get("Quadsim-LangNav-Heldout-v0")
            if a or h:
                rows.append(["langnav (map)", lab, _m(a, "success", True), _m(h, "success", True),
                             _m(a, "wrong_landmark", True)])
        rgb = il.get("langnav_rgb", {})
        for tag, lab in (("expert", "scripted expert (privileged)"), ("langnav_rgb_bc", "BC, FiLM-CNN + instruction"),
                         ("langnav_rgb_dagger1", "+ DAgger round 1"),
                         ("langnav_rgb_dagger1_clip", "same data, frozen CLIP text encoder"),
                         ("vla_color_servo", "VLA adapter: colour servo")):
            r = rgb.get(tag, {})
            a, h = r.get("Quadsim-LangNav-RGB-v0"), r.get("Quadsim-LangNav-RGB-Heldout-v0")
            if a or h:
                rows.append(["langnav (RGB)", lab, _m(a, "success", True), _m(h, "success", True),
                             _m(a, "wrong_landmark", True)])
        out.append(_table("Imitation learning. Hover: RL-to-BC distillation (last column: final position error [cm]). "
                          "LangNav: success with training / held-out instruction phrasings and wrong-landmark rate "
                          "(training phrasing). Source: \\texttt{imitation.json}.", "tab:fw_il",
                          ["Task", "Policy", "Success (train phr.)", "Success (held-out)", "Wrong / pos. err"], rows,
                          "llccc"))
    ex = L("examples_check.json")
    if ex:
        rows = [[k.replace("_", r"\_"), "pass" if r["pass"] else "FAIL",
                 f"{100 * r['metrics']['success']:.0f}\\,\\%" if r.get("metrics", {}).get("success") is not None
                 else "--"] for k, r in ex.items()]
        out.append(_table("Extension examples executed in the simulator (\\texttt{examples\\_check.py}); "
                          "success of the classical baseline where the check is an evaluation.", "tab:fw_examples",
                          ["Example", "Result", "Baseline success"], rows, "lcc"))
    w("fw_results.tex", "".join(out))


if __name__ == "__main__":
    framework()
    m2()
    depth()
    sitl()
    hover_v1()
    evals()
    bench()
    print(sorted(os.listdir(OUT)))
