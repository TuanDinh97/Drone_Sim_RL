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
    files = [(f.split("eval_nav_")[1][:-5].replace("_", " "), os.path.basename(f)) for f in sorted(glob.glob(f"{R}/eval_nav_*.json"))]
    body = eval_table(files, nc, "nav")
    w("nav.tex", "\\begin{tabular}{llccccc}\\toprule\nPolicy input / eval & Controller & success [\\%] & collision [\\%] & time [s] & "
      "min clear. [cm] & episodes\\\\\\midrule\n" + body + "\n\\bottomrule\\end{tabular}\n")
    files = [(f.split("eval_dr_ablation_")[1][:-5].replace("_", " "), os.path.basename(f)) for f in sorted(glob.glob(f"{R}/eval_dr_ablation_*.json"))]
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


if __name__ == "__main__":
    m2()
    depth()
    evals()
    bench()
    print(sorted(os.listdir(OUT)))
