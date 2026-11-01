"""Figure for the sim-vs-SITL comparison: 1 m step responses, hover collective / heave, roll-rate tracking.

Usage: PYTHONPATH=source/quadsim python scripts/sitl/plot_comparison.py --sitl "logs/sitl/curve_r*.npz" \
           --sitl_linear logs/sitl/default_r1.npz --quadsim logs/sitl/quadsim_nominal.npz \
           --out docs/figures/sitl_vs_quadsim.png
"""

import argparse
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from quadsim.deploy.interface import frd_to_flu, ned_to_enu  # noqa: E402

# categorical slots 1-3 (fixed order) + text/grid inks
QS, PX4, PX4_LIN, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"


def load_sitl(path):
    base = path[:-4]
    d = np.load(path)
    s = np.genfromtxt(base + ".bridge_steps.csv", delimiter=",", names=True)
    t0 = s["t"][0]
    goal = np.stack([s["goal_x"], s["goal_y"], s["goal_z"]], -1)
    i_x = np.nonzero(np.abs(goal[:, 0] - goal[0, 0]) > 1e-6)[0][0]
    i_z = np.nonzero(np.abs(goal[:, 2] - goal[0, 2]) > 1e-6)[0][0]
    o, g = d["odom"], d["gt_pos"]
    return {"g0": goal[0], "tx": s["t"][i_x] - t0, "tz": s["t"][i_z] - t0,
            "t_est": o[:, 2] - t0, "est": ned_to_enu(o[:, 3:6]),
            "t_true": g[:, 1] - t0, "true": ned_to_enu(g[:, 2:5]), "vz_true": -g[:, 7],
            "t_cmd": s["t"] - t0, "a0": s["a0"], "roll_cmd": s["a1"] * 6.0,
            "t_rate": o[:, 2] - t0, "roll": frd_to_flu(o[:, 13:16])[:, 0]}


def style(ax, title, ylabel, xlabel="time since step [s]"):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def window(t, a, b):
    return (t >= a) & (t <= b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sitl", nargs="+", required=True, help="SITL runs for the steps / rate panels")
    p.add_argument("--sitl_linear", required=True, help="SITL run with the linear thrust map (limit cycle)")
    p.add_argument("--quadsim", required=True, help="quadsim_scenario .npz (raw series of a few envs)")
    p.add_argument("--out", default="docs/figures/sitl_vs_quadsim.png")
    args = p.parse_args()
    runs = [load_sitl(f) for pat in args.sitl for f in sorted(glob.glob(pat))]
    lin = load_sitl(args.sitl_linear)
    q = np.load(args.quadsim)
    sched = json.load(open(args.quadsim[:-4] + ".json"))["schedule"]
    g0q = np.array(sched["goal0"])
    fig, axs = plt.subplots(3, 2, figsize=(10, 9.4), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    # ---- row 1: steps
    for ax, axis, key, label in ((axs[0, 0], 0, "tx", "x"), (axs[0, 1], 2, "tz", "z")):
        tq = q["t"] - (sched["t_step_x"] if axis == 0 else sched["t_step_z"])
        w = window(tq, -1, 5)
        for e in range(q["pos"].shape[1]):
            ax.plot(tq[w], q["pos"][w, e, axis] - g0q[axis], color=QS, lw=1.2, alpha=0.8,
                    label="quadsim (truth, 4 envs)" if e == 0 else None)
        for i, r in enumerate(runs):
            for t, y, ls, lab in ((r["t_est"], r["est"], "-", "PX4 SITL (EKF2 estimate)"),
                                  (r["t_true"], r["true"], ":", "PX4 SITL (gz ground truth)")):
                tt = t - r[key]
                w = window(tt, -1, 5)
                ax.plot(tt[w], y[w, axis] - r["g0"][axis], color=PX4, lw=1.4, ls=ls, label=lab if i == 0 else None)
        ax.plot([-1, 0, 0, 5], [0, 0, 1, 1], color=MUTED, lw=1.0, ls="--", label="goal")
        style(ax, f"1 m step in {label} (SITL: thrust-curve map)", f"{label} - {label}$_0$ [m]")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    # ---- row 2: hover collective and heave, 2 s
    t_a, t_b = 20.0, 22.0
    r = runs[0]
    for ax, key, ylab, title in ((axs[1, 0], "a0", "collective action a0", "hover: collective command"),
                                 (axs[1, 1], "vz", "vertical velocity [m/s]", "hover: vertical velocity (truth)")):
        if key == "a0":
            tc = q["t_cmd"]
            w = window(tc, t_a, t_b)
            ax.step(tc[w] - t_a, q["cmd"][w, 0, 0], where="post", color=QS, lw=1.2, label="quadsim")
            for d, c, lab in ((r, PX4, "PX4 SITL, thrust-curve map"), (lin, PX4_LIN, "PX4 SITL, linear map (run r1)")):
                w = window(d["t_cmd"], t_a, t_b)
                ax.step(d["t_cmd"][w] - t_a, d["a0"][w], where="post", color=c, lw=1.4, label=lab)
        else:
            w = window(q["t"], t_a, t_b)
            ax.plot(q["t"][w] - t_a, np.gradient(q["pos"][:, 0, 2], q["t"])[w], color=QS, lw=1.2, label="quadsim")
            for d, c, lab in ((r, PX4, "PX4 SITL, thrust-curve map"), (lin, PX4_LIN, "PX4 SITL, linear map (run r1)")):
                w = window(d["t_true"], t_a, t_b)
                ax.plot(d["t_true"][w] - t_a, d["vz_true"][w], color=c, lw=1.4, label=lab)
        style(ax, title, ylab, "time [s]")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.45 * (hi - lo))  # head-room for the legend
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    # ---- row 3: roll-rate tracking
    w = window(q["t"], t_a, t_b)
    wc = window(q["t_cmd"], t_a, t_b)
    ax = axs[2, 0]
    ax.step(q["t_cmd"][wc] - t_a, q["cmd"][wc, 0, 1] * 6.0, where="post", color=MUTED, lw=1.0, label="command")
    ax.plot(q["t"][w] - t_a, q["rate"][w, 0, 0], color=QS, lw=1.4, label="true body rate")
    style(ax, "quadsim: roll-rate command vs response (hover)", "roll rate [rad/s]", "time [s]")
    ax.legend(fontsize=8, frameon=False)
    ax = axs[2, 1]
    wc = window(r["t_cmd"], t_a, t_b)
    w = window(r["t_rate"], t_a, t_b)
    ax.step(r["t_cmd"][wc] - t_a, r["roll_cmd"][wc], where="post", color=MUTED, lw=1.0, label="command")
    ax.plot(r["t_rate"][w] - t_a, r["roll"][w], color=PX4, lw=1.4, label="EKF2 angular velocity")
    style(ax, "PX4 SITL (thrust-curve map): roll-rate command vs response", "roll rate [rad/s]", "time [s]")
    ax.legend(fontsize=8, frameon=False)
    lo = min(axs[2, 0].get_ylim()[0], axs[2, 1].get_ylim()[0])
    hi = max(axs[2, 0].get_ylim()[1], axs[2, 1].get_ylim()[1])
    for a in axs[2]:
        a.set_ylim(lo, hi)
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
