"""Top view of the navigation layouts with the PX4 SITL trajectories (default vs matched velocity loop).

Usage: PYTHONPATH=source/quadsim python3 scripts/sitl/plot_nav.py --out docs/figures/sitl_nav_layouts.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from quadsim.deploy.interface import ned_to_enu  # noqa: E402

PX4_DEF, PX4_MATCH, INK, MUTED, GRID, OBST = "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#b8b6ae"


def run_track(path):
    d = np.load(path)
    m = json.load(open(path[:-4] + ".json"))
    s = np.genfromtxt(path[:-4] + ".bridge_steps.csv", delimiter=",", names=True)
    ev, t0 = m["events"], s["t"][0]
    t_end = (ev.get("goal_reached") or ev.get("collision") or ev["handback"]) - t0
    g = d["gt_pos"]
    tg = g[:, 1] - t0
    w = (tg >= 0) & (tg <= t_end)
    nav = m["nav"]
    # back into layout coordinates (inverse of the rigid hand-over transform) so both runs share one frame
    dx, dy, dyaw = nav["transform"]
    st = np.asarray(nav["start_layout"][:2])
    p = ned_to_enu(g[w, 2:5])[:, :2] - st - [dx, dy]
    c, s_ = np.cos(-dyaw), np.sin(-dyaw)
    p = np.stack([c * p[:, 0] - s_ * p[:, 1], s_ * p[:, 0] + c * p[:, 1]], -1) + st
    return p, ("goal" if "goal_reached" in ev else "collision" if "collision" in ev else "timeout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", default="logs/sitl/nav_layouts.json")
    ap.add_argument("--quadsim", default="logs/sitl/quadsim_nav_range_v2.json")
    ap.add_argument("--out", default="docs/figures/sitl_nav_layouts.png")
    args = ap.parse_args()
    lay = json.load(open(args.layouts))["layouts"]
    q = json.load(open(args.quadsim))["per_layout"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    for i, ax in enumerate(axs.flat):
        L = lay[i]
        for cx, cy, r, h in L["cylinders"]:
            ax.add_patch(Circle((cx, cy), r + 0.35, color=GRID, lw=0))  # collision margin (vehicle radius)
            ax.add_patch(Circle((cx, cy), r, color=OBST, lw=0))
        ax.add_patch(Circle(L["goal"][:2], 0.5, fill=False, ec=INK, lw=1.0, ls="--"))
        ax.plot(*L["start"][:2], "o", color=INK, ms=4)
        for f, col, lab in ((f"logs/sitl/nav_l{i}_r1.npz", PX4_DEF, "PX4 default velocity loop"),
                            (f"logs/sitl/nav_matched_l{i}_r1.npz", PX4_MATCH, "PX4 velocity loop matched")):
            p, outc = run_track(f)
            ax.plot(p[:, 0], p[:, 1], color=col, lw=1.6, label=f"{lab}: {outc}")
            if outc == "collision":
                ax.plot(p[-1, 0], p[-1, 1], "x", color=col, ms=8, mew=2)
        ql = q[str(i)]
        ax.set_title(f"layout {i}: quadsim {100 * ql['success']:.0f} % success", loc="left", fontsize=10, color=INK)
        ax.set_xlim(-1, 13)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.5)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    axs[1, 0].set_xlabel("x [m]", color=MUTED)
    axs[1, 0].set_ylabel("y [m]", color=MUTED)
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
