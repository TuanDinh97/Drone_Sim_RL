"""Figure: hover benchmark suite - success per evaluation condition for each controller (95 % CI).

Usage: python scripts/tools/plot_framework.py  ->  docs/figures/framework_hover_benchmark.png
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = "docs/results/framework/bench_hover_algos.json"
OUT = "docs/figures/framework_hover_benchmark.png"
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("ppo", "PPO", "#2a78d6", "o"), ("sac", "SAC", "#eb6834", "s"),
          ("bc_distilled", "BC (distilled from PPO)", "#1baf7a", "^"), ("cascade", "cascade PID", "#eda100", "D")]
NAMES = {"nominal": "nominal\n(training dist.)", "wind_push": "wind + pushes", "harsh_dynamics": "harsh dynamics",
         "sensor_noise": "sensor noise x2-3", "latency": "latency 20-30 ms"}


def main():
    b = json.load(open(SRC))
    conds = [c["name"] for c in b["conditions"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for j, (key, label, col, mk) in enumerate(SERIES):
        r = b["results"][key]["per_condition"]
        xs = [i + (j - 1.5) * 0.12 for i in range(len(conds))]
        ys = [100 * r[c]["success"]["mean"] for c in conds]
        es = [100 * (r[c]["success"].get("ci95") or 0) for c in conds]
        ax.errorbar(xs, ys, yerr=es, fmt=mk, color=col, ms=8, lw=0, elinewidth=1.5, capsize=0,
                    mec=SURF, mew=1.5, label=label, zorder=3)
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(0.55, 61, "out of distribution", color=MUTED, fontsize=9, va="bottom")
    ppo_lat = 100 * b["results"]["ppo"]["per_condition"]["latency"]["success"]["mean"]
    ax.annotate(f"PPO {ppo_lat:.0f} %", (len(conds) - 1 - 0.18, ppo_lat), xytext=(-62, -4),
                textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.set_xticks(range(len(conds)), [NAMES.get(c, c) for c in conds], color=INK, fontsize=9)
    ax.set_ylim(60, 101)
    ax.set_ylabel("success [%]", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    ax.set_title("Hover benchmark: success per condition (1000 episodes each, 95 % CI)", loc="left",
                 color=INK, fontsize=11, pad=26)
    fig.savefig(OUT, dpi=160, facecolor=SURF)
    print(f"[plot] wrote {OUT}")


if __name__ == "__main__":
    main()
