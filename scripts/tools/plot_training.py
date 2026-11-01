"""Training curves (from TensorBoard logs of scripts/rl/train.py) for the report.

Usage: python scripts/tools/plot_training.py  -> docs/figures/training_curves.{pdf,png}
Metrics are those logged by the environments at episode end (stochastic training policy).
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

RUNS = {
    "hover (PPO, DR)": "logs/rsl_rl/quadsim_hover/*_ctbr_s42",
    "hover (PPO, no DR)": "logs/rsl_rl/quadsim_hover/*_noDR_s42",
    "tracking": "logs/rsl_rl/quadsim_track/*_track_ctbr_s42",
    "waypoints": "logs/rsl_rl/quadsim_waypoint/*_waypoint_ctbr_s42",
    "nav range, reward v1": "logs/rsl_rl/quadsim_nav_range/*_nav_range_s42",
    "nav range, reward v2": "logs/rsl_rl/quadsim_nav_range/*_nav_range_v2_s42",
    "nav depth (CNN), reward v2": "logs/rsl_rl/quadsim_nav_depth/*_nav_depth_v2_s42",
}


def scalars(run_glob, *tags):
    dirs = sorted(glob.glob(run_glob))
    if not dirs:
        return None
    ev = glob.glob(os.path.join(dirs[-1], "events*"))
    if not ev:
        return None
    ea = EventAccumulator(ev[0], size_guidance={"scalars": 0})
    ea.Reload()
    for t in tags:
        if t in ea.Tags()["scalars"]:
            v = ea.Scalars(t)
            return np.array([x.step for x in v]), np.array([x.value for x in v])
    return None


def smooth(y, k=15):
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def main():
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for name in ("hover (PPO, DR)", "hover (PPO, no DR)"):
        d = scalars(RUNS[name], "Metrics/final_pos_err")
        if d:
            ax[0].semilogy(d[0], smooth(d[1]), label=name)
    ax[0].set_title("hover: final position error [m]")
    d = scalars(RUNS["tracking"], "Metrics/track_rmse")
    if d:
        ax[1].plot(d[0], smooth(d[1]), color="C2")
    ax[1].set_title("tracking: RMSE [m]")
    ax[1].set_ylim(0, 0.6)
    d = scalars(RUNS["waypoints"], "Metrics/success", "Metrics/success_rate")
    if d:
        ax[2].plot(d[0], smooth(d[1]), color="C3", label="success")
    d = scalars(RUNS["waypoints"], "Metrics/crash")
    if d:
        ax[2].plot(d[0], smooth(d[1]), color="C3", ls="--", label="crash")
    ax[2].set_title("waypoints")
    ax[2].legend(fontsize=7)
    for i, name in enumerate(("nav range, reward v1", "nav range, reward v2", "nav depth (CNN), reward v2")):
        s = scalars(RUNS[name], "Metrics/success")
        c = scalars(RUNS[name], "Metrics/collision")
        if s:
            ax[3].plot(s[0], smooth(s[1]), color=f"C{4 + i}", label=f"{name}: success")
        if c:
            ax[3].plot(c[0], smooth(c[1]), color=f"C{4 + i}", ls="--", label=f"{name}: collision")
    ax[3].set_title("obstacle navigation")
    ax[3].legend(fontsize=6)
    ax[0].legend(fontsize=7)
    for a in ax:
        a.set_xlabel("PPO iteration")
        a.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig("docs/figures/training_curves.pdf")
    fig.savefig("docs/figures/training_curves.png", dpi=110)
    print("docs/figures/training_curves.pdf")


if __name__ == "__main__":
    main()
