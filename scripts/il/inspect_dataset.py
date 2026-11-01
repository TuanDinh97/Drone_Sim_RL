"""Inspect / replay a demonstration file: summary, trajectories, actions and camera frames of an episode.

Usage: python scripts/il/inspect_dataset.py datasets/il/langnav_rgb_expert.h5 [--episode 0] [--out fig.png]
Prints quadsim.il.dataset.summarize() as JSON and writes a figure: top view of up to 50 episodes (colour = success),
altitude and actions of the selected episode, and (if recorded) a strip of its camera frames with the instruction.
"""

import argparse
import json
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "quadsim"))
from quadsim.il.dataset import summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--episode", type=int, default=0)
ap.add_argument("--out", default=None)
ap.add_argument("--frames", type=int, default=6)
args = ap.parse_args()

INK, MUTED, OK, BAD, GRID = "#0b0b0b", "#52514e", "#1baf7a", "#eb6834", "#e4e3df"


def main():
    s = summarize(args.file)
    print(json.dumps(s, indent=2))
    f = h5py.File(args.file, "r")
    names = sorted(f["episodes"])
    g = f["episodes"][names[args.episode]]
    has_img = "image" in g
    fig = plt.figure(figsize=(13, 7.5 if has_img else 4.2), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    gs = fig.add_gridspec(2 if has_img else 1, 3)
    ax = fig.add_subplot(gs[0, 0])
    for name in [names[i] for i in np.linspace(0, len(names) - 1, min(50, len(names))).astype(int)]:
        e = f["episodes"][name]
        p = e["true/pos"][()]
        ax.plot(p[:, 0], p[:, 1], color=OK if e.attrs.get("success", 0) > 0.5 else BAD, lw=0.8, alpha=0.6)
    p = g["true/pos"][()]
    ax.plot(p[:, 0], p[:, 1], color=INK, lw=2.0, label=f"episode {args.episode}")
    ax.set_aspect("equal")
    ax.set_title("top view (green = success)", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    t = np.arange(len(p)) * float(f.attrs["step_dt"])
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, p[:, 2], color=INK, lw=1.5)
    ax.set_title("altitude [m]", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("t [s]", color=MUTED)
    ax = fig.add_subplot(gs[0, 2])
    a = g["action"][()]
    for i in range(a.shape[1]):
        ax.plot(t, a[:, i], lw=1.2, label=f"a{i}")
    ax.set_title(f"actions ({f.attrs.get('control_mode', '')})", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, ncol=a.shape[1])
    for axx in fig.axes:
        axx.grid(color=GRID, lw=0.5)
        axx.tick_params(colors=MUTED, labelsize=8)
    if has_img:
        img = g["image"][()]
        idx = np.linspace(0, len(img) - 1, args.frames).astype(int)
        sub = gs[1, :].subgridspec(1, args.frames)
        for j, k in enumerate(idx):
            ax = fig.add_subplot(sub[0, j])
            ax.imshow(img[k].transpose(1, 2, 0))
            ax.set_title(f"t = {t[k]:.1f} s", fontsize=8, color=MUTED)
            ax.axis("off")
    fig.suptitle(f"{os.path.basename(args.file)} - episode {args.episode}: \"{g.attrs.get('instruction', '')}\" "
                 f"(success {g.attrs.get('success', float('nan')):.0f})", fontsize=11, color=INK)
    out = args.out or os.path.splitext(args.file)[0] + f"_ep{args.episode}.png"
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(f"[inspect] wrote {out}")


if __name__ == "__main__":
    main()
