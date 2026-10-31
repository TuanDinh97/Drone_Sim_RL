"""Plot a recording made with scripts/data/record.py (no Isaac Sim needed).

Usage: python scripts/data/plot_recording.py datasets/<file>.h5 [--env 0] [--out docs/figures/rec.png]
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from quadsim.data.recorder import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = load(a.file)
    e = a.env
    t = d["time"] - d["time"][0]
    done = d["terminated"][:, e] | d["truncated"][:, e] if "terminated" in d else np.zeros_like(t, bool)
    has_img = "image" in d
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(3, 4)
    ax = fig.add_subplot(gs[:, :2], projection="3d")
    p = d["true/pos"][:, e]
    est = d["est/odom"][:, e, :3]
    # split at resets
    idx = np.flatnonzero(done) + 1
    for seg_p, seg_e in zip(np.split(p, idx), np.split(est, idx)):
        if len(seg_p) > 1:
            ax.plot(*seg_p.T, lw=1.5)
            ax.plot(*seg_e.T, lw=0.6, ls="--", color="gray")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")  # noqa: E702
    ax.set_title(f"{d['attrs'].get('task', '')} / {d['attrs'].get('controller', '')}\n(solid: truth, dashed: estimate)")
    a1 = fig.add_subplot(gs[0, 2])
    a1.plot(t, d["sensors/imu"][:, e, :3]); a1.set_title("gyro [rad/s]")  # noqa: E702
    a2 = fig.add_subplot(gs[1, 2])
    a2.plot(t, d["sensors/imu"][:, e, 3:]); a2.set_title("accelerometer [m/s²]")  # noqa: E702
    a3 = fig.add_subplot(gs[2, 2])
    a3.plot(t, d["motor/omega"][:, e]); a3.set_title("rotor speed [rad/s]"); a3.set_xlabel("t [s]")  # noqa: E702
    a4 = fig.add_subplot(gs[0, 3])
    a4.plot(t, d["action"][:, e]); a4.set_title("normalized action")  # noqa: E702
    a5 = fig.add_subplot(gs[1, 3])
    a5.plot(t, d["true/pos"][:, e, 2], label="true z"); a5.plot(t, d["sensors/baro"][:, e, 0], label="baro", lw=0.7)  # noqa: E702
    a5.legend(fontsize=7); a5.set_title("altitude [m]")  # noqa: E702
    a6 = fig.add_subplot(gs[2, 3])
    if has_img:
        img = d["image"][len(t) // 2, e]
        if img.shape[0] == 1:
            a6.imshow(img[0] / 1000.0, cmap="viridis_r", vmin=0, vmax=10); a6.set_title("depth [m] (mid-episode)")  # noqa: E702
        else:
            a6.imshow(np.transpose(img[:3], (1, 2, 0))); a6.set_title("RGB (mid-episode)")  # noqa: E702
        a6.set_xticks([]); a6.set_yticks([])  # noqa: E702
    else:
        a6.plot(t, d["env/wind_w"][:, e]); a6.set_title("wind [m/s]")  # noqa: E702
    for x in (a1, a2, a3, a4, a5):
        for r in np.flatnonzero(done):
            x.axvline(t[r], color="k", lw=0.4, alpha=0.4)
    fig.tight_layout()
    out = a.out or a.file.replace(".h5", ".png")
    fig.savefig(out, dpi=110)
    print(json.dumps({"figure": out, "samples": len(t)}))


if __name__ == "__main__":
    main()
