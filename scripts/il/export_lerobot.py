"""Export a quadsim demo file (quadsim.il.dataset HDF5) to a LeRobot dataset and verify it with LeRobot's own loader.

LeRobot is an *optional* dependency, installed in a separate environment so that it cannot change the simulator's
pinned packages (it needs NumPy 2 / newer pyarrow):

    bash scripts/il/setup_lerobot_env.sh          # third_party/lerobot-venv (lerobot 0.4.4, dataset format v3.0)
    third_party/lerobot-venv/bin/python scripts/il/export_lerobot.py datasets/il/langnav_rgb_expert.h5 \
        --repo_id quadsim/langnav_rgb --out datasets/lerobot/langnav_rgb

Mapping: ``observation.state`` = the policy observation without instruction-token blocks, ``observation.images.front``
= onboard camera (HWC uint8; PNG frames or video with ``--videos``), ``action`` = normalized action, ``task`` = the
natural-language instruction (or the task name). Writes <out>/quadsim_export.json with a ``pass`` flag after reloading
the dataset with ``LeRobotDataset`` and comparing frames.
"""

import argparse
import json
import os
import shutil
import time

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("demo_file")
ap.add_argument("--repo_id", default="quadsim/demos")
ap.add_argument("--out", required=True)
ap.add_argument("--videos", action="store_true")
ap.add_argument("--max_episodes", type=int, default=None)
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()


def main():
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    f = h5py.File(args.demo_file, "r")
    layout = json.loads(f.attrs["obs_layout"])
    keep_cols, names, off = [], [], 0
    for n, d in layout:
        if n != "tokens":
            keep_cols += list(range(off, off + d))
            names += [f"{n}_{i}" for i in range(d)]
        off += d
    eps = sorted(f["episodes"])[: args.max_episodes]
    g0 = f["episodes"][eps[0]]
    a_dim = g0["action"].shape[1]
    fps = int(round(1.0 / float(f.attrs["step_dt"])))
    features = {"observation.state": {"dtype": "float32", "shape": (len(keep_cols),), "names": names},
                "action": {"dtype": "float32", "shape": (a_dim,), "names": [f"a{i}" for i in range(a_dim)]}}
    has_img = "image" in g0
    if has_img:
        c, h, w = g0["image"].shape[1:]
        features["observation.images.front"] = {"dtype": "video" if args.videos else "image", "shape": (h, w, c),
                                                "names": ["height", "width", "channels"]}
    if os.path.exists(args.out):
        if not args.overwrite:
            raise SystemExit(f"{args.out} exists (use --overwrite)")
        shutil.rmtree(args.out)
    t0 = time.time()
    ds = LeRobotDataset.create(args.repo_id, fps, features, root=args.out, robot_type="quadrotor",
                               use_videos=args.videos)
    task_name = str(f.attrs.get("task", "quadsim"))
    for name in eps:
        g = f["episodes"][name]
        obs, act = g["obs/policy"][()][:, keep_cols].astype(np.float32), g["action"][()].astype(np.float32)
        img = g["image"][()].transpose(0, 2, 3, 1) if has_img else None
        task = str(g.attrs.get("instruction", "")) or task_name
        for t in range(len(act)):
            fr = {"observation.state": obs[t], "action": act[t], "task": task}
            if has_img:
                fr["observation.images.front"] = img[t]
            ds.add_frame(fr)
        ds.save_episode()
    ds.finalize()
    wall = time.time() - t0

    # ---- verification with LeRobot's loader
    rd = LeRobotDataset(args.repo_id, root=args.out)
    rng = np.random.default_rng(0)
    errs, img_ok, task_ok = [], True, True
    ep_starts = np.cumsum([0] + [int(f["episodes"][e].attrs["length"]) for e in eps])
    for _ in range(min(50, rd.num_frames)):
        k = int(rng.integers(0, len(eps)))
        t = int(rng.integers(0, ep_starts[k + 1] - ep_starts[k]))
        item = rd[int(ep_starts[k] + t)]
        g = f["episodes"][eps[k]]
        errs.append(float(np.abs(item["observation.state"].numpy() - g["obs/policy"][t][keep_cols]).max()))
        errs.append(float(np.abs(item["action"].numpy() - g["action"][t]).max()))
        task_ok &= item["task"] == (str(g.attrs.get("instruction", "")) or task_name)
        if has_img:
            im = (item["observation.images.front"].numpy() * 255).round().astype(np.int16)  # CHW float [0, 1]
            img_ok &= bool(np.abs(im - g["image"][t].astype(np.int16)).max() <= (8 if args.videos else 0))
    rep = {"source": os.path.abspath(args.demo_file), "out": os.path.abspath(args.out), "repo_id": args.repo_id,
           "lerobot_codebase_version": CODEBASE_VERSION, "fps": fps, "episodes": rd.num_episodes,
           "frames": rd.num_frames, "expected_frames": int(ep_starts[-1]), "features": list(rd.features),
           "num_tasks": len(rd.meta.tasks), "max_abs_err_state_action": max(errs), "images_identical": img_ok,
           "tasks_match": bool(task_ok), "export_wall_s": wall}
    rep["pass"] = bool(rep["episodes"] == len(eps) and rep["frames"] == rep["expected_frames"]
                       and rep["max_abs_err_state_action"] < 1e-6 and img_ok and task_ok)
    with open(os.path.join(args.out, "quadsim_export.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
