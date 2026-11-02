"""Write a Hugging Face / LeRobot dataset card (README.md) for a dataset exported by scripts/il/export_lerobot.py.

Usage: python scripts/il/dataset_card.py datasets/lerobot/<name> [--demo_file datasets/il/<demos>.h5]
Called automatically at the end of export_lerobot.py. Reads the LeRobot metadata (meta/info.json), the export report
(quadsim_export.json) and the quadsim demonstration file (provenance: task, expert, seed, observation layout, vehicle,
code version) and writes <dir>/README.md with the YAML front matter used on the Hugging Face hub
(``task_categories: robotics``, ``tags: LeRobot``). Pure Python + h5py: runs in the simulator or the LeRobot env.
"""

import argparse
import json
import os
import subprocess

CAMERA_FIX = "c3de0df"  # "Rendered cameras: an episode's first image shows its own scene" (session-5 review)

CONTROL = {
    "velocity": "velocity set-point in the yaw-aligned body frame (x, y, z) normalized by max_velocity, plus yaw rate "
                "normalized by max_yaw_rate, all in [-1, 1]; tracked by the onboard PX4-like velocity/attitude/rate loops",
    "ctbr": "collective thrust and body rates (roll, pitch, yaw) normalized to [-1, 1]; tracked by the onboard rate PID",
    "ctatt": "collective thrust, roll and pitch angles and yaw rate normalized to [-1, 1]",
    "position": "position set-point in a box around position_center plus yaw rate, normalized to [-1, 1]",
    "motor": "per-rotor thrust normalized to [-1, 1] (0 .. max rotor thrust)",
}


def _has_camera_fix(commit: str) -> bool | None:
    """Whether data collected at ``commit`` includes the camera re-render fix (None: unknown)."""
    try:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", CAMERA_FIX, commit.split("-")[0]],
                           cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return {0: True, 1: False}.get(r.returncode)


def stale_frame_note(images: bool, commit: str) -> str:
    if not images:
        return ""
    fixed = _has_camera_fix(commit) if commit and commit != "unknown" else None
    if fixed:
        return ""
    if fixed is False:
        return (f" This dataset was collected before quadsim's camera re-render fix (commit `{CAMERA_FIX}`): the first "
                "image of every episode still shows the previous episode's scene (about 1 of 145 frames).")
    return (f" Datasets collected before quadsim's camera re-render fix (commit `{CAMERA_FIX}`) contain a stale first "
            "image in every episode (about 1 of 145 frames).")


def card(export_dir: str, demo_file: str | None = None) -> str:
    info = json.load(open(os.path.join(export_dir, "meta", "info.json")))
    rep_path = os.path.join(export_dir, "quadsim_export.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    demo_file = demo_file or rep.get("source")
    attrs = {}
    if demo_file and os.path.exists(demo_file):
        import h5py

        with h5py.File(demo_file, "r") as f:
            attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    layout = json.loads(attrs["obs_layout"]) if "obs_layout" in attrs else []
    dropped = set(rep.get("dropped_blocks", []))
    state_blocks = ", ".join(f"`{n}` ({d})" for n, d in layout if n != "tokens" and n not in dropped)
    vehicle = json.loads(attrs["vehicle"])["name"] if "vehicle" in attrs else "x500"
    mode = str(attrs.get("control_mode", ""))
    images = "observation.images.front" in info.get("features", {})
    repo_id = rep.get("repo_id", os.path.basename(os.path.normpath(export_dir)))
    expert = attrs.get("expert", "")
    executor = attrs.get("executor", "")
    collection = (f"DAgger: the learner `{executor}` flies, the expert `{expert}` labels every state"
                  if executor else f"expert `{expert}` (successful episodes kept)")
    lines = [
        "---",
        "license: bsd-3-clause",
        "task_categories:",
        "- robotics",
        "tags:",
        "- LeRobot",
        "- quadsim",
        "- quadrotor",
        "- drone",
        "- simulation",
        *(["- vision-language-action"] if images else []),
        "configs:",
        "- config_name: default",
        "  data_files: data/*/*.parquet",
        "---",
        "",
        f"# {repo_id}",
        "",
        f"Synthetic {'vision-language ' if images else ''}navigation demonstrations of a simulated **{vehicle}** "
        f"quadrotor, generated with [quadsim](https://github.com/TuanDinh97/Drone_Sim_RL) (Isaac Lab / Isaac Sim) "
        f"on the task `{attrs.get('task', '?')}` and converted to the LeRobot dataset format "
        f"(codebase {info.get('codebase_version')}). Every episode carries its natural-language instruction as the "
        "LeRobot `task`.",
        "",
        "## Contents",
        "",
        "| | |",
        "|---|---|",
        f"| episodes / frames | {info.get('total_episodes')} / {info.get('total_frames')} |",
        f"| frame rate | {info.get('fps')} Hz (policy rate) |",
        f"| distinct instructions | {info.get('total_tasks')} |",
        f"| robot type | {info.get('robot_type')} |",
        "",
        "## Features",
        "",
        f"- `observation.state`: the deployable policy observation without the instruction-token block"
        f"{''.join(f' and `{d}`' for d in sorted(dropped))}: {state_blocks} (state estimate in the body frame and gyro"
        f"{'' if 'last_action' in dropped else ', previous action'}; see quadsim's `obs_layout`).",
        *([f"- `observation.images.front`: onboard RGB camera, {info['features']['observation.images.front']['shape']} "
           "(H, W, C), rendered with RTX (procedural scene, no texture/lighting randomization)."] if images else []),
        f"- `action`: {CONTROL.get(mode, 'normalized action of the quadsim action interface')}.",
        "- `task`: the instruction string (training templates; held-out phrasings are evaluated in the simulator).",
        "",
        "## Provenance",
        "",
        f"- Collection: {collection}; collection seed {attrs.get('seed', '?')}, "
        f"{attrs.get('episodes_seen', '?')} episodes simulated, {attrs.get('num_episodes', '?')} kept.",
        f"- quadsim commit: `{attrs.get('git_commit', 'unknown')}`; the full environment configuration is stored "
        "in the source HDF5 file (`config` attribute).",
        f"- Export verified by reloading with LeRobot's loader: `pass = {rep.get('pass')}` "
        f"(state/action max error {rep.get('max_abs_err_state_action')}, images identical: "
        f"{rep.get('images_identical')}).",
        "",
        "## Intended use and limitations",
        "",
        "Training and evaluating imitation-learning / vision-language-action policies for drones in simulation; the "
        "policies can be evaluated in closed loop in quadsim (`scripts/rl/evaluate.py --algo vla`; LeRobot policies "
        "trained on this dataset through `scripts/il/serve_lerobot_policy.py`). The data are synthetic: procedural pillars, idealized rendering, a simulated "
        "flight controller and sensor models; no real flight data." + stale_frame_note(images, attrs.get("git_commit", "")),
        "",
        "## Citation",
        "",
        "See `CITATION.cff` in the quadsim repository.",
        "",
    ]
    return "\n".join(lines)


def write_card(export_dir: str, demo_file: str | None = None) -> str:
    path = os.path.join(export_dir, "README.md")
    with open(path, "w") as f:
        f.write(card(export_dir, demo_file))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--demo_file", default=None)
    a = ap.parse_args()
    print(f"[card] wrote {write_card(a.export_dir, a.demo_file)}")
