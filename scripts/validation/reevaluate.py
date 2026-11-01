"""Re-run evaluations from their result JSONs (``scripts/rl/evaluate.py`` output) with the current code.

Usage: python scripts/validation/reevaluate.py --backup logs/_reeval_backup [--jobs 2] FILE.json [...]

The command is rebuilt from each file's metadata (task, checkpoint, controllers, seed, num_envs, episodes,
overrides, plugins, algorithm; older files lack some of these, see ``_infer``), the original file is copied to the
backup directory and deleted, and the evaluation writes a new file with the same name. A failed re-evaluation
therefore leaves no stale result. State-based evaluations run ``--jobs`` at a time, rendered-camera ones alone.
Writes ``logs/validation/reevaluate.json``: old vs new mean of every metric, per file and controller.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--backup", required=True, help="directory receiving the original files (relative paths kept)")
ap.add_argument("--jobs", type=int, default=2)
ap.add_argument("--timeout", type=float, default=2400.0)
ap.add_argument("--report", default="logs/validation/reevaluate.json")
args = ap.parse_args()
PY = sys.executable
ENV = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES",
       "PYTHONPATH": os.path.abspath("source/quadsim") + os.pathsep + os.environ.get("PYTHONPATH", "")}
RENDER = ("RGB", "Render")


def _infer(d):
    """Plugins / algorithm / num_envs of files written before evaluate.py recorded them."""
    task, ckpt, ov = d["task"], str(d.get("checkpoint") or ""), d.get("overrides", [])
    plugins, algo = list(d.get("plugins") or []), d.get("algo")
    if not plugins:
        if "Example-Land" in task:
            plugins = ["examples/custom_task.py"]
        elif "Example-Hover-GNSS" in task or "Example-Nav-Lidar" in task:
            plugins = ["examples/custom_sensor.py"]
        elif "Example-Hover-Curriculum" in task:
            plugins = ["examples/custom_reward_curriculum.py"]
        elif "env.control.mode=acceleration" in ov:
            plugins = ["examples/custom_action_interface.py"]
        elif "/ars/" in ckpt:
            plugins = ["examples/custom_algorithm.py"]
        elif any(o.startswith("env.dynamics.preset=") and not o.endswith((".yaml", ".json")) and
                 o.split("=", 1)[1] in ("quad_250", "hexa_550") for o in ov):
            plugins = ["examples/custom_vehicle.py"]
    if algo is None and ":" in ckpt and not ckpt.endswith(".pt"):
        algo = "vla"
    if algo is None and "/ars/" in ckpt:
        algo = "ars"
    n = d.get("num_envs")
    if n is None:  # the scripts of sessions 1-3 (final_evals.sh, dr_ablation.sh, regression_evals.sh)
        n = 100 if "Render" in task else (500 if d.get("episodes_requested", 500) >= 1000 else 250)
    return plugins, algo, n


def command(path):
    d = json.load(open(path))
    task = d["task"]
    name = os.path.basename(path)[:-5]
    if not name.startswith(task):
        raise ValueError(f"{path}: file name does not start with the task name")
    tag = name[len(task) + 1:]
    plugins, algo, n = _infer(d)
    cmd = [PY, "scripts/rl/evaluate.py", "--headless", "--task", task, "--controllers", *d["controllers"],
           "--episodes", str(d["episodes_requested"]), "--num_envs", str(n), "--seed", str(d.get("seed", 1234)),
           "--out_dir", os.path.dirname(path) or ".", "--tag", tag]
    if d.get("checkpoint"):
        cmd += ["--checkpoint", d["checkpoint"]]
    if algo:
        cmd += ["--algo", algo]
    if plugins:
        cmd += ["--plugins", *plugins]
    if any(r in task for r in RENDER):
        cmd += ["--enable_cameras"]
    return cmd + list(d.get("overrides", [])), d


def _means(d):
    return {c: {k: v["mean"] for k, v in m.items() if isinstance(v, dict) and "mean" in v}
            for c, m in d.get("controllers", {}).items()}


def main():
    jobs = []
    for f in args.files:
        cmd, old = command(f)
        jobs.append({"file": f, "cmd": cmd, "old": old, "render": any(r in old["task"] for r in RENDER)})
    report = json.load(open(args.report)) if os.path.exists(args.report) else {}
    queue = [j for j in jobs if not j["render"]] + [j for j in jobs if j["render"]]
    running = []

    def finish(j, rc):
        f = j["file"]
        ok = os.path.exists(f)  # exit codes of Isaac scripts are unreliable; the result file is written last
        entry = {"cmd": " ".join(j["cmd"]), "pass": ok, "wall_s": time.time() - j["t0"]}
        if ok:
            new = json.load(open(f))
            o, n = _means(j["old"]), _means(new)
            entry["old"], entry["new"] = o, n
            entry["n_old"] = {c: j["old"]["controllers"][c].get("_meta", {}).get("episodes") for c in o}
            entry["n_new"] = {c: new["controllers"][c].get("_meta", {}).get("episodes") for c in n}
        report[f] = entry
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"[reeval] {'ok  ' if ok else 'FAIL'} {f} ({entry['wall_s']:.0f} s)", flush=True)

    while queue or running:
        limit = 1 if (queue and queue[0]["render"]) or any(r["render"] for r in running) else args.jobs
        while queue and len(running) < limit and not (running and queue[0]["render"]):
            j = queue.pop(0)
            dst = os.path.join(args.backup, os.path.relpath(j["file"], "logs") if j["file"].startswith("logs/")
                               else j["file"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):  # keep the first (pre-fix) original if re-run twice
                shutil.copy2(j["file"], dst)
            os.remove(j["file"])
            log = j["file"][:-5] + ".reeval.log"
            j["t0"] = time.time()
            j["proc"] = subprocess.Popen(j["cmd"], stdout=open(log, "w"), stderr=subprocess.STDOUT, env=ENV,
                                         start_new_session=True)
            running.append(j)
        time.sleep(2)
        for j in list(running):
            rc = j["proc"].poll()
            if rc is None and time.time() - j["t0"] > args.timeout:
                os.killpg(j["proc"].pid, 9)
                rc = "timeout"
            if rc is not None:
                running.remove(j)
                finish(j, rc)
    print(f"[reeval] {sum(r['pass'] for r in report.values())}/{len(report)} pass -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
