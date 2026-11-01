"""Execute every extension example (examples/*.py, examples/vehicles/*.yaml) briefly and record pass flags.

Usage: python scripts/validation/examples_check.py [--only name ...]
Writes logs/validation/examples_check.json. Each check runs a real simulator process (train a few iterations or
evaluate a few episodes) and passes only if the process produced its expected artefact / metrics.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*", default=None)
args = ap.parse_args()
PY = sys.executable
ENV = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES",
       "PYTHONPATH": os.path.abspath("source/quadsim") + os.pathsep + os.environ.get("PYTHONPATH", "")}
OUT = "logs/validation/examples"
os.makedirs(OUT, exist_ok=True)


def run(name, cmd, timeout=900):
    log = os.path.join(OUT, f"{name}.log")
    t0 = time.time()
    with open(log, "w") as f:
        try:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    return log, rc, time.time() - t0


def train_check(name, task, plugin, extra=(), iters=5, algo=None):
    run_name = f"excheck_{name}"
    cmd = [PY, "scripts/rl/train.py", "--headless", "--task", task, "--plugins", plugin, "--num_envs", "256",
           "--max_iterations", str(iters), "--run_name", run_name, *extra] + (["--algo", algo] if algo else [])
    log, rc, wall = run(name, cmd)
    runs = sorted(glob.glob(f"logs/**/*_{run_name}", recursive=True))
    ckpts = glob.glob(os.path.join(runs[-1], "model_*.pt")) if runs else []
    txt = open(log).read()
    return {"cmd": " ".join(cmd), "wall_s": wall, "run_dir": runs[-1] if runs else None, "checkpoints": len(ckpts),
            "pass": bool(ckpts) and "Traceback" not in txt}


def eval_check(name, task, controller, extra=(), episodes=100, num_envs=100, plugins=(), key="success", tag=None):
    tag = tag or f"excheck_{name}"
    cmd = [PY, "scripts/rl/evaluate.py", "--headless", "--task", task, "--controllers", controller, "--episodes",
           str(episodes), "--num_envs", str(num_envs), "--out_dir", OUT, "--tag", tag, *extra]
    if plugins:
        cmd += ["--plugins", *plugins]
    log, rc, wall = run(name, cmd)
    res = os.path.join(OUT, f"{task}_{tag}.json")
    out = {"cmd": " ".join(cmd), "wall_s": wall, "pass": False}
    if os.path.exists(res):
        m = next(iter(json.load(open(res))["controllers"].values()))
        out["metrics"] = {k: v["mean"] for k, v in m.items() if isinstance(v, dict) and "mean" in v}
        out["pass"] = out["metrics"].get("episode_length_s", 0) > 0
    return out


CHECKS = {
    "vehicle_yaml_hexa": lambda: eval_check("vehicle_yaml_hexa", "Quadsim-Hover-v0", "cascade",
                                            ["env.dynamics.preset=examples/vehicles/hexa_550.yaml"]),
    "vehicle_yaml_inherit": lambda: eval_check("vehicle_yaml_inherit", "Quadsim-Hover-v0", "cascade",
                                               ["env.dynamics.preset=examples/vehicles/x500_heavy.yaml"]),
    "vehicle_plugin_quad250": lambda: eval_check("vehicle_plugin_quad250", "Quadsim-Hover-v0", "cascade",
                                                 ["env.dynamics.preset=quad_250"],
                                                 plugins=["examples/custom_vehicle.py"]),
    "sensor_gnss": lambda: train_check("sensor_gnss", "Quadsim-Example-Hover-GNSS-v0", "examples/custom_sensor.py"),
    "task_landing": lambda: train_check("task_landing", "Quadsim-Example-Land-v0", "examples/custom_task.py"),
    "task_landing_baseline": lambda: eval_check("task_landing_baseline", "Quadsim-Example-Land-v0", "cascade",
                                                plugins=["examples/custom_task.py"]),
    "reward_curriculum": lambda: train_check("reward_curriculum", "Quadsim-Example-Hover-Curriculum-v0",
                                             "examples/custom_reward_curriculum.py"),
    "action_interface_accel": lambda: train_check("action_interface_accel", "Quadsim-Hover-v0",
                                                  "examples/custom_action_interface.py",
                                                  ["env.control.mode=acceleration"]),
    "algorithm_ars": lambda: train_check("algorithm_ars", "Quadsim-Hover-v0", "examples/custom_algorithm.py",
                                         ["agent.rollout_steps=50", "agent.envs_per_direction=8"], iters=3,
                                         algo="ars"),
    "vla_adapter_color_servo": lambda: eval_check(
        "vla_adapter_color_servo", "Quadsim-LangNav-RGB-v0", "policy",
        ["--enable_cameras", "--algo", "vla", "--checkpoint", "examples/vla_color_servo.py:make_adapter"],
        episodes=100, num_envs=64),
}


def main():
    path = "logs/validation/examples_check.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name, fn in CHECKS.items():
        if args.only and name not in args.only:
            continue
        print(f"[examples] {name} ...", flush=True)
        res[name] = fn()
        print(f"[examples] {name}: pass={res[name]['pass']} "
              f"{res[name].get('metrics', {}).get('success', '')} ({res[name]['wall_s']:.0f} s)", flush=True)
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
    print(f"[examples] {sum(r['pass'] for r in res.values())}/{len(res)} pass -> {path}")


if __name__ == "__main__":
    main()
