"""Milestone 6: deployment pipeline in the loop.

The simulator is flown by the *exported ONNX* policy whose observation is rebuilt every step by the
NumPy :class:`quadsim.deploy.interface.ObservationBuilder` from onboard quantities only (state
estimate, gyro, target) — exactly what a companion computer would do. Each step we compare
  (a) deploy-built observation vs the env's own policy observation, and
  (b) ONNX action vs the PyTorch policy action on the env observation,
and finally the closed-loop task metrics against the PyTorch policy.

Acceptance: max |obs_deploy - obs_env| < 1e-4, max |a_onnx - a_torch| < 1e-4, success rate within
3 percentage points of the PyTorch policy.

Usage: python scripts/validation/m6_deploy_consistency.py --headless --checkpoint <run> [--task Quadsim-Hover-v0]
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Quadsim-Hover-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.deploy.interface import ObservationBuilder  # noqa: E402
from quadsim.rl.policy_io import latest_checkpoint, load_policy  # noqa: E402


def run(env, mode, pol, sess, builder):
    u = env.unwrapped
    u.seed(args.seed)
    obs, _ = env.reset()
    u.episode_metric_buffer.clear()
    u.collect_episode_metrics = True
    max_obs_err, max_act_err = 0.0, 0.0
    for _ in range(args.steps):
        with torch.no_grad():
            a_torch = pol(obs)
        if mode == "onnx":
            est = u.odom_meas.cpu().numpy().astype(np.float64)
            pos, q, vel = est[:, :3], est[:, 3:7], est[:, 7:10]
            q = q / np.linalg.norm(q, axis=-1, keepdims=True)
            gyro = u.imu_meas[:, :3].cpu().numpy()
            builder.last_action = u.actions.cpu().numpy()
            state = builder.state_obs(pos, q, vel, gyro)
            task = builder.hover_task(pos, q, u.target_pos.cpu().numpy(), u.target_yaw.cpu().numpy())
            o = builder.build(state, task)
            max_obs_err = max(max_obs_err, float(np.abs(o - obs["policy"].cpu().numpy()).max()))
            a = sess.run(None, {"obs": o})[0]
            max_act_err = max(max_act_err, float(np.abs(a - a_torch.clamp(-1, 1).cpu().numpy()).max()))
            action = torch.tensor(a, device=u.device)
        else:
            action = a_torch
        obs, *_ = env.step(action)
    u.collect_episode_metrics = False
    recs = u.episode_metric_buffer
    succ = torch.cat([r["success"] for r in recs]).mean().item()
    crash = torch.cat([r["crash"] for r in recs]).mean().item()
    err = torch.cat([r["final_pos_err"] for r in recs]).mean().item()
    return {"episodes": int(sum(len(r["env_id"]) for r in recs)), "success": succ, "crash": crash,
            "final_pos_err": err, "max_obs_err": max_obs_err, "max_act_err": max_act_err}


def main():
    assert args.task.startswith("Quadsim-Hover"), "deploy obs builder check implemented for the hover task"
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg)
    ckpt = args.checkpoint if args.checkpoint.endswith(".pt") else latest_checkpoint(args.checkpoint)
    pol = load_policy(ckpt, env, task=args.task)
    exp_dir = os.path.join(os.path.dirname(ckpt), "exported")
    contract = json.load(open(os.path.join(exp_dir, "contract.json")))
    sess = ort.InferenceSession(os.path.join(exp_dir, "policy.onnx"), providers=["CPUExecutionProvider"])
    builder = ObservationBuilder(contract)
    r_torch = run(env, "torch", pol, sess, builder)
    r_onnx = run(env, "onnx", pol, sess, builder)
    res = {"torch": r_torch, "onnx_deploy_pipeline": r_onnx}
    res["pass"] = bool(r_onnx["max_obs_err"] < 1e-4 and r_onnx["max_act_err"] < 1e-4
                       and abs(r_onnx["success"] - r_torch["success"]) < 0.03)
    print("[M6-DEPLOY] " + json.dumps(res, indent=2), flush=True)
    os.makedirs("logs/validation", exist_ok=True)
    with open("logs/validation/m6_deploy_consistency.json", "w") as f:
        json.dump(res, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
