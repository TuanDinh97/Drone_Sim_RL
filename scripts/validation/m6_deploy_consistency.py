"""Milestone 6: deployment pipeline in the loop (hover, tracking, waypoint, range-ring navigation).

The simulator is flown by the *exported ONNX* policy whose observation is rebuilt every step by the
NumPy deployment code (:class:`quadsim.deploy.interface.ObservationBuilder` + the task helpers used by the
ROS 2 bridge: ``SinusoidReference``, ``waypoint_task``, ``nav_task``) from onboard quantities only (state
estimate, gyro, task definition, range ring) — exactly what a companion computer would do. Each step we compare
  (a) deploy-built observation vs the env's own policy observation, and
  (b) ONNX action vs the PyTorch policy action on the env observation,
and finally the closed-loop task metrics against the PyTorch policy (same seed).

Acceptance: max |obs_deploy - obs_env| < 1e-4, max |a_onnx - a_torch| < 1e-4, success / crash rate within
3 percentage points of the PyTorch policy (tasks without a success metric: crash only).

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
parser.add_argument("--out", default=None, help="result JSON (default logs/validation/m6_deploy_<task>.json)")
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.deploy.interface import ObservationBuilder, SinusoidReference  # noqa: E402
from quadsim.rl.overrides import apply_overrides  # noqa: E402
from quadsim.rl.policy_io import agent_cfg_for, latest_checkpoint, load_policy  # noqa: E402


def deploy_task_obs(u, b: ObservationBuilder, pos, q, contract):
    """Task observation rebuilt with the deployment code from the task definition + onboard estimate."""
    name, tp = type(u).__name__, contract.get("task_params", {})
    if name == "HoverEnv":
        return b.hover_task(pos, q, u.target_pos.cpu().numpy(), u.target_yaw.cpu().numpy())
    if name == "TrackingEnv":
        amp, freq, phase, cen = (x.cpu().numpy().astype(np.float64) for x in (u.amp, u.freq, u.phase, u.center))
        t = u.episode_time().cpu().numpy().astype(np.float64)
        hz = np.asarray(tp["horizon_s"])
        out = []
        for e in range(u.num_envs):
            ref = SinusoidReference(amp[e], freq[e], phase[e], cen[e])
            p_h, _, _ = ref(t[e] + hz)
            _, v0, _ = ref(t[e])
            out.append(b.track_task(pos[e], q[e], p_h, v0))
        return np.stack(out)
    if name == "WaypointEnv":
        wps, idx = u.waypoints.cpu().numpy().astype(np.float64), u.wp_idx.cpu().numpy()
        ar, w = np.arange(u.num_envs), tp["num_waypoints"]
        return b.waypoint_task(pos, q, wps[ar, idx], wps[ar, np.minimum(idx + 1, w - 1)], (w - idx) / w,
                               tp["max_obs_distance"])
    if name == "NavEnv":
        ring = u.range_ring.cpu().numpy() if u.cfg.vision.mode == "none" else None
        return b.nav_task(pos, q, u.goal.cpu().numpy(), ring, tp.get("range_max", 8.0))
    raise NotImplementedError(name)


def run(env, mode, pol, sess, builder, contract):
    u = env.unwrapped
    torch.manual_seed(args.seed)
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
            o = builder.build(builder.state_obs(pos, q, vel, gyro), deploy_task_obs(u, builder, pos, q, contract))
            max_obs_err = max(max_obs_err, float(np.abs(o - obs["policy"].cpu().numpy()).max()))
            a = sess.run(None, {"obs": o})[0]
            max_act_err = max(max_act_err, float(np.abs(a - a_torch.clamp(-1, 1).cpu().numpy()).max()))
            action = torch.tensor(a, device=u.device)
        else:
            action = a_torch
        obs, *_ = env.step(action)
    u.collect_episode_metrics = False
    recs = u.episode_metric_buffer
    res = {"episodes": int(sum(len(r["env_id"]) for r in recs))}
    for k in recs[0]:
        if k == "env_id":
            continue
        v = torch.cat([r[k] for r in recs]).float()
        v = v[torch.isfinite(v)]
        res[k] = float(v.mean()) if len(v) else float("nan")
    res.update(max_obs_err=max_obs_err, max_act_err=max_act_err)
    return res


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    apply_overrides(cfg, agent_cfg_for(args.task), overrides)
    env = gym.make(args.task, cfg=cfg)
    ckpt = args.checkpoint if args.checkpoint.endswith(".pt") else latest_checkpoint(args.checkpoint)
    pol = load_policy(ckpt, env, task=args.task)
    exp_dir = os.path.join(os.path.dirname(ckpt), "exported")
    contract = json.load(open(os.path.join(exp_dir, "contract.json")))
    sess = ort.InferenceSession(os.path.join(exp_dir, "policy.onnx"), providers=["CPUExecutionProvider"])
    builder = ObservationBuilder(contract)
    r_torch = run(env, "torch", pol, sess, builder, contract)
    r_onnx = run(env, "onnx", pol, sess, builder, contract)
    res = {"task": args.task, "checkpoint": ckpt, "overrides": overrides, "torch": r_torch,
           "onnx_deploy_pipeline": r_onnx}
    key = "success" if "success" in r_torch else "crash"
    res["pass"] = bool(r_onnx["max_obs_err"] < 1e-4 and r_onnx["max_act_err"] < 1e-4
                       and abs(r_onnx[key] - r_torch[key]) < 0.03 and abs(r_onnx["crash"] - r_torch["crash"]) < 0.03)
    print("[M6-DEPLOY] " + json.dumps(res, indent=2), flush=True)
    out = args.out or f"logs/validation/m6_deploy_{args.task}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
