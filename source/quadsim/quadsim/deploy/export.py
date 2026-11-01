"""Policy export (TorchScript + ONNX) with an explicit, machine-readable interface contract.

The exported graph contains the empirical observation normalizer, the actor network and the final
action clamp, so a deployment runtime only has to (1) build the observation exactly as described in
``contract.json`` and (2) decode the action with the contract's scaling.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn


class MlpPolicyExport(nn.Module):
    def __init__(self, normalizer: nn.Module, actor: nn.Module):
        super().__init__()
        self.normalizer = normalizer
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.actor(self.normalizer(obs)), -1.0, 1.0)


class CnnPolicyExport(nn.Module):
    def __init__(self, normalizer: nn.Module, encoder: nn.Module, actor: nn.Module):
        super().__init__()
        self.normalizer = normalizer
        self.encoder = encoder
        self.actor = actor

    def forward(self, obs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.normalizer(obs), self.encoder(image)], dim=-1)
        return torch.clamp(self.actor(x), -1.0, 1.0)


def task_params(env) -> dict:
    """Task constants a deployment needs to rebuild the task observation (quadsim.deploy.offboard)."""
    c, name = env.cfg, type(env).__name__
    if name == "TrackingEnv":
        return {"horizon_s": list(c.horizon_s)}
    if name == "WaypointEnv":
        return {"num_waypoints": c.num_waypoints, "reach_radius": c.reach_radius,
                "max_obs_distance": c.max_obs_distance}
    if name == "NavEnv":
        return {"range_rays": c.range_rays, "range_max": c.range_max, "goal_radius": c.goal_radius,
                "goal_dist_clip": 10.0, "vehicle_radius": c.vehicle_radius,
                "range_ring_frame": "yaw-aligned, CCW from heading, azimuth 2 pi k / range_rays"}
    return {}


def contract_from_env(env, task: str) -> dict:
    """Describe exactly how the policy observation/action are built in simulation."""
    c = env.cfg
    a_dim = env.action_dim
    layout = [
        ["lin_vel_b_est", 3, "estimated linear velocity in body FLU frame [m/s]"],
        ["rot_mat_cols01", 6, "first two columns of R_WB (row-major R[:, :2]); world z-up, body FLU"],
        ["gyro_b", 3, "IMU angular rate, body FLU [rad/s]"],
    ]
    if c.include_last_action:
        layout.append(["last_action", a_dim, "previous normalized action in [-1, 1]"])
    task_dim = env.task_obs_dim(c)
    task_desc = {
        "HoverEnv": "target position in est. body frame (3) [m], sin/cos of yaw error (2)",
        "TrackingEnv": "reference positions at t+horizon in est. body frame (3*H) [m], ref. velocity in body (3)",
        "WaypointEnv": "current waypoint rel. in body (3, clipped to 5 m), next-minus-current in body (3), "
                       "remaining fraction (1)",
        "NavEnv": "goal unit direction in body (3), min(dist,10)/10 (1)"
                  + (", range ring / 8 m (32, yaw-aligned, CCW from heading)" if c.vision.mode == "none" else ""),
    }.get(type(env).__name__, "task specific")
    layout.append(["task", task_dim, task_desc])
    for o in getattr(getattr(env, "terms", None), "policy_obs", []):  # configured extra (deployable) terms
        c_ = o.cfg
        layout.append([o.name, o.dim, f"observation term {getattr(c_.func, '__name__', c_.func)}{c_.params or ''}: "
                                      f"scale {c_.scale}, noise std {c_.noise_std}, every {c_.period_steps} policy "
                                      f"steps, {c_.delay_steps} steps latency"])
    mode = c.control.mode
    action = {
        "motor": "a_i in [-1,1] -> rotor thrust (a_i+1)/2 * T_max_rotor",
        "ctbr": "a0 -> collective thrust (lo + (a0+1)/2 (hi-lo)) * T_max_total [N]; a1..3 -> body rates FLU * max_body_rate",
        "ctatt": "a0 -> collective; a1,a2 -> roll, pitch * max_tilt; a3 -> yaw rate * max_yaw_rate",
        "velocity": "a0..2 -> velocity in yaw-aligned FLU frame * max_velocity [m/s]; a3 -> yaw rate * max_yaw_rate",
        "position": "a0..2 -> position set-point position_center + a * position_range (local frame) [m]; "
                    "a3 -> yaw rate * max_yaw_rate",
    }.get(mode, f"custom action interface '{mode}' (see its registered class)")
    p = env.params
    contract = {
        "task": task,
        "policy_rate_hz": 1.0 / env.step_dt,
        "onboard_loop_rate_hz": 1.0 / env.physics_dt,
        "frames": {"world": "z-up (ENU-like)", "body": "FLU", "quaternion": "wxyz, q_WB (body->world)"},
        "observation": {"dim": int(sum(x[1] for x in layout)), "layout": layout},
        "image": None,
        "action": {
            "dim": a_dim, "control_mode": mode, "semantics": action,
            "collective_range": list(c.control.collective_range),
            "max_total_thrust_N": p.max_total_thrust,
            "max_body_rate_rad_s": list(c.control.max_body_rate),
            "max_tilt_rad": c.control.max_tilt,
            "max_velocity_m_s": list(c.control.max_velocity),
            "max_yaw_rate_rad_s": c.control.max_yaw_rate,
            "hover_action": env.hover_action.tolist(),
        },
        "training_assumptions": {
            "action_delay_steps_physics": list(c.control.action_delay_steps),
            "odometry_noise": asdict(c.sensors.odometry), "imu": asdict(c.sensors.imu),
            "domain_randomization": c.randomization.to_dict() if hasattr(c.randomization, "to_dict") else str(c.randomization),
        },
        "vehicle": asdict(p),
        "task_params": task_params(env),
    }
    if c.vision.mode != "none":
        v = c.vision
        contract["image"] = {
            "mode": v.mode, "shape_chw": list(env.image.shape[1:]), "hfov_deg": v.hfov_deg,
            "camera_offset_body_m": list(v.offset_pos), "camera_pitch_deg": v.pitch_deg,
            "depth": f"distance to image plane / {v.depth_noise.max_range} m, clipped to [0,1]; invalid = 0; "
                     "no return = 1",
            "rgb": "linear [0,1], channels RGB",
        }
    return contract


def export_policy(policy_module, contract: dict, out_dir: str, example_obs: torch.Tensor,
                  example_image: torch.Tensor | None = None, opset: int = 17) -> dict:
    """Export an rsl_rl actor-critic (ActorCritic or ActorCriticCNN) or a quadsim off-policy actor
    (:class:`quadsim.rl.offpolicy.Actor`, normalizer inside) and verify ONNX == PyTorch."""
    os.makedirs(out_dir, exist_ok=True)
    pol = policy_module
    norm = getattr(pol, "actor_obs_normalizer", None)
    if norm is None:  # off-policy actor: normalization + tanh are part of its forward()
        model = MlpPolicyExport(nn.Identity(), pol).cpu().eval()
        inputs = (example_obs.cpu(),)
        names = ["obs"]
    elif hasattr(pol, "encoder"):
        model = CnnPolicyExport(norm, pol.encoder, pol.actor).cpu().eval()
        inputs = (example_obs.cpu(), example_image.cpu())
        names = ["obs", "image"]
    else:
        model = MlpPolicyExport(norm, pol.actor).cpu().eval()
        inputs = (example_obs.cpu(),)
        names = ["obs"]
    import copy

    model = copy.deepcopy(model)
    with torch.no_grad():
        ref = model(*inputs).numpy()
        traced = torch.jit.trace(model, inputs)
        traced.save(os.path.join(out_dir, "policy.pt"))
        onnx_path = os.path.join(out_dir, "policy.onnx")
        torch.onnx.export(model, inputs, onnx_path, input_names=names, output_names=["action"], opset_version=opset,
                          dynamic_axes={n: {0: "batch"} for n in names + ["action"]}, dynamo=False)
    report = {"onnx": onnx_path, "torchscript": os.path.join(out_dir, "policy.pt")}
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {n: x.numpy() for n, x in zip(names, inputs)})[0]
        ts = traced(*inputs).detach().numpy()
        report["max_abs_diff_onnx_vs_torch"] = float(np.abs(out - ref).max())
        report["max_abs_diff_torchscript_vs_torch"] = float(np.abs(ts - ref).max())
        report["verified"] = bool(report["max_abs_diff_onnx_vs_torch"] < 1e-5)
    except ImportError:
        report["verified"] = False
        report["note"] = "onnxruntime not installed"
    contract = dict(contract)
    contract["export"] = report
    with open(os.path.join(out_dir, "contract.json"), "w") as f:
        json.dump(contract, f, indent=2, default=str)
    return report
