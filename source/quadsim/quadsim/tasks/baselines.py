"""Classical baseline controllers that act through the *same* action interface as RL policies.

They only read deployable information (the onboard state estimate ``env.odom_meas``, IMU and task
targets), and their actions pass through the same delays, onboard loops and actuator models, so
comparisons with learned policies are fair.

* :class:`CascadeBaseline`   PX4-style position/velocity PID + attitude P (outputs CTBR or velocity actions)
* :class:`GeometricBaseline` Lee SE(3) controller with trajectory feed-forward (outputs CTBR actions)
* :class:`PotentialFieldBaseline` reactive obstacle avoidance from the range ring (velocity actions)
"""

from __future__ import annotations

import torch

from quadsim.core import math as qm
from quadsim.core.controllers import (
    AttitudeController,
    AttitudeGains,
    GeometricController,
    PositionController,
    PositionGains,
    thrust_vector_to_setpoints,
)


def _targets(env):
    """(pos_d, vel_d, acc_d, yaw_d) from the task (env-local frame)."""
    n, dev = env.num_envs, env.device
    z = torch.zeros(n, 3, device=dev)
    name = type(env).__name__
    if name == "HoverEnv":
        return env.target_pos, z, z, env.target_yaw
    if name == "TrackingEnv":
        pos, vel, acc = env.reference(env.episode_time()[:, None])
        return pos[:, 0], vel[:, 0], acc[:, 0], torch.zeros(n, device=dev)
    if name == "WaypointEnv":
        wp = env._wp(0)
        d = wp - env.est_pos()
        return wp, z, z, torch.atan2(d[:, 1], d[:, 0])
    if name == "NavEnv":
        d = env.goal - env.est_pos()
        return env.goal, z, z, torch.atan2(d[:, 1], d[:, 0])
    raise ValueError(f"no baseline targets for {name}")


def _ctbr_action(env, collective, rate_sp):
    lo, hi = env.cfg.control.collective_range
    a0 = 2.0 * (collective / env.t_max_nominal - lo) / (hi - lo) - 1.0
    return torch.cat([a0[:, None], rate_sp / env.max_rate], -1).clamp(-1, 1)


class CascadeBaseline:
    def __init__(self, env, gains: PositionGains | None = None, att_gains: AttitudeGains | None = None):
        self.env = env
        dt = env.step_dt  # runs at the policy rate (like an offboard/companion controller)
        self.pos = PositionController(env.params, env.num_envs, dt, gains or PositionGains(), env.device)
        self.att = AttitudeController(att_gains or AttitudeGains(), env.device)

    def reset(self, env_ids=None):
        self.pos.reset(env_ids)

    def __call__(self) -> torch.Tensor:
        env = self.env
        est = env.odom_meas
        p, q, v = est[:, :3], qm.quat_normalize(est[:, 3:7]), est[:, 7:10]
        pos_d, vel_d, acc_d, yaw_d = _targets(env)
        mode = env.cfg.control.mode
        if mode == "velocity":
            vel_sp = self.pos.kp_pos * (pos_d - p) + vel_d
            vel_sp = torch.maximum(torch.minimum(vel_sp, env.max_vel), -env.max_vel)
            yaw = qm.yaw_from_quat(q)
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            v_h = torch.stack([cy * vel_sp[:, 0] + sy * vel_sp[:, 1], -sy * vel_sp[:, 0] + cy * vel_sp[:, 1], vel_sp[:, 2]], -1)
            yaw_rate = 1.5 * qm.wrap_to_pi(yaw_d - yaw)
            return torch.cat([v_h / env.max_vel, (yaw_rate / env.cfg.control.max_yaw_rate)[:, None]], -1).clamp(-1, 1)
        thrust_w = self.pos(pos_d, p, v, vel_ff=vel_d, acc_ff=acc_d)
        collective, q_sp = thrust_vector_to_setpoints(thrust_w, q, yaw_d)
        rate_sp = self.att(q, q_sp)
        if mode == "ctbr":
            return _ctbr_action(env, collective, rate_sp)
        raise ValueError(f"CascadeBaseline does not support control mode {mode}")


class GeometricBaseline:
    def __init__(self, env):
        self.env = env
        self.ctrl = GeometricController(env.params, device=env.device)
        self.inertia = torch.tensor(env.params.inertia, device=env.device)
        self.kp_rate = torch.tensor(env.cfg.control.rate_gains.kp, device=env.device)

    def reset(self, env_ids=None):
        pass

    def __call__(self) -> torch.Tensor:
        env = self.env
        assert env.cfg.control.mode == "ctbr"
        est = env.odom_meas
        p, q, v = est[:, :3], qm.quat_normalize(est[:, 3:7]), est[:, 7:10]
        w = env.imu_meas[:, :3]
        pos_d, vel_d, acc_d, yaw_d = _targets(env)
        thrust, torque = self.ctrl(p, q, v, w, pos_d, vel_d, acc_d, yaw_d)
        # map the desired angular acceleration to a body-rate set-point for the onboard rate loop
        ang_acc = (torque - torch.cross(w, self.inertia * w, dim=-1)) / self.inertia
        rate_sp = w + ang_acc / self.kp_rate
        return _ctbr_action(env, thrust, rate_sp)


class PotentialFieldBaseline:
    """Attractive goal field + repulsion from the range-ring (LiDAR-like) readings, velocity mode."""

    def __init__(self, env, speed: float = 2.0, influence: float = 2.5, k_rep: float = 1.5):
        self.env, self.speed, self.influence, self.k_rep = env, speed, influence, k_rep

    def reset(self, env_ids=None):
        pass

    def __call__(self) -> torch.Tensor:
        env = self.env
        assert env.cfg.control.mode == "velocity"
        est = env.odom_meas
        p, q = est[:, :3], qm.quat_normalize(est[:, 3:7])
        yaw = qm.yaw_from_quat(q)
        goal = env.goal - p
        dist = goal.norm(dim=-1, keepdim=True)
        v_att = goal / dist.clamp_min(1e-6) * torch.clamp(dist, max=self.speed)
        ring = env.range_ring  # yaw-aligned frame ranges
        dirs = env.ring_dirs[:, :2]  # (R, 2) in yaw frame
        w = torch.clamp(1.0 / ring.clamp_min(0.05) - 1.0 / self.influence, min=0.0)  # (N, R)
        rep_yaw = -(w[..., None] * dirs[None]).sum(1) * self.k_rep
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        rep_w = torch.stack([cy * rep_yaw[:, 0] - sy * rep_yaw[:, 1], sy * rep_yaw[:, 0] + cy * rep_yaw[:, 1]], -1)
        v = v_att.clone()
        v[:, :2] += rep_w
        n_xy = v[:, :2].norm(dim=-1, keepdim=True)
        v[:, :2] = v[:, :2] * torch.clamp(self.speed / n_xy.clamp_min(1e-6), max=1.0)
        v_h = torch.stack([cy * v[:, 0] + sy * v[:, 1], -sy * v[:, 0] + cy * v[:, 1], v[:, 2]], -1)
        yaw_d = torch.atan2(v[:, 1], v[:, 0])
        yaw_rate = 2.0 * qm.wrap_to_pi(yaw_d - yaw)
        return torch.cat([v_h / env.max_vel, (yaw_rate / env.cfg.control.max_yaw_rate)[:, None]], -1).clamp(-1, 1)


BASELINES = {"cascade": CascadeBaseline, "geometric": GeometricBaseline, "potential_field": PotentialFieldBaseline}
