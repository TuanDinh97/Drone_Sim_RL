"""Classical baseline controllers that act through the *same* action interface as RL policies.

They only read deployable information (the onboard state estimate ``env.odom_meas``, IMU and task
targets), and their actions pass through the same delays, onboard loops and actuator models, so
comparisons with learned policies are fair.

* :class:`CascadeBaseline`   PX4-style position/velocity PID + attitude P (outputs CTBR or velocity actions)
* :class:`GeometricBaseline` Lee SE(3) controller with trajectory feed-forward (outputs CTBR actions)
* :class:`PotentialFieldBaseline` reactive obstacle avoidance from the range ring (velocity actions)
"""

from __future__ import annotations

import math

import torch

from quadsim.core import math as qm
from quadsim.core.controllers import (
    AttitudeController,
    AttitudeGains,
    PositionController,
    PositionGains,
    thrust_vector_to_setpoints,
)


def _targets(env):
    """(pos_d, vel_d, acc_d, yaw_d) from the task (env-local frame)."""
    if hasattr(env, "baseline_targets"):  # task hook (new tasks implement it instead of extending this function)
        return env.baseline_targets()
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
    """Lee et al. SE(3) position law with full acceleration feed-forward, for a 50 Hz offboard loop.

    ``f = m (-k_x e_x - k_v e_v + g e3 + a_d)`` defines the thrust vector; the desired attitude follows
    from (f, yaw_d) and is tracked by the attitude P loop -> body-rate set-points for the onboard rate PID.
    (A first version mapped Lee's *torque-level* attitude law to rate set-points at 50 Hz; with the
    delays in the loop it was unstable — 53 % crashes on the tracking task — see docs/PROGRESS.md.)
    """

    def __init__(self, env, kx=(4.0, 4.0, 6.0), kv=(3.0, 3.0, 4.0), max_tilt: float = 0.7,
                 att_gains: AttitudeGains | None = None):
        self.env = env
        dev = env.device
        self.kx, self.kv = torch.tensor(kx, device=dev), torch.tensor(kv, device=dev)
        self.max_tilt = max_tilt
        self.att = AttitudeController(att_gains or AttitudeGains(), dev)
        self.mass = env.params.mass

    def reset(self, env_ids=None):
        pass

    def __call__(self) -> torch.Tensor:
        env = self.env
        assert env.cfg.control.mode == "ctbr"
        est = env.odom_meas
        p, q, v = est[:, :3], qm.quat_normalize(est[:, 3:7]), est[:, 7:10]
        pos_d, vel_d, acc_d, yaw_d = _targets(env)
        acc = -self.kx * (p - pos_d) - self.kv * (v - vel_d) + acc_d
        acc[:, 2] = (acc[:, 2] + 9.81).clamp(min=0.2 * 9.81)
        max_xy = acc[:, 2:3] * math.tan(self.max_tilt)
        n_xy = acc[:, :2].norm(dim=-1, keepdim=True)
        acc[:, :2] = acc[:, :2] * torch.clamp(max_xy / n_xy.clamp_min(1e-9), max=1.0)
        collective, q_sp = thrust_vector_to_setpoints(self.mass * acc, q, yaw_d)
        return _ctbr_action(env, collective, self.att(q, q_sp))


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

from quadsim.registry import BASELINES as _REGISTRY  # noqa: E402

for _name, _cls in BASELINES.items():
    _REGISTRY.register(_name, _cls, overwrite=True)
