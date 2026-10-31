"""Batched classical flight controllers (PX4-style cascade and geometric SE(3)).

All controllers act on the *nominal* vehicle parameters (they never see randomized
values) and run at the physics rate inside the environments. Outputs flow down the
cascade: position -> velocity -> (thrust vector, attitude) -> body rates -> torques -> mixer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..math import quat_conjugate, quat_log_rotvec, quat_mul, quat_rotate, quat_rotate_inverse, rotmat_to_quat, vee
from ..math import quat_to_rotmat
from ..params import QuadrotorParams

GRAVITY = 9.81


@dataclass
class RateGains:
    kp: tuple[float, float, float] = (25.0, 25.0, 12.0)
    ki: tuple[float, float, float] = (5.0, 5.0, 2.0)
    kd: tuple[float, float, float] = (0.4, 0.4, 0.0)
    i_limit: float = 2.0  # rad/s * s (integrator clamp on angular-acceleration units)


class RateController:
    """Body-rate PID producing torques ``tau = I (Kp e + Ki int(e) - Kd w_dot) + w x I w``.

    Derivative acts on the measurement (as in PX4) using a first-order low-pass filtered
    finite difference to avoid kick on set-point changes.
    """

    def __init__(self, params: QuadrotorParams, num_envs: int, dt: float, gains: RateGains | None = None,
                 device="cpu", d_cutoff_hz: float = 30.0):
        g = gains or RateGains()
        self.inertia = torch.tensor(params.inertia, device=device)
        self.kp = torch.tensor(g.kp, device=device)
        self.ki = torch.tensor(g.ki, device=device)
        self.kd = torch.tensor(g.kd, device=device)
        self.i_limit = g.i_limit
        self.dt = dt
        self.alpha = dt / (dt + 1.0 / (2 * torch.pi * d_cutoff_hz))
        self.integral = torch.zeros(num_envs, 3, device=device)
        self.prev_w = torch.zeros(num_envs, 3, device=device)
        self.w_dot_f = torch.zeros(num_envs, 3, device=device)

    def reset(self, env_ids=None, ang_vel: torch.Tensor | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self.integral[ids] = 0.0
        self.w_dot_f[ids] = 0.0
        self.prev_w[ids] = 0.0 if ang_vel is None else ang_vel

    def __call__(self, rate_sp: torch.Tensor, ang_vel: torch.Tensor) -> torch.Tensor:
        err = rate_sp - ang_vel
        self.integral = torch.clamp(self.integral + err * self.dt, -self.i_limit, self.i_limit)
        w_dot = (ang_vel - self.prev_w) / self.dt
        self.w_dot_f = self.w_dot_f + self.alpha * (w_dot - self.w_dot_f)
        self.prev_w = ang_vel.clone()
        ang_acc = self.kp * err + self.ki * self.integral - self.kd * self.w_dot_f
        return self.inertia * ang_acc + torch.cross(ang_vel, self.inertia * ang_vel, dim=-1)


@dataclass
class AttitudeGains:
    kp: tuple[float, float, float] = (8.0, 8.0, 4.0)
    max_rate: tuple[float, float, float] = (6.0, 6.0, 3.0)


class AttitudeController:
    """Quaternion P controller -> body-rate set-point (tilt prioritized via reduced yaw gain)."""

    def __init__(self, gains: AttitudeGains | None = None, device="cpu"):
        g = gains or AttitudeGains()
        self.kp = torch.tensor(g.kp, device=device)
        self.max_rate = torch.tensor(g.max_rate, device=device)

    def __call__(self, quat: torch.Tensor, quat_sp: torch.Tensor, yaw_rate_ff: torch.Tensor | None = None):
        q_err = quat_mul(quat_conjugate(quat), quat_sp)  # rotation from current to desired, body frame
        rotvec = quat_log_rotvec(q_err)
        rate = self.kp * rotvec
        if yaw_rate_ff is not None:
            # feed-forward yaw rate expressed in body frame
            ff_w = torch.zeros_like(rate)
            ff_w[:, 2] = yaw_rate_ff
            rate = rate + quat_rotate_inverse(quat, ff_w)
        return torch.maximum(torch.minimum(rate, self.max_rate), -self.max_rate)


def attitude_from_thrust_and_yaw(thrust_dir_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Desired attitude whose body z axis aligns with ``thrust_dir_w`` and heading equals ``yaw``."""
    b3 = thrust_dir_w / thrust_dir_w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c1 = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
    b2 = torch.cross(b3, c1, dim=-1)
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    b1 = torch.cross(b2, b3, dim=-1)
    r = torch.stack([b1, b2, b3], dim=-1)
    return rotmat_to_quat(r)


@dataclass
class PositionGains:
    kp_pos: tuple[float, float, float] = (1.5, 1.5, 2.0)
    kp_vel: tuple[float, float, float] = (3.0, 3.0, 4.0)
    ki_vel: tuple[float, float, float] = (0.4, 0.4, 1.0)
    max_vel: tuple[float, float, float] = (5.0, 5.0, 3.0)
    max_acc_xy: float = 6.0
    max_tilt: float = 0.6  # rad
    i_limit: float = 3.0


class PositionController:
    """PX4-like cascaded P(position) -> PI(velocity) -> thrust vector + attitude set-point."""

    def __init__(self, params: QuadrotorParams, num_envs: int, dt: float, gains: PositionGains | None = None,
                 device="cpu"):
        g = gains or PositionGains()
        self.mass = params.mass
        self.t_max = params.max_total_thrust
        self.g = g
        self.dt = dt
        self.kp_pos = torch.tensor(g.kp_pos, device=device)
        self.kp_vel = torch.tensor(g.kp_vel, device=device)
        self.ki_vel = torch.tensor(g.ki_vel, device=device)
        self.max_vel = torch.tensor(g.max_vel, device=device)
        self.integral = torch.zeros(num_envs, 3, device=device)

    def reset(self, env_ids=None):
        self.integral[slice(None) if env_ids is None else env_ids] = 0.0

    def velocity_to_thrust(self, vel_sp, vel, acc_ff=None):
        err = vel_sp - vel
        self.integral = torch.clamp(self.integral + err * self.dt, -self.g.i_limit, self.g.i_limit)
        acc = self.kp_vel * err + self.ki_vel * self.integral
        if acc_ff is not None:
            acc = acc + acc_ff
        # limit horizontal acceleration and tilt
        acc_xy = acc[:, :2]
        n_xy = acc_xy.norm(dim=-1, keepdim=True)
        acc_xy = acc_xy * torch.clamp(self.g.max_acc_xy / n_xy.clamp_min(1e-9), max=1.0)
        acc_z = acc[:, 2:3] + GRAVITY
        acc_z = torch.clamp(acc_z, min=0.2 * GRAVITY)
        max_xy = acc_z * torch.tan(torch.tensor(self.g.max_tilt))
        acc_xy = acc_xy * torch.clamp(max_xy / acc_xy.norm(dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
        return self.mass * torch.cat([acc_xy, acc_z], dim=-1)  # desired thrust vector, world frame

    def __call__(self, pos_sp, pos, vel, vel_ff=None, acc_ff=None):
        vel_sp = self.kp_pos * (pos_sp - pos)
        if vel_ff is not None:
            vel_sp = vel_sp + vel_ff
        vel_sp = torch.maximum(torch.minimum(vel_sp, self.max_vel), -self.max_vel)
        return self.velocity_to_thrust(vel_sp, vel, acc_ff)


def thrust_vector_to_setpoints(thrust_w: torch.Tensor, quat: torch.Tensor, yaw_sp: torch.Tensor):
    """Collective thrust (projected on current body z, as in PX4) and attitude set-point."""
    q_sp = attitude_from_thrust_and_yaw(thrust_w, yaw_sp)
    body_z = quat_rotate(quat, torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand_as(thrust_w))
    collective = torch.clamp((thrust_w * body_z).sum(-1), min=0.0)
    return collective, q_sp


@dataclass
class GeometricGains:
    kx: tuple[float, float, float] = (6.0, 6.0, 8.0)
    kv: tuple[float, float, float] = (4.0, 4.0, 5.0)
    kr: tuple[float, float, float] = (60.0, 60.0, 20.0)  # angular-acceleration units (scaled by I)
    kw: tuple[float, float, float] = (12.0, 12.0, 6.0)


class GeometricController:
    """Lee, Leok & McClamroch (2010) geometric tracking controller on SE(3).

    Gains are expressed per unit mass / inertia so they transfer across vehicles.
    Returns collective thrust (N,) and body torque (N, 3).
    """

    def __init__(self, params: QuadrotorParams, gains: GeometricGains | None = None, device="cpu"):
        g = gains or GeometricGains()
        self.mass = params.mass
        self.inertia = torch.tensor(params.inertia, device=device)
        self.kx = torch.tensor(g.kx, device=device)
        self.kv = torch.tensor(g.kv, device=device)
        self.kr = torch.tensor(g.kr, device=device)
        self.kw = torch.tensor(g.kw, device=device)

    def __call__(self, pos, quat, vel, ang_vel, pos_d, vel_d, acc_d, yaw_d, ang_vel_d=None):
        e_x = pos - pos_d
        e_v = vel - vel_d
        e3 = torch.zeros_like(pos)
        e3[:, 2] = 1.0
        f_des = self.mass * (-self.kx * e_x - self.kv * e_v + GRAVITY * e3 + acc_d)
        r = quat_to_rotmat(quat)
        b3 = r[..., 2]
        thrust = (f_des * b3).sum(-1).clamp_min(0.0)
        q_d = attitude_from_thrust_and_yaw(f_des, yaw_d)
        r_d = quat_to_rotmat(q_d)
        e_r = 0.5 * vee(r_d.transpose(-1, -2) @ r - r.transpose(-1, -2) @ r_d)
        w_d = torch.zeros_like(ang_vel) if ang_vel_d is None else ang_vel_d
        e_w = ang_vel - (r.transpose(-1, -2) @ r_d @ w_d[..., None]).squeeze(-1)
        torque = self.inertia * (-self.kr * e_r - self.kw * e_w) + torch.cross(ang_vel, self.inertia * ang_vel, dim=-1)
        return thrust, torque
