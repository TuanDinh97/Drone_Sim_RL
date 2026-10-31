"""Reference 6-DoF rigid-body integrator for multirotors (pure PyTorch).

Newton-Euler equations (world W z-up, body B FLU, ``q = q_WB``)::

    p_dot = v
    v_dot = R(q) F_B / m + g + F_ext,W / m
    q_dot = 0.5 q (x) [0, w_B]
    w_dot = I^-1 (tau_B - w_B x I w_B)

Used as an independent reference to validate the PhysX-integrated Isaac Lab
environments and for fast unit tests / controller tuning. Two integrators are offered:
``rk4`` (high-accuracy reference) and ``semi_implicit`` (symplectic Euler, the scheme
PhysX uses for rigid bodies), both with the body wrench held constant over a step
(zero-order hold, identical to how the Isaac environments apply wrenches).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .math import quat_exp_rotvec, quat_mul, quat_normalize, quat_rotate, quat_rotate_inverse
from .model import QuadrotorModel

GRAVITY = 9.81


@dataclass
class RigidState:
    pos: torch.Tensor  # (N, 3) world
    quat: torch.Tensor  # (N, 4) wxyz, body->world
    lin_vel: torch.Tensor  # (N, 3) world
    ang_vel: torch.Tensor  # (N, 3) body

    def clone(self) -> RigidState:
        return RigidState(self.pos.clone(), self.quat.clone(), self.lin_vel.clone(), self.ang_vel.clone())

    @staticmethod
    def zeros(n: int, device="cpu") -> RigidState:
        q = torch.zeros(n, 4, device=device)
        q[:, 0] = 1.0
        z = torch.zeros(n, 3, device=device)
        return RigidState(z.clone(), q, z.clone(), z.clone())


def _derivatives(state: RigidState, force_b, torque_b, force_w, mass, inertia, gravity):
    acc = quat_rotate(state.quat, force_b) / mass[:, None]
    acc[:, 2] -= gravity
    if force_w is not None:
        acc = acc + force_w / mass[:, None]
    w = state.ang_vel
    iw = inertia * w
    w_dot = (torque_b - torch.cross(w, iw, dim=-1)) / inertia
    omega_quat = torch.cat([torch.zeros_like(w[:, :1]), w], dim=-1)
    q_dot = 0.5 * quat_mul(state.quat, omega_quat)
    return state.lin_vel, acc, q_dot, w_dot


def integrate_rigid_body(
    state: RigidState,
    force_b: torch.Tensor,
    torque_b: torch.Tensor,
    mass: torch.Tensor,
    inertia: torch.Tensor,
    dt: float,
    method: str = "rk4",
    force_w: torch.Tensor | None = None,
    gravity: float = GRAVITY,
) -> RigidState:
    """Advance the rigid body by ``dt`` with a constant body wrench (+ optional world force)."""
    if method == "semi_implicit":
        _, acc, _, w_dot = _derivatives(state, force_b, torque_b, force_w, mass, inertia, gravity)
        v = state.lin_vel + acc * dt
        w = state.ang_vel + w_dot * dt
        p = state.pos + v * dt
        q = quat_normalize(quat_mul(state.quat, quat_exp_rotvec(w * dt)))
        return RigidState(p, q, v, w)
    if method != "rk4":
        raise ValueError(f"unknown integrator '{method}'")

    def f(s: RigidState):
        return _derivatives(s, force_b, torque_b, force_w, mass, inertia, gravity)

    def add(s: RigidState, d, h):
        return RigidState(s.pos + h * d[0], quat_normalize(s.quat + h * d[2]), s.lin_vel + h * d[1], s.ang_vel + h * d[3])

    k1 = f(state)
    k2 = f(add(state, k1, 0.5 * dt))
    k3 = f(add(state, k2, 0.5 * dt))
    k4 = f(add(state, k3, dt))
    comb = [(a + 2 * b + 2 * c + d) / 6.0 for a, b, c, d in zip(k1, k2, k3, k4)]
    return RigidState(
        state.pos + dt * comb[0],
        quat_normalize(state.quat + dt * comb[2]),
        state.lin_vel + dt * comb[1],
        state.ang_vel + dt * comb[3],
    )


class ReferenceQuadrotorSim:
    """Batched stand-alone quadrotor simulator (motor model + aerodynamics + rigid body).

    Per step: rotor speeds are advanced exactly over ``dt`` (ZOH command), the body wrench is
    evaluated from the new rotor state, then the rigid body is integrated with that wrench held
    constant. This is exactly the sequence the Isaac Lab environment performs per physics step.
    """

    def __init__(self, model: QuadrotorModel, dt: float, method: str = "rk4", substeps: int = 1):
        self.model = model
        self.dt = dt
        self.method = method
        self.substeps = substeps
        self.state = RigidState.zeros(model.num_envs, model.device)
        self.wind_w = torch.zeros(model.num_envs, 3, device=model.device)
        self.ground_height = 0.0
        self.use_ground_effect = False

    def rotor_heights(self) -> torch.Tensor:
        arms_w = quat_rotate(self.state.quat[:, None, :].expand(-1, self.model.num_rotors, -1), self.model.lever_arms())
        return self.state.pos[:, None, 2] + arms_w[..., 2] - self.ground_height

    def step(self, omega_cmd: torch.Tensor, force_w: torch.Tensor | None = None):
        self.model.step_motors(omega_cmd, self.dt)
        v_b = quat_rotate_inverse(self.state.quat, self.state.lin_vel)
        wind_b = quat_rotate_inverse(self.state.quat, self.wind_w)
        heights = self.rotor_heights() if self.use_ground_effect else None
        wrench = self.model.compute_wrench(v_b, self.state.ang_vel, wind_b, heights)
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            self.state = integrate_rigid_body(
                self.state, wrench.force, wrench.torque, self.model.mass, self.model.inertia, h, self.method, force_w
            )
        return wrench
