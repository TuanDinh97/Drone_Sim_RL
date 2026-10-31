"""Batched multirotor physics: rotor/motor dynamics, aerodynamics and body wrench.

Everything here is framework-agnostic PyTorch so that it can be (i) unit-tested without
Isaac Sim, (ii) shared verbatim between the Isaac Lab environments (where PhysX integrates
the rigid body) and the reference integrator in :mod:`quadsim.core.dynamics`, and
(iii) reused for model-based baselines and system identification.

Wrench model (body frame B, rotor ``i`` at ``r_i`` with spin ``s_i``)::

    T_i     = k_f,i * omega_i^2 * g_e(h_i)                  thrust along +z_B
    F_rd,i  = -c_rd * omega_i * v_perp,i                    rotor (blade-flapping) drag
    M_rm,i  = -c_rm * s_i * omega_i * v_perp,i              rotor rolling moment
    Q_i     = -s_i (k_m T_i + J_r d(omega_i)/dt) e_z         aerodynamic + spin-up reaction torque
    tau_gyr = -w_B x sum_i J_r s_i omega_i e_z               rotor gyroscopic torque
    F_body  = -D_lin v_rel - D_quad |v_rel| v_rel            parasitic body drag

    F_B   = sum_i (T_i e_z + F_rd,i) + F_body
    tau_B = sum_i [ r_i x (T_i e_z + F_rd,i) + M_rm,i + Q_i ] + tau_gyr

where ``v_rel`` is the air-relative velocity (wind subtracted), ``v_perp,i`` its component
perpendicular to the rotor axis at the rotor hub and ``g_e`` the Cheeseman-Bennett ground-effect
factor. The rigid-body term ``w x I w`` is *not* included (PhysX / the reference integrator add it).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .params import QuadrotorParams


class QuadrotorModel:
    """Per-environment parameter tensors (enables domain randomization) + rotor state."""

    def __init__(self, params: QuadrotorParams, num_envs: int, device: str | torch.device = "cpu"):
        self.nominal = params
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.num_rotors = params.num_rotors
        n, r, dev = num_envs, params.num_rotors, self.device

        def full(shape, value):
            return torch.full(shape, float(value), device=dev)

        self.mass = full((n,), params.mass)
        self.inertia = torch.tensor(params.inertia, device=dev).repeat(n, 1)
        self.rotor_pos_nominal = torch.tensor(params.rotor_positions, device=dev)
        self.rotor_pos = self.rotor_pos_nominal.repeat(n, 1, 1)
        self.spin = torch.tensor(params.rotor_spin, device=dev, dtype=torch.float32)
        self.k_f = full((n, r), params.k_f)
        self.k_m = full((n, r), params.k_m)
        self.omega_min = full((n, r), params.omega_min)
        self.omega_max = full((n, r), params.omega_max)
        self.tau_up = full((n, r), params.tau_up)
        self.tau_down = full((n, r), params.tau_down)
        self.rotor_inertia = full((n,), params.rotor_inertia)
        self.rotor_radius = full((n,), params.rotor_radius)
        self.rotor_drag_coeff = full((n,), params.rotor_drag_coeff)
        self.rolling_moment_coeff = full((n,), params.rolling_moment_coeff)
        self.body_drag_lin = torch.tensor(params.body_drag_lin, device=dev).repeat(n, 1)
        self.body_drag_quad = torch.tensor(params.body_drag_quad, device=dev).repeat(n, 1)
        self.ground_effect = params.ground_effect
        # CoM offset w.r.t. the nominal geometric CoM (domain randomization); shifts rotor lever arms
        self.com_offset = torch.zeros(n, 3, device=dev)

        # rotor state
        self.omega = full((n, r), params.hover_omega)
        self.omega_dot = torch.zeros(n, r, device=dev)

    # ------------------------------------------------------------------ state
    def reset(self, env_ids: torch.Tensor | slice | None = None, omega: torch.Tensor | float | None = None):
        ids = slice(None) if env_ids is None else env_ids
        if omega is None:
            omega = torch.sqrt(self.mass[ids, None] * 9.81 / (self.k_f[ids].sum(-1, keepdim=True)))
            omega = omega.expand(-1, self.num_rotors)
        self.omega[ids] = omega
        self.omega_dot[ids] = 0.0

    def lever_arms(self) -> torch.Tensor:
        return self.rotor_pos - self.com_offset[:, None, :]

    # ------------------------------------------------------------------ motors
    def thrust_to_omega(self, thrust: torch.Tensor, k_f: torch.Tensor | None = None) -> torch.Tensor:
        k_f = self.k_f if k_f is None else k_f
        return torch.sqrt(torch.clamp(thrust, min=0.0) / k_f)

    def step_motors(self, omega_cmd: torch.Tensor, dt: float) -> torch.Tensor:
        """Exact zero-order-hold solution of the first-order rotor-speed model.

        ``d omega/dt = (omega_cmd - omega) / tau`` with ``tau = tau_up`` when accelerating and
        ``tau_down`` when decelerating; the command is saturated to ``[omega_min, omega_max]``.
        """
        cmd = torch.minimum(torch.maximum(omega_cmd, self.omega_min), self.omega_max)
        tau = torch.where(cmd >= self.omega, self.tau_up, self.tau_down)
        alpha = torch.exp(-dt / tau.clamp_min(1e-6))
        new = cmd + (self.omega - cmd) * alpha
        self.omega_dot = (new - self.omega) / dt
        self.omega = new
        return new

    # ------------------------------------------------------------------ forces
    def ground_effect_factor(self, rotor_height: torch.Tensor | None) -> torch.Tensor:
        if rotor_height is None or not self.ground_effect:
            return torch.ones_like(self.omega)
        radius = self.rotor_radius[:, None]
        h = torch.maximum(rotor_height, 0.5 * radius)  # model invalid below h = R/2; cap at 4/3 gain
        ratio = radius / (4.0 * h)
        return 1.0 / (1.0 - ratio**2)

    def compute_wrench(
        self,
        lin_vel_b: torch.Tensor,
        ang_vel_b: torch.Tensor,
        wind_b: torch.Tensor | None = None,
        rotor_height: torch.Tensor | None = None,
        omega: torch.Tensor | None = None,
        omega_dot: torch.Tensor | None = None,
    ) -> WrenchOutput:
        """Body-frame force & torque about the CoM from the current rotor state.

        Args:
            lin_vel_b: CoM linear velocity in body frame (N, 3).
            ang_vel_b: angular velocity in body frame (N, 3).
            wind_b: wind velocity expressed in body frame (N, 3).
            rotor_height: height of each rotor hub above the ground (N, R) for ground effect.
        """
        omega = self.omega if omega is None else omega
        omega_dot = self.omega_dot if omega_dot is None else omega_dot
        n = omega.shape[0]
        e_z = torch.zeros(n, 1, 3, device=omega.device)
        e_z[..., 2] = 1.0
        arms = self.lever_arms()  # (N, R, 3)

        thrust = self.k_f * omega**2 * self.ground_effect_factor(rotor_height)  # (N, R)
        f_thrust = thrust[..., None] * e_z  # (N, R, 3)

        v_rel = lin_vel_b if wind_b is None else lin_vel_b - wind_b
        v_rotor = v_rel[:, None, :] + torch.cross(ang_vel_b[:, None, :].expand_as(arms), arms, dim=-1)
        v_perp = v_rotor.clone()
        v_perp[..., 2] = 0.0
        f_rd = -self.rotor_drag_coeff[:, None, None] * omega[..., None] * v_perp
        m_rm = -self.rolling_moment_coeff[:, None, None] * (self.spin * omega)[..., None] * v_perp

        f_rotor = f_thrust + f_rd
        tau = torch.cross(arms, f_rotor, dim=-1).sum(1) + m_rm.sum(1)
        # yaw reaction: aerodynamic drag torque + spin-up reaction
        q_yaw = -(self.spin * (self.k_m * thrust + self.rotor_inertia[:, None] * omega_dot)).sum(-1)
        tau[:, 2] += q_yaw
        # rotor gyroscopic torque
        h_rot = (self.rotor_inertia[:, None] * self.spin * omega).sum(-1)  # (N,)
        h_vec = torch.zeros(n, 3, device=omega.device)
        h_vec[:, 2] = h_rot
        tau = tau - torch.cross(ang_vel_b, h_vec, dim=-1)

        f_body = -self.body_drag_lin * v_rel - self.body_drag_quad * v_rel.abs() * v_rel
        force = f_rotor.sum(1) + f_body
        return WrenchOutput(force=force, torque=tau, thrust=thrust, body_drag=f_body, rotor_drag=f_rd.sum(1))


@dataclass
class WrenchOutput:
    force: torch.Tensor
    torque: torch.Tensor
    thrust: torch.Tensor
    body_drag: torch.Tensor
    rotor_drag: torch.Tensor


def allocation_matrix(model: QuadrotorModel) -> torch.Tensor:
    """Per-env matrix A (N, 4, R): [F_z, tau_x, tau_y, tau_z]^T = A @ T (hover, no drag)."""
    arms = model.lever_arms()
    a = torch.stack(
        [
            torch.ones_like(model.k_m),
            arms[..., 1],
            -arms[..., 0],
            -model.spin * model.k_m,
        ],
        dim=1,
    )
    return a
