"""Control allocation (wrench -> rotor thrusts) with actuator-saturation handling."""

from __future__ import annotations

import torch

from ..params import QuadrotorParams


class Mixer:
    """Maps collective thrust + body torques to per-rotor thrusts using the *nominal* model.

    Desaturation follows the spirit of PX4's sequential desaturation ("airmode"):
      1. roll/pitch torques are scaled down only if their spread alone exceeds the actuator range;
      2. collective thrust is shifted (up or down) so that roll/pitch authority is preserved;
      3. yaw torque is added with the largest feasible scale in [0, 1];
      4. the result is clipped to [T_min, T_max].
    """

    def __init__(self, params: QuadrotorParams, device="cpu"):
        pos = torch.tensor(params.rotor_positions, device=device)
        spin = torch.tensor(params.rotor_spin, device=device, dtype=torch.float32)
        a = torch.stack([torch.ones_like(spin), pos[:, 1], -pos[:, 0], -spin * params.k_m], dim=0)
        self.A = a
        self.A_inv = torch.linalg.pinv(a)  # (R, 4)
        # per-rotor thrust per newton of collective (zero torque); 1/R on symmetric frames
        self.collective_split = self.A_inv[:, 0].clone()
        d = self.collective_split
        self.symmetric = bool((d.max() - d.min()) <= 1e-4 * d.abs().max())  # float32 pinv noise ~1e-6
        self.t_min = params.k_f * params.omega_min**2
        self.t_max = params.k_f * params.omega_max**2
        self.k_f = params.k_f
        self.num_rotors = params.num_rotors

    def wrench_from_thrusts(self, thrusts: torch.Tensor) -> torch.Tensor:
        return thrusts @ self.A.T

    def allocate(self, collective: torch.Tensor, torque: torch.Tensor) -> torch.Tensor:
        """Args: collective (N,), torque (N, 3). Returns rotor thrusts (N, R) in [T_min, T_max]."""
        n = collective.shape[0]
        zeros = torch.zeros(n, device=collective.device)
        t_rp = torch.stack([zeros, torque[:, 0], torque[:, 1], zeros], -1) @ self.A_inv.T
        t_yaw = torch.stack([zeros, zeros, zeros, torque[:, 2]], -1) @ self.A_inv.T
        t_coll = torch.stack([collective, zeros, zeros, zeros], -1) @ self.A_inv.T  # equal split for symmetric

        span = self.t_max - self.t_min
        rp_spread = t_rp.max(-1).values - t_rp.min(-1).values
        s = torch.clamp(span / rp_spread.clamp_min(1e-9), max=1.0)
        t_rp = t_rp * s[:, None]

        if self.symmetric:  # shift collective so roll/pitch fits (keeps per-rotor collective equal)
            c = t_coll.mean(-1)
            c_lo = self.t_min - t_rp.min(-1).values
            c_hi = self.t_max - t_rp.max(-1).values
            c = torch.minimum(torch.maximum(c, c_lo), c_hi)
            base = c[:, None] + t_rp
        else:  # asymmetric geometry: an equal split would produce a torque; shift along the zero-torque split
            d = self.collective_split
            s_lo = ((self.t_min - t_rp) / d).max(-1).values
            s_hi = ((self.t_max - t_rp) / d).min(-1).values
            s = torch.minimum(torch.maximum(collective, s_lo), s_hi)
            base = s[:, None] * d + t_rp

        # largest feasible yaw scale
        room_up = (self.t_max - base) / t_yaw.clamp_min(1e-12)
        room_dn = (self.t_min - base) / t_yaw.clamp_max(-1e-12)
        k = torch.where(t_yaw > 1e-12, room_up, torch.where(t_yaw < -1e-12, room_dn, torch.full_like(base, 1.0)))
        k = torch.clamp(k.min(-1).values, 0.0, 1.0)
        return torch.clamp(base + k[:, None] * t_yaw, self.t_min, self.t_max)

    def thrust_to_omega(self, thrusts: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(thrusts, min=0.0) / self.k_f)
