"""Batched rotation utilities (pure PyTorch, no Isaac dependency).

Conventions (identical to Isaac Lab 2.x):
  * Quaternions are scalar-first ``(w, x, y, z)``.
  * ``q_wb`` rotates vectors from the body frame B into the world frame W:
    ``v_w = quat_rotate(q_wb, v_b)``.
  * World frame W is Z-up (ENU-like); body frame B is FLU (x forward, y left, z up).
"""

from __future__ import annotations

import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a ⊗ b``."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by ``q`` (body -> world for ``q_wb``)."""
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by ``q^{-1}`` (world -> body for ``q_wb``)."""
    return quat_rotate(quat_conjugate(q), v)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_normalize(q).unbind(-1)
    r = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return r.reshape(q.shape[:-1] + (3, 3))


def rotmat_to_quat(r: torch.Tensor) -> torch.Tensor:
    """Robust (Shepperd) rotation-matrix to quaternion conversion, ``w >= 0``."""
    m00, m01, m02 = r[..., 0, 0], r[..., 0, 1], r[..., 0, 2]
    m10, m11, m12 = r[..., 1, 0], r[..., 1, 1], r[..., 1, 2]
    m20, m21, m22 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], dim=-1
            ),
            min=0.0,
        )
    )
    cands = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    cands = cands / (2.0 * q_abs[..., None].clamp_min(0.1))
    idx = q_abs.argmax(dim=-1)
    q = torch.gather(cands, -2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
    q = quat_normalize(q)
    return torch.where(q[..., :1] < 0, -q, q)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Intrinsic Z-Y'-X'' (yaw, pitch, roll) Tait-Bryan angles to quaternion."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ],
        dim=-1,
    )


def euler_xyz_from_quat(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    half = 0.5 * angle[..., None]
    return torch.cat([torch.cos(half), torch.sin(half) * axis], dim=-1)


def quat_exp_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    """Quaternion from a rotation vector (axis * angle); stable near zero."""
    angle = rotvec.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    # sinc(half)/2 written to remain smooth at 0
    k = torch.where(angle > 1e-8, torch.sin(half) / angle.clamp_min(1e-12), 0.5 - angle**2 / 48.0)
    return torch.cat([torch.cos(half), k * rotvec], dim=-1)


def quat_log_rotvec(q: torch.Tensor) -> torch.Tensor:
    """Rotation vector of ``q`` (shortest arc, ``|angle| <= pi``)."""
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, q[..., :1])
    k = torch.where(s > 1e-8, angle / s.clamp_min(1e-12), 2.0 / q[..., :1].clamp_min(1e-12))
    return k * v


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    yaw = yaw_from_quat(q)
    zeros = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)], dim=-1)


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).reshape(v.shape[:-1] + (3, 3))


def vee(m: torch.Tensor) -> torch.Tensor:
    return torch.stack([m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]], dim=-1)


def random_quat_tilt(n: int, max_tilt: float, device, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random attitudes with uniformly distributed yaw and tilt angle ``<= max_tilt`` (rad)."""
    yaw = (torch.rand(n, device=device, generator=generator) * 2 - 1) * torch.pi
    tilt = torch.rand(n, device=device, generator=generator) * max_tilt
    az = (torch.rand(n, device=device, generator=generator) * 2 - 1) * torch.pi
    axis = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)], dim=-1)
    q_tilt = quat_from_angle_axis(tilt, axis)
    zeros = torch.zeros_like(yaw)
    q_yaw = torch.stack([torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)], dim=-1)
    return quat_mul(q_yaw, q_tilt)
