"""Analytic, batched ray casting against primitive worlds (GPU friendly, exact).

Supported geometry (per environment, environment-local coordinates):
  * ground plane ``z = 0`` (optional ceiling ``z = h_ceiling``),
  * vertical finite cylinders ``(cx, cy, radius, height)`` standing on the ground,
  * yaw-rotated boxes ``(cx, cy, cz, hx, hy, hz, yaw)``.

Used for range finders, LiDAR scans and *synthetic* depth images. Because the obstacle
fields of the navigation tasks are built from the very same primitives that are spawned in
USD, these analytic measurements are exact; they are validated against Isaac Sim's rendered
depth in ``scripts/validation/m3_depth_consistency.py``.
"""

from __future__ import annotations

import math

import torch

INF = 1e6


def ray_ground(origins, dirs, ceiling: float | None = None):
    t = torch.where(dirs[..., 2] < -1e-9, -origins[..., 2] / dirs[..., 2].clamp(max=-1e-9), torch.full_like(dirs[..., 2], INF))
    if ceiling is not None:
        tc = torch.where(dirs[..., 2] > 1e-9, (ceiling - origins[..., 2]) / dirs[..., 2].clamp(min=1e-9),
                         torch.full_like(t, INF))
        t = torch.minimum(t, tc)
    return t


def ray_cylinders(origins, dirs, cyl):
    """origins/dirs (N, R, 3); cyl (N, K, 4) -> distance (N, R) to the closest vertical cylinder."""
    if cyl is None or cyl.shape[1] == 0:
        return torch.full(origins.shape[:-1], INF, device=origins.device)
    o = origins[:, :, None, :]  # (N,R,1,3)
    d = dirs[:, :, None, :]
    c = cyl[:, None, :, :]  # (N,1,K,4)
    ox, oy = o[..., 0] - c[..., 0], o[..., 1] - c[..., 1]
    dx, dy = d[..., 0], d[..., 1]
    r, h = c[..., 2], c[..., 3]
    a = (dx * dx + dy * dy).clamp_min(1e-12)
    b = 2 * (ox * dx + oy * dy)
    c0 = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * c0
    sq = torch.sqrt(disc.clamp_min(0.0))
    t_side = (-b - sq) / (2 * a)
    z_side = o[..., 2] + t_side * d[..., 2]
    hit_side = (disc >= 0) & (t_side >= 0) & (z_side >= 0) & (z_side <= h)
    t = torch.where(hit_side, t_side, torch.full_like(t_side, INF))
    # top cap
    t_cap = (h - o[..., 2]) / torch.where(d[..., 2].abs() > 1e-9, d[..., 2], torch.full_like(d[..., 2], -1e-9))
    px, py = ox + t_cap * dx, oy + t_cap * dy
    hit_cap = (t_cap >= 0) & (px * px + py * py <= r * r) & (o[..., 2] > h)
    t = torch.minimum(t, torch.where(hit_cap, t_cap, torch.full_like(t_cap, INF)))
    # origin inside the cylinder -> zero range
    inside = (c0 <= 0) & (o[..., 2] >= 0) & (o[..., 2] <= h)
    t = torch.where(inside, torch.zeros_like(t), t)
    return t.min(dim=-1).values


def ray_boxes(origins, dirs, boxes):
    """boxes (N, M, 7) = (cx, cy, cz, hx, hy, hz, yaw) -> distance (N, R)."""
    if boxes is None or boxes.shape[1] == 0:
        return torch.full(origins.shape[:-1], INF, device=origins.device)
    o = origins[:, :, None, :]
    d = dirs[:, :, None, :]
    b = boxes[:, None, :, :]
    cy, sy = torch.cos(b[..., 6]), torch.sin(b[..., 6])
    rx, ry, rz = o[..., 0] - b[..., 0], o[..., 1] - b[..., 1], o[..., 2] - b[..., 2]
    # rotate into box frame (by -yaw)
    lo = torch.stack([cy * rx + sy * ry, -sy * rx + cy * ry, rz], -1)
    ld = torch.stack([cy * d[..., 0] + sy * d[..., 1], -sy * d[..., 0] + cy * d[..., 1], d[..., 2].expand_as(cy)], -1)
    half = b[..., 3:6]
    inv = 1.0 / torch.where(ld.abs() > 1e-9, ld, torch.full_like(ld, 1e-9))
    t1 = (-half - lo) * inv
    t2 = (half - lo) * inv
    tmin = torch.minimum(t1, t2).max(-1).values
    tmax = torch.maximum(t1, t2).min(-1).values
    hit = (tmax >= tmin.clamp_min(0.0))
    t = torch.where(hit, tmin.clamp_min(0.0), torch.full_like(tmin, INF))
    return t.min(dim=-1).values


def cast(origins, dirs, cylinders=None, boxes=None, ground=True, ceiling=None, max_range=INF):
    t = torch.full(origins.shape[:-1], INF, device=origins.device)
    if ground:
        t = torch.minimum(t, ray_ground(origins, dirs, ceiling))
    t = torch.minimum(t, ray_cylinders(origins, dirs, cylinders))
    t = torch.minimum(t, ray_boxes(origins, dirs, boxes))
    return t.clamp(max=max_range)


def pinhole_ray_dirs(width: int, height: int, hfov_deg: float, device) -> torch.Tensor:
    """Unit ray directions (H*W, 3) in the camera body frame (x forward, y left, z up), row-major.

    Pixel centres are used; the vertical FOV follows from square pixels.
    """
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    u = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    v = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    d = torch.stack([torch.ones_like(uu), -(uu - width / 2.0) / fx, -(vv - height / 2.0) / fx], -1).reshape(-1, 3)
    return d / d.norm(dim=-1, keepdim=True)


def lidar_ray_dirs(num_h: int, num_v: int, v_fov_deg: tuple[float, float], device, h_fov_deg: float = 360.0):
    """Spinning-LiDAR style directions (num_v*num_h, 3) in the sensor frame."""
    if h_fov_deg >= 360.0:
        az = torch.arange(num_h, device=device) * (2 * math.pi / num_h)
    else:
        az = torch.linspace(-math.radians(h_fov_deg) / 2, math.radians(h_fov_deg) / 2, num_h, device=device)
    el = torch.linspace(math.radians(v_fov_deg[0]), math.radians(v_fov_deg[1]), num_v, device=device) if num_v > 1 \
        else torch.tensor([math.radians(0.5 * (v_fov_deg[0] + v_fov_deg[1]))], device=device)
    ee, aa = torch.meshgrid(el, az, indexing="ij")
    d = torch.stack([torch.cos(ee) * torch.cos(aa), torch.cos(ee) * torch.sin(aa), torch.sin(ee)], -1)
    return d.reshape(-1, 3)
