"""NumPy versions of the navigation task's analytic sensing / collision geometry (for SITL tooling and analysis).

Mirror ``quadsim.core.sensors.raycast.cast`` (vertical cylinders, no ground) and ``NavEnv._obstacle_clearance``;
unit-tested against them (tests/core/test_virtual_sensors.py).
"""

from __future__ import annotations

import numpy as np


def ray_cylinders(origin_xy, z, dirs_xy, cyl, max_range: float):
    """Distances along horizontal unit rays ``dirs_xy`` (R, 2) from ``origin_xy`` at height ``z`` to vertical
    cylinders ``cyl`` (M, 4: cx, cy, r, h; r = 0 inactive), capped at ``max_range``."""
    out = np.full(len(dirs_xy), float(max_range))
    o = np.asarray(origin_xy, dtype=np.float64)
    for cx, cy, r, h in np.asarray(cyl, dtype=np.float64):
        if r <= 0 or z > h or z < 0:
            continue
        oc = o - np.array([cx, cy])
        b = dirs_xy @ oc
        disc = b * b - (oc @ oc - r * r)
        hit = disc >= 0
        sq = np.sqrt(np.where(hit, disc, 0.0))
        s = np.where(-b - sq >= 0, -b - sq, -b + sq)  # origin inside a cylinder: distance to its wall
        out = np.where(hit & (s >= 0), np.minimum(out, s), out)
    return out


def ring_dirs(num_rays: int, yaw: float):
    """Unit directions of the yaw-aligned ring (ray k at azimuth yaw + 2 pi k / num_rays)."""
    az = yaw + 2.0 * np.pi * np.arange(num_rays) / num_rays
    return np.stack([np.cos(az), np.sin(az)], -1)


def clearance(pos, cyl):
    """``NavEnv._obstacle_clearance``: horizontal distance from ``pos`` (..., 3) to the nearest active cylinder
    surface below whose top the vehicle is (1e3 if none)."""
    p = np.asarray(pos, dtype=np.float64)
    c = np.asarray(cyl, dtype=np.float64)
    d = np.linalg.norm(p[..., None, :2] - c[:, :2], axis=-1) - c[:, 2]
    valid = (c[:, 2] > 0) & ~(p[..., None, 2] > c[:, 3])
    return np.where(valid, d, 1e3).min(-1)


def transform_layout(cyl, start_xy, transform):
    """Rigid map of a navigation layout: rotate by ``dyaw`` about ``start_xy``, then translate by ``(dx, dy)``.
    ``cyl`` (M, >=2) columns 0-1 are x, y (other columns unchanged); returns a transformed copy."""
    dx, dy, dyaw = (float(v) for v in transform)
    c, s = np.cos(dyaw), np.sin(dyaw)
    out = np.array(cyl, dtype=np.float64, copy=True)
    rel = out[..., :2] - np.asarray(start_xy, dtype=np.float64)
    out[..., 0] = start_xy[0] + c * rel[..., 0] - s * rel[..., 1] + dx
    out[..., 1] = start_xy[1] + s * rel[..., 0] + c * rel[..., 1] + dy
    return out
