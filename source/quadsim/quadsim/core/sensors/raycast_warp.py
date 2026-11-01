"""NVIDIA Warp kernel for the analytic ray caster (ground + vertical cylinders).

Memory is O(num_envs * num_rays) instead of the O(num_envs * num_rays * num_obstacles) temporaries of the
PyTorch implementation in :mod:`raycast`, which matters for depth images (thousands of rays per env).
Semantics are identical to ``raycast.cast(origins, dirs, cylinders, None, ground=True)``; this is
checked in ``tests/core/test_raycast_warp.py``.
"""

from __future__ import annotations

import torch
import warp as wp

_INF = 1.0e6


@wp.kernel
def _cast_kernel(
    origins: wp.array(dtype=wp.vec3, ndim=2),
    dirs: wp.array(dtype=wp.vec3, ndim=2),
    cyl: wp.array(dtype=wp.vec4, ndim=2),
    use_ground: int,
    max_range: float,
    out: wp.array(dtype=float, ndim=2),
):
    e, r = wp.tid()
    o = origins[e, r]
    d = dirs[e, r]
    t = float(_INF)
    if use_ground != 0 and d[2] < -1.0e-9:
        t = -o[2] / d[2]
    a = d[0] * d[0] + d[1] * d[1]
    for k in range(cyl.shape[1]):
        c = cyl[e, k]
        rad = c[2]
        h = c[3]
        if rad <= 0.0:
            continue
        ox = o[0] - c[0]
        oy = o[1] - c[1]
        c0 = ox * ox + oy * oy - rad * rad
        if c0 <= 0.0 and o[2] >= 0.0 and o[2] <= h:
            t = 0.0
            continue
        if a > 1.0e-12:
            b = 2.0 * (ox * d[0] + oy * d[1])
            disc = b * b - 4.0 * a * c0
            if disc >= 0.0:
                ts = (-b - wp.sqrt(disc)) / (2.0 * a)
                z = o[2] + ts * d[2]
                if ts >= 0.0 and z >= 0.0 and z <= h and ts < t:
                    t = ts
        if o[2] > h and wp.abs(d[2]) > 1.0e-9:
            tc = (h - o[2]) / d[2]
            px = ox + tc * d[0]
            py = oy + tc * d[1]
            if tc >= 0.0 and px * px + py * py <= rad * rad and tc < t:
                t = tc
    out[e, r] = wp.min(t, max_range)


def cast_warp(origins: torch.Tensor, dirs: torch.Tensor, cylinders: torch.Tensor | None, ground: bool = True,
              max_range: float = _INF) -> torch.Tensor:
    """origins, dirs: (N, R, 3) float32; cylinders: (N, K, 4) -> distances (N, R)."""
    n, r = origins.shape[:2]
    dev = str(origins.device)
    if cylinders is None or cylinders.shape[1] == 0:
        cylinders = torch.zeros(n, 1, 4, device=origins.device)
    out = torch.empty(n, r, device=origins.device, dtype=torch.float32)
    wp.launch(
        _cast_kernel, dim=(n, r),
        inputs=[wp.from_torch(origins.contiguous().float(), dtype=wp.vec3),
                wp.from_torch(dirs.contiguous().float(), dtype=wp.vec3),
                wp.from_torch(cylinders.contiguous().float(), dtype=wp.vec4),
                int(ground), float(max_range)],
        outputs=[wp.from_torch(out, dtype=wp.float32)], device=dev,
    )
    return out
