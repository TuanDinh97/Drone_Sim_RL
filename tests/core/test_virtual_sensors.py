import numpy as np
import torch

from quadsim.core.sensors.raycast import cast, lidar_ray_dirs
from quadsim.deploy.virtual_sensors import clearance, ray_cylinders, ring_dirs


def _layout(rng, m=12):
    cyl = np.zeros((m, 4))
    cyl[:, 0] = rng.uniform(2, 10, m)
    cyl[:, 1] = rng.uniform(-4, 4, m)
    cyl[:, 2] = np.where(rng.random(m) < 0.8, rng.uniform(0.15, 0.45, m), 0.0)  # some inactive
    cyl[:, 3] = 4.0
    return cyl


def test_ring_matches_training_ray_caster():
    """Virtual LiDAR ring == the ring NavEnv computes (torch caster, yaw-aligned lidar_ray_dirs)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        cyl = _layout(rng)
        pos = np.array([rng.uniform(0, 12), rng.uniform(-3, 3), rng.uniform(0.5, 3.0)])
        yaw = rng.uniform(-np.pi, np.pi)
        r_np = ray_cylinders(pos[:2], pos[2], ring_dirs(32, yaw), cyl, 8.0)
        d = lidar_ray_dirs(32, 1, (0.0, 0.0), "cpu").double()
        c, s = np.cos(yaw), np.sin(yaw)
        rot = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)
        dirs = (d @ rot.T)[None]
        r_t = cast(torch.tensor(pos)[None, None].expand(1, 32, 3), dirs, torch.tensor(cyl)[None], None,
                   ground=False, max_range=8.0)[0].numpy()
        assert np.allclose(r_np, r_t, atol=1e-6)


def test_clearance_matches_nav_env_formula():
    rng = np.random.default_rng(1)
    cyl = _layout(rng)
    cyl[0] = [5.0, 0.0, 0.3, 4.0]
    pos = np.array([[5.0, 0.6, 2.0], [5.0, 0.6, 4.5], [20.0, 20.0, 1.0]])
    c = clearance(pos, cyl)
    d = (np.linalg.norm(pos[0, :2] - cyl[:, :2], axis=1) - cyl[:, 2])[cyl[:, 2] > 0]
    assert np.isclose(c[0], min(0.3, d.min()))
    assert c[1] == 1e3  # above every cylinder
    assert c[2] > 10.0


def test_transform_layout_is_rigid():
    from quadsim.deploy.virtual_sensors import transform_layout

    rng = np.random.default_rng(2)
    cyl = _layout(rng)
    start = np.array([0.0, 1.5])
    tf = [0.3, -1.5, 0.4]
    out = transform_layout(cyl, start, tf)
    assert np.allclose(out[:, 2:], cyl[:, 2:])  # radii / heights unchanged
    d0 = np.linalg.norm(cyl[:, None, :2] - cyl[None, :, :2], axis=-1)
    assert np.allclose(np.linalg.norm(out[:, None, :2] - out[None, :, :2], axis=-1), d0)  # distances preserved
    assert np.allclose(transform_layout(start[None], start, tf)[0], start + [0.3, -1.5])  # start -> start + (dx, dy)
    # relative bearing rotates by dyaw
    b0 = np.arctan2(*(cyl[0, :2] - start)[::-1])
    b1 = np.arctan2(*(out[0, :2] - start - [0.3, -1.5])[::-1])
    assert np.isclose(np.angle(np.exp(1j * (b1 - b0))), 0.4)
