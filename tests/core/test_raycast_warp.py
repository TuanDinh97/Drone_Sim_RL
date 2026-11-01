import pytest
import torch

wp = pytest.importorskip("warp")

from quadsim.core.sensors import cast, pinhole_ray_dirs  # noqa: E402
from quadsim.core.sensors.raycast_warp import cast_warp  # noqa: E402


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_warp_matches_torch(device):
    wp.init()
    g = torch.Generator().manual_seed(0)
    n, k = 8, 12
    dirs_c = pinhole_ray_dirs(32, 24, 87.0, "cpu")
    r = dirs_c.shape[0]
    yaw = torch.rand(n, generator=g) * 6.28
    c, s = torch.cos(yaw), torch.sin(yaw)
    rot = torch.zeros(n, 3, 3)
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1.0
    pitch_down = torch.tensor([[0.94, 0.0, 0.34], [0.0, 1.0, 0.0], [-0.34, 0.0, 0.94]])
    dirs = torch.einsum("nij,jk,rk->nri", rot, pitch_down, dirs_c)
    origins = torch.zeros(n, r, 3)
    origins[..., 2] = 1.0 + torch.rand(n, 1, generator=g)
    cyl = torch.zeros(n, k, 4)
    cyl[..., 0] = torch.rand(n, k, generator=g) * 8 - 4
    cyl[..., 1] = torch.rand(n, k, generator=g) * 8 - 4
    cyl[..., 2] = 0.2 + 0.3 * torch.rand(n, k, generator=g)
    cyl[..., 3] = torch.where(torch.rand(n, k, generator=g) < 0.3, torch.full((n, k), 0.8), torch.full((n, k), 4.0))
    cyl[:, :2, 2] = 0.0  # inactive obstacles
    ref = cast(origins, dirs, cyl, None, ground=True, max_range=10.0)
    out = cast_warp(origins.to(device), dirs.to(device), cyl.to(device), ground=True, max_range=10.0).cpu()
    assert torch.allclose(out, ref, atol=2e-4), (out - ref).abs().max()
