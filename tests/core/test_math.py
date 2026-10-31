import math

import torch

from quadsim.core import math as qm


def _rand_quat(n, gen):
    q = torch.randn(n, 4, generator=gen, dtype=torch.float64)
    return qm.quat_normalize(q)


def test_rotate_roundtrip_and_rotmat():
    g = torch.Generator().manual_seed(0)
    q = _rand_quat(256, g)
    v = torch.randn(256, 3, generator=g, dtype=torch.float64)
    assert torch.allclose(qm.quat_rotate_inverse(q, qm.quat_rotate(q, v)), v, atol=1e-12)
    r = qm.quat_to_rotmat(q)
    assert torch.allclose((r @ v[..., None]).squeeze(-1), qm.quat_rotate(q, v), atol=1e-12)
    q2 = qm.rotmat_to_quat(r)
    # equal up to sign
    assert torch.allclose(q2 * torch.sign((q2 * q).sum(-1, keepdim=True)), q, atol=1e-10)


def test_euler_roundtrip():
    g = torch.Generator().manual_seed(1)
    roll = (torch.rand(200, generator=g, dtype=torch.float64) - 0.5) * 2 * 3.0
    pitch = (torch.rand(200, generator=g, dtype=torch.float64) - 0.5) * 2 * 1.5
    yaw = (torch.rand(200, generator=g, dtype=torch.float64) - 0.5) * 2 * 3.0
    q = qm.quat_from_euler_xyz(roll, pitch, yaw)
    r2, p2, y2 = qm.euler_xyz_from_quat(q)
    assert torch.allclose(qm.wrap_to_pi(r2 - roll), torch.zeros_like(roll), atol=1e-9)
    assert torch.allclose(p2, pitch, atol=1e-9)
    assert torch.allclose(qm.wrap_to_pi(y2 - yaw), torch.zeros_like(yaw), atol=1e-9)


def test_exp_log():
    g = torch.Generator().manual_seed(2)
    rv = torch.randn(300, 3, generator=g, dtype=torch.float64)
    rv = rv / rv.norm(dim=-1, keepdim=True) * torch.rand(300, 1, generator=g, dtype=torch.float64) * (math.pi - 1e-3)
    assert torch.allclose(qm.quat_log_rotvec(qm.quat_exp_rotvec(rv)), rv, atol=1e-9)
    tiny = torch.tensor([[1e-10, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(qm.quat_log_rotvec(qm.quat_exp_rotvec(tiny)), tiny, atol=1e-15)


def test_frame_convention_flu():
    # +90 deg yaw maps body x (forward) to world y
    q = qm.quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([math.pi / 2]))
    v = qm.quat_rotate(q, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(v, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)
    # positive pitch about +y (FLU) pitches the nose DOWN (body x gets negative world z)
    q = qm.quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.3]), torch.tensor([0.0]))
    assert qm.quat_rotate(q, torch.tensor([[1.0, 0.0, 0.0]]))[0, 2] < 0


def test_random_tilt_bound():
    q = qm.random_quat_tilt(1000, 0.5, "cpu", torch.Generator().manual_seed(3))
    z = qm.quat_rotate(q, torch.tensor([0.0, 0.0, 1.0]).expand(1000, 3))
    tilt = torch.acos(z[:, 2].clamp(-1, 1))
    assert tilt.max() <= 0.5 + 1e-5
