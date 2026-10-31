import math

import pytest
import torch

from quadsim.core import QuadrotorModel, ReferenceQuadrotorSim, RigidState, integrate_rigid_body, make_x500
from quadsim.core import math as qm


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_free_fall_rk4_exact():
    s = RigidState.zeros(2)
    s.pos[:, 2] = 10.0
    s.lin_vel[:, 0] = 1.0
    m = torch.ones(2)
    inertia = torch.ones(2, 3)
    dt, n = 0.01, 100
    for _ in range(n):
        s = integrate_rigid_body(s, torch.zeros(2, 3), torch.zeros(2, 3), m, inertia, dt, "rk4")
    t = dt * n
    assert torch.allclose(s.pos[:, 2], torch.full((2,), 10.0 - 0.5 * 9.81 * t * t), atol=1e-10)
    assert torch.allclose(s.pos[:, 0], torch.full((2,), t), atol=1e-12)


def test_semi_implicit_matches_physx_free_fall_formula():
    s = RigidState.zeros(1)
    dt, n = 1 / 400, 200
    for _ in range(n):
        s = integrate_rigid_body(s, torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(1), torch.ones(1, 3), dt,
                                 "semi_implicit")
    assert s.pos[0, 2].item() == pytest.approx(-9.81 * dt * dt * n * (n + 1) / 2, rel=1e-12)


def test_constant_yaw_torque():
    s = RigidState.zeros(1)
    izz = 0.04
    inertia = torch.tensor([[0.02, 0.02, izz]])
    tau = torch.tensor([[0.0, 0.0, 0.01]])
    dt, n = 1e-3, 1000
    g_comp = torch.tensor([[0.0, 0.0, 9.81]])  # cancel gravity with body force (attitude stays level)
    for _ in range(n):
        s = integrate_rigid_body(s, g_comp, tau, torch.ones(1), inertia, dt, "rk4")
    yaw = qm.yaw_from_quat(s.quat)[0].item()
    assert yaw == pytest.approx(0.5 * 0.01 / izz * 1.0**2, rel=1e-6)
    assert s.pos.abs().max().item() < 1e-9


def test_torque_free_conservation():
    """Euler's equations: angular momentum (world) and rotational energy are conserved."""
    s = RigidState.zeros(1)
    inertia = torch.tensor([[0.02, 0.03, 0.05]])
    s.ang_vel = torch.tensor([[0.1, 4.0, 0.1]])  # near the unstable intermediate axis -> tumbling
    h0 = qm.quat_rotate(s.quat, inertia * s.ang_vel)
    e0 = 0.5 * (inertia * s.ang_vel**2).sum()
    for _ in range(4000):
        s = integrate_rigid_body(s, torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(1), inertia, 1e-3, "rk4",
                                 gravity=0.0)
    h1 = qm.quat_rotate(s.quat, inertia * s.ang_vel)
    e1 = 0.5 * (inertia * s.ang_vel**2).sum()
    assert torch.allclose(h0, h1, atol=1e-7)
    assert e1.item() == pytest.approx(e0.item(), rel=1e-7)
    # it did tumble (Dzhanibekov effect): the y rate changed sign at some point
    assert s.ang_vel[0, 1].abs().item() < 4.0


def test_reference_sim_hover():
    p = make_x500()
    m = QuadrotorModel(p, 3)
    sim = ReferenceQuadrotorSim(m, dt=1 / 200, method="rk4")
    sim.state.pos[:, 2] = 1.0
    m.reset()
    cmd = torch.full((3, 4), p.hover_omega)
    for _ in range(400):
        sim.step(cmd)
    assert sim.state.pos[:, 2].sub(1.0).abs().max().item() < 1e-6
    assert sim.state.ang_vel.abs().max().item() < 1e-9


def test_semi_implicit_converges_to_rk4():
    p = make_x500()
    errs = []
    for dt in (1 / 100, 1 / 400):
        outs = {}
        for method in ("rk4", "semi_implicit"):
            m = QuadrotorModel(p, 1)
            m.reset()
            sim = ReferenceQuadrotorSim(m, dt=dt, method=method)
            for k in range(int(round(1.0 / dt))):
                t = k * dt
                cmd = torch.full((1, 4), p.hover_omega)
                cmd[0, [1, 2]] += 15 * math.sin(6 * t)
                cmd[0, [2, 3]] += 10 * math.cos(4 * t)
                sim.step(cmd)
            outs[method] = sim.state.pos.clone()
        errs.append((outs["rk4"] - outs["semi_implicit"]).norm().item())
    assert errs[1] < errs[0] / 2.5  # first-order convergence (ratio ~4)
