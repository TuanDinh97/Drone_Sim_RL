import math

import pytest
import torch

from quadsim.core import QuadrotorModel, allocation_matrix, make_crazyflie, make_x500


@pytest.fixture(params=["x500", "crazyflie"])
def params(request):
    return make_x500() if request.param == "x500" else make_crazyflie()


def _zero(n):
    return torch.zeros(n, 3)


def test_hover_equilibrium(params):
    m = QuadrotorModel(params, 4)
    m.reset()
    w = m.compute_wrench(_zero(4), _zero(4))
    assert torch.allclose(w.force[:, 2], m.mass * 9.81, rtol=1e-5)
    assert torch.allclose(w.force[:, :2], torch.zeros(4, 2), atol=1e-7)
    assert torch.allclose(w.torque, torch.zeros(4, 3), atol=1e-7)


def test_allocation_matches_wrench(params):
    m = QuadrotorModel(params, 8)
    gen = torch.Generator().manual_seed(0)
    m.omega = params.omega_max * (0.3 + 0.7 * torch.rand(8, 4, generator=gen))
    w = m.compute_wrench(_zero(8), _zero(8))
    a = allocation_matrix(m)
    pred = (a @ w.thrust[..., None]).squeeze(-1)
    assert torch.allclose(pred[:, 0], w.force[:, 2], rtol=1e-5)
    assert torch.allclose(pred[:, 1:], w.torque, rtol=1e-4, atol=1e-7)


def test_torque_signs_x500():
    p = make_x500()
    m = QuadrotorModel(p, 1)
    base = torch.full((1, 4), p.hover_omega)
    # rotors: 0 FR(ccw) 1 RL(ccw) 2 FL(cw) 3 RR(cw)
    left = base.clone(); left[0, [1, 2]] *= 1.05  # noqa: E702
    assert m.compute_wrench(_zero(1), _zero(1), omega=left).torque[0, 0] > 0  # left side up = positive roll about +x (FLU)
    front = base.clone(); front[0, [0, 2]] *= 1.05  # noqa: E702
    assert m.compute_wrench(_zero(1), _zero(1), omega=front).torque[0, 1] < 0  # nose up = negative pitch (FLU)
    cw = base.clone(); cw[0, [2, 3]] *= 1.05  # noqa: E702
    assert m.compute_wrench(_zero(1), _zero(1), omega=cw).torque[0, 2] > 0  # CW props -> +yaw reaction


def test_motor_first_order_response():
    p = make_x500()
    m = QuadrotorModel(p, 1)
    m.reset(omega=torch.zeros(1, 4))
    dt = 1e-4
    target = torch.full((1, 4), 800.0)
    for _ in range(int(round(p.tau_up / dt))):
        m.step_motors(target, dt)
    assert m.omega[0, 0].item() == pytest.approx(800.0 * (1 - math.exp(-1)), rel=1e-3)
    # decelerate: uses tau_down
    m.reset(omega=torch.full((1, 4), 800.0))
    for _ in range(int(round(p.tau_down / dt))):
        m.step_motors(torch.zeros(1, 4), dt)
    assert m.omega[0, 0].item() == pytest.approx(800.0 * math.exp(-1), rel=1e-3)


def test_motor_exact_discretization_dt_invariant():
    p = make_x500()
    res = []
    for dt in (1e-3, 5e-3):
        m = QuadrotorModel(p, 1)
        m.reset(omega=torch.full((1, 4), 300.0))
        for _ in range(int(round(0.05 / dt))):
            m.step_motors(torch.full((1, 4), 900.0), dt)
        res.append(m.omega.clone())
    assert torch.allclose(res[0], res[1], rtol=1e-5)


def test_motor_saturation():
    p = make_x500()
    m = QuadrotorModel(p, 1)
    for _ in range(2000):
        m.step_motors(torch.full((1, 4), 5000.0), 1e-3)
    assert torch.all(m.omega <= p.omega_max + 1e-4)


def test_rotor_drag_and_body_drag():
    p = make_x500()
    m = QuadrotorModel(p, 1)
    m.reset()
    v = torch.tensor([[2.0, 0.0, 0.0]])
    w = m.compute_wrench(v, _zero(1))
    expected_rd = -p.rotor_drag_coeff * m.omega.sum() * 2.0
    assert w.rotor_drag[0, 0].item() == pytest.approx(expected_rd.item(), rel=1e-5)
    assert w.body_drag[0, 0].item() == pytest.approx(-p.body_drag_quad[0] * 4.0, rel=1e-6)
    # pure vertical motion: no rotor drag
    w = m.compute_wrench(torch.tensor([[0.0, 0.0, 1.0]]), _zero(1))
    assert w.rotor_drag.abs().max() < 1e-9
    # wind equal to velocity -> zero aerodynamic force
    w = m.compute_wrench(v, _zero(1), wind_b=v)
    assert w.rotor_drag.abs().max() < 1e-9 and w.body_drag.abs().max() < 1e-9


def test_ground_effect_factor():
    p = make_x500()
    m = QuadrotorModel(p, 1)
    r = p.rotor_radius
    f = m.ground_effect_factor(torch.full((1, 4), r))
    assert f[0, 0].item() == pytest.approx(1.0 / (1.0 - 1.0 / 16.0), rel=1e-6)
    assert m.ground_effect_factor(torch.full((1, 4), 50 * r))[0, 0].item() == pytest.approx(1.0, abs=1e-4)
    assert m.ground_effect_factor(torch.zeros(1, 4)).max().item() == pytest.approx(4.0 / 3.0, rel=1e-6)


def test_gyroscopic_torque():
    p = make_x500()
    m = QuadrotorModel(p, 1)
    m.reset()
    m.rolling_moment_coeff[:] = 0.0  # isolate the gyroscopic term (rolling moment also acts about y here)
    # balanced rotors: zero net angular momentum -> no gyro torque
    w = m.compute_wrench(_zero(1), torch.tensor([[1.0, 0.0, 0.0]]))
    assert abs(w.torque[0, 1].item()) < 1e-9
    # CCW rotors faster -> net +z rotor momentum h; tau = -w x h. For w=+x, h=+z: -(x cross z) = +y
    omega = m.omega.clone()
    omega[0, [0, 1]] += 50.0
    w1 = m.compute_wrench(_zero(1), torch.tensor([[1.0, 0.0, 0.0]]), omega=omega)
    w0 = m.compute_wrench(_zero(1), _zero(1), omega=omega)
    h = p.rotor_inertia * (omega[0, 0] + omega[0, 1] - omega[0, 2] - omega[0, 3])
    assert (w1.torque[0, 1] - w0.torque[0, 1]).item() == pytest.approx(h.item(), rel=1e-4)
