import math

import torch

from quadsim.core import QuadrotorModel, ReferenceQuadrotorSim, make_crazyflie, make_x500
from quadsim.core import math as qm
from quadsim.core.controllers import (
    AttitudeController,
    GeometricController,
    Mixer,
    PositionController,
    RateController,
    thrust_vector_to_setpoints,
)


def test_mixer_exact_when_feasible():
    p = make_x500()
    mix = Mixer(p)
    f = torch.full((5,), p.weight)
    tau = torch.tensor([[0.1, -0.2, 0.05], [0, 0, 0], [0.3, 0.3, -0.1], [-0.2, 0.1, 0.02], [0.0, 0.0, 0.1]])
    t = mix.allocate(f, tau)
    w = mix.wrench_from_thrusts(t)
    assert torch.allclose(w[:, 0], f, rtol=1e-5)
    assert torch.allclose(w[:, 1:], tau, atol=1e-5)


def test_mixer_saturation_priorities():
    p = make_x500()
    mix = Mixer(p)
    # near-max collective with large yaw request: roll/pitch must be honoured, yaw reduced
    f = torch.tensor([0.95 * p.max_total_thrust])
    tau = torch.tensor([[0.5, 0.0, 1.0]])
    t = mix.allocate(f, tau)
    assert torch.all(t <= mix.t_max + 1e-6) and torch.all(t >= mix.t_min - 1e-6)
    w = mix.wrench_from_thrusts(t)
    assert abs(w[0, 1].item() - 0.5) < 1e-4  # roll preserved
    assert w[0, 3].item() < 1.0  # yaw reduced
    assert w[0, 3].item() >= 0.0


class Stack:
    """Full cascade: position -> attitude -> rate -> mixer -> motors -> reference rigid body."""

    def __init__(self, p, n, dt=1 / 200):
        self.p = p
        self.model = QuadrotorModel(p, n)
        self.model.reset()
        self.sim = ReferenceQuadrotorSim(self.model, dt=dt, method="rk4")
        self.pos_ctrl = PositionController(p, n, dt)
        self.att_ctrl = AttitudeController()
        self.rate_ctrl = RateController(p, n, dt)
        self.mixer = Mixer(p)
        self.dt = dt

    def step(self, pos_sp, yaw_sp):
        s = self.sim.state
        thrust_w = self.pos_ctrl(pos_sp, s.pos, s.lin_vel)
        coll, q_sp = thrust_vector_to_setpoints(thrust_w, s.quat, yaw_sp)
        rate_sp = self.att_ctrl(s.quat, q_sp)
        tau = self.rate_ctrl(rate_sp, s.ang_vel)
        thrusts = self.mixer.allocate(coll, tau)
        self.sim.step(self.mixer.thrust_to_omega(thrusts))


def test_cascade_position_step_x500():
    p = make_x500()
    st = Stack(p, 2)
    sp = torch.tensor([[1.0, -1.0, 1.0], [-2.0, 0.5, 2.0]])
    yaw = torch.tensor([0.5, -1.0])
    for _ in range(int(6.0 / st.dt)):
        st.step(sp, yaw)
    err = (st.sim.state.pos - sp).norm(dim=-1)
    assert err.max().item() < 0.05, err
    yaw_err = qm.wrap_to_pi(qm.yaw_from_quat(st.sim.state.quat) - yaw).abs()
    assert yaw_err.max().item() < 0.05
    assert st.sim.state.lin_vel.norm(dim=-1).max().item() < 0.05


def test_cascade_crazyflie_hover_recovery_from_tilt():
    p = make_crazyflie()
    st = Stack(p, 4)
    st.sim.state.quat = qm.random_quat_tilt(4, 0.8, "cpu", torch.Generator().manual_seed(0))
    st.sim.state.ang_vel = torch.tensor([[3.0, -2.0, 1.0]]).repeat(4, 1)
    sp = torch.zeros(4, 3)
    for _ in range(int(5.0 / st.dt)):
        st.step(sp, torch.zeros(4))
    assert (st.sim.state.pos - sp).norm(dim=-1).max().item() < 0.05


def test_geometric_controller_tracks_circle():
    p = make_x500()
    n, dt = 1, 1 / 200
    model = QuadrotorModel(p, n)
    model.reset()
    sim = ReferenceQuadrotorSim(model, dt=dt, method="rk4")
    ctrl = GeometricController(p)
    mixer = Mixer(p)
    r, w = 1.5, 1.2  # 1.8 m/s tangential speed
    sim.state.pos[:] = torch.tensor([r, 0.0, 1.0])
    sim.state.lin_vel[:] = torch.tensor([0.0, r * w, 0.0])
    errs = []
    for k in range(int(8.0 / dt)):
        t = k * dt
        pos_d = torch.tensor([[r * math.cos(w * t), r * math.sin(w * t), 1.0]])
        vel_d = torch.tensor([[-r * w * math.sin(w * t), r * w * math.cos(w * t), 0.0]])
        acc_d = torch.tensor([[-r * w * w * math.cos(w * t), -r * w * w * math.sin(w * t), 0.0]])
        s = sim.state
        f, tau = ctrl(s.pos, s.quat, s.lin_vel, s.ang_vel, pos_d, vel_d, acc_d, torch.zeros(1))
        sim.step(mixer.thrust_to_omega(mixer.allocate(f, tau)))
        if t > 2.0:
            errs.append((sim.state.pos - pos_d).norm().item())
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    assert rmse < 0.1, rmse
