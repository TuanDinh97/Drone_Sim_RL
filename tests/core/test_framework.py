"""Framework extension points: registries, vehicle files / generic frames, action interfaces (CPU, no Isaac)."""

import math
import types

import pytest
import torch

from quadsim.core.controllers import FlightController, PositionGains
from quadsim.core.controllers.cascade import AttitudeGains, RateGains
from quadsim.core.dynamics import ReferenceQuadrotorSim
from quadsim.core.model import QuadrotorModel
from quadsim.core.vehicles import (
    from_isaaclab_multirotor,
    get_vehicle,
    load_vehicle,
    multirotor_layout,
    params_from_dict,
    validate_params,
)
from quadsim.registry import ACTION_INTERFACES, VEHICLES, Registry, import_plugins


def _ctrl_cfg(**kw):
    c = dict(max_body_rate=(6.0, 6.0, 3.0), max_tilt=0.6, max_velocity=(3.0, 3.0, 2.0), max_yaw_rate=2.0,
             position_center=(0.0, 0.0, 1.5), position_range=(3.0, 3.0, 1.5), collective_range=(0.0, 1.0),
             rate_gains=RateGains(), attitude_gains=AttitudeGains(), position_gains=PositionGains())
    c.update(kw)
    return types.SimpleNamespace(**c)


def test_registry_basics():
    r = Registry("thing")
    r.register("a", 1)

    @r.register("b")
    def b():
        return 2

    assert r.get("a") == 1 and r.get("b")() == 2 and "a" in r and r.names() == ["a", "b"]
    with pytest.raises(KeyError, match="already registered"):
        r.register("a", 3)
    with pytest.raises(KeyError, match="Registered"):
        r.get("zzz")


def test_builtin_registries():
    assert {"x500", "crazyflie", "racer5", "hexa_x"} <= set(VEHICLES.names())
    assert {"motor", "ctbr", "ctatt", "velocity", "position"} <= set(ACTION_INTERFACES.names())


def test_quad_x_layout_matches_px4_order():
    pos, spin = multirotor_layout(4, math.sqrt(2) * 0.174, "x")
    x500 = get_vehicle("x500")
    for p, q in zip(pos, x500.rotor_positions):
        assert p[0] == pytest.approx(q[0]) and p[1] == pytest.approx(q[1])
    assert spin == x500.rotor_spin


@pytest.mark.parametrize("n,layout", [(6, "x"), (6, "plus"), (8, "x")])
def test_generic_frames_balanced(n, layout):
    pos, spin = multirotor_layout(n, 0.3, layout)
    p = torch.tensor(pos)
    assert torch.allclose(p.sum(0), torch.zeros(3), atol=1e-9)  # CoM-centered
    assert sum(spin) == 0  # yaw-balanced
    assert all(abs(math.hypot(q[0], q[1]) - 0.3) < 1e-9 for q in pos)


def test_vehicle_file_roundtrip(tmp_path):
    f = tmp_path / "my_hexa.yaml"
    f.write_text("base: x500\nframe: {num_rotors: 6, arm_length: 0.3}\nthrust_to_weight: 2.0\n"
                 "inertia: auto\nmotor_mass: 0.06\nmass: 2.2\n")
    p = load_vehicle(str(f))
    assert p.name == "my_hexa" and p.num_rotors == 6 and p.mass == 2.2
    assert p.thrust_to_weight == pytest.approx(2.0)
    assert p.inertia[2] > p.inertia[0] > 0  # planar mass distribution: Izz largest
    assert get_vehicle(str(f)).num_rotors == 6


def test_vehicle_validation_errors():
    with pytest.raises(KeyError, match="unknown vehicle fields"):
        params_from_dict({"base": "x500", "mas": 1.0})
    with pytest.raises(ValueError, match="thrust-to-weight"):
        params_from_dict({"base": "x500", "mass": 10.0})
    with pytest.raises(ValueError, match="triangle"):
        params_from_dict({"base": "x500", "inertia": [0.01, 0.01, 0.1]})
    with pytest.raises(ValueError, match="rank"):
        params_from_dict({"base": "x500", "rotor_positions": [[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0.4, 0, 0]]})


def test_isaaclab_multirotor_conversion():
    """ARL_ROBOT_1-like allocation matrix (Isaac Lab isaaclab_contrib) -> quadsim parameters."""
    th = types.SimpleNamespace(thrust_range=(0.1, 10.0), thrust_const_range=(9.26312e-06, 1.826312e-05),
                               tau_inc_range=(0.05, 0.08), tau_dec_range=(0.005, 0.005), torque_to_thrust_ratio=0.07)
    cfg = types.SimpleNamespace(actuators={"thrusters": th}, allocation_matrix=[
        [0.0] * 4, [0.0] * 4, [1.0] * 4,
        [-0.13, -0.13, 0.13, 0.13], [-0.13, 0.13, 0.13, -0.13], [-0.07, 0.07, -0.07, 0.07]])
    p = from_isaaclab_multirotor(cfg, mass=1.2, inertia=(0.01, 0.01, 0.018))
    c = sum(th.thrust_const_range) / 2
    assert p.k_f * (2 * math.pi) ** 2 == pytest.approx(c)
    assert p.max_thrust_per_rotor == pytest.approx(10.0)
    assert p.k_m == pytest.approx(0.07)
    # allocation reproduced: Mx = y T, My = -x T, Mz = -spin k_m T
    for i in range(4):
        x, y, _ = p.rotor_positions[i]
        assert y == pytest.approx(cfg.allocation_matrix[3][i]) and -x == pytest.approx(cfg.allocation_matrix[4][i])
        assert -p.rotor_spin[i] * p.k_m == pytest.approx(cfg.allocation_matrix[5][i])


@pytest.mark.parametrize("vehicle", ["x500", "hexa_x", "racer5", "crazyflie"])
@pytest.mark.parametrize("mode", ["motor", "ctbr", "ctatt", "velocity", "position"])
def test_hover_action_holds_hover(vehicle, mode):
    """The interface's hover action produces (near) weight-compensating thrust at hover for every vehicle."""
    p = get_vehicle(vehicle)
    cfg = _ctrl_cfg(position_center=(0.0, 0.0, 1.0))
    fc = FlightController(p, cfg, 2, 0.005, "cpu")
    iface = ACTION_INTERFACES.get(mode)(fc)
    a = iface.hover_action().expand(2, -1)
    assert a.shape[-1] == iface.action_dim(cfg, p)
    quat = torch.tensor([[1.0, 0, 0, 0]] * 2)
    est = torch.zeros(2, 13)
    est[:, 2] = 1.0
    est[:, 3] = 1.0
    fc.reset(torch.arange(2), quat, torch.zeros(2, 3))
    t = iface(a, quat, est, torch.zeros(2, 3))
    assert t.shape == (2, p.num_rotors)
    assert t.sum(-1) == pytest.approx(torch.full((2,), p.weight), rel=2e-3)


@pytest.mark.parametrize("vehicle", ["hexa_x", "racer5"])
def test_position_interface_closed_loop(vehicle):
    """Position interface + reference integrator: a hexarotor / racer flies to the set-point."""
    p = get_vehicle(vehicle)
    cfg = _ctrl_cfg(position_center=(0.0, 0.0, 0.0), position_range=(2.0, 2.0, 2.0))
    dt = 0.005
    fc = FlightController(p, cfg, 1, dt, "cpu")
    iface = ACTION_INTERFACES.get("position")(fc)
    sim = ReferenceQuadrotorSim(QuadrotorModel(p, 1, "cpu"), dt)
    sim.state.pos[:] = torch.tensor([[0.0, 0.0, 1.0]])
    sim.model.reset(None, p.hover_omega)
    fc.reset(torch.arange(1), sim.state.quat, torch.zeros(1, 3))
    a = torch.tensor([[0.5, -0.25, 0.75, 0.0]])  # target (1.0, -0.5, 1.5)
    for _ in range(int(6.0 / dt)):
        s = sim.state
        est = torch.cat([s.pos, s.quat, s.lin_vel, s.ang_vel], -1)
        thrust = iface(a, s.quat, est, s.ang_vel)
        sim.step(fc.mixer.thrust_to_omega(thrust))
    err = (sim.state.pos[0] - torch.tensor([1.0, -0.5, 1.5])).norm().item()
    assert err < 0.05, err


def test_import_plugins_from_file(tmp_path):
    f = tmp_path / "plug.py"
    f.write_text("from quadsim.registry import VEHICLES\nfrom quadsim.core.vehicles import make_racer5\n"
                 "VEHICLES.register('test_plugin_racer', make_racer5, overwrite=True)\n")
    import_plugins([str(f)])
    assert get_vehicle("test_plugin_racer").name == "racer5"
    validate_params(get_vehicle("test_plugin_racer"))


def test_overrides_after_list_options():
    """`--plugins my.py env.a=b` / `--controllers cascade env.a=b`: argparse attaches the override to the list."""
    import argparse

    from quadsim.rl.overrides import split_overrides

    p = argparse.ArgumentParser()
    p.add_argument("--task")
    p.add_argument("--plugins", nargs="*", default=[])
    p.add_argument("--controllers", nargs="+", default=["policy"])
    argv = ["--task", "T", "env.a=1", "--plugins", "x.py", "env.control.mode=acceleration", "--controllers",
            "cascade", "env.b=(1,2)", "agent.c=3"]
    args, overrides = p.parse_known_args(argv)
    assert "env.control.mode=acceleration" in args.plugins  # the argparse behaviour being corrected
    assert split_overrides(args, overrides, argv) == ["env.a=1", "env.control.mode=acceleration", "env.b=(1,2)",
                                                      "agent.c=3"]
    assert args.plugins == ["x.py"] and args.controllers == ["cascade"]


def test_mixer_asymmetric_frame_hovers_without_torque():
    """CoM off the geometric centre (short rear arms): the hover split must not produce a torque."""
    from quadsim.core.controllers.mixer import Mixer

    x500 = get_vehicle("x500")
    pos = tuple((x * (1.0 if x > 0 else 0.75), y, z) for x, y, z in x500.rotor_positions)
    p = x500.copy(name="x500_short_rear", rotor_positions=pos)
    m = Mixer(p)
    assert not m.symmetric and Mixer(x500).symmetric
    t = m.allocate(torch.tensor([p.weight]), torch.zeros(1, 3))
    w = m.wrench_from_thrusts(t)[0]
    assert w[0] == pytest.approx(p.weight, rel=1e-5) and w[1:].abs().max() < 1e-5
    cfg = _ctrl_cfg()
    fc = FlightController(p, cfg, 1, 0.005, "cpu")
    iface = ACTION_INTERFACES.get("motor")(fc)
    t_h = iface(iface.hover_action()[None], torch.tensor([[1.0, 0, 0, 0]]), torch.zeros(1, 13), torch.zeros(1, 3))
    assert m.wrench_from_thrusts(t_h)[0, 1:].abs().max() < 1e-5


def test_position_hold_asymmetric_frame():
    """Position interface on the asymmetric frame: holds the set-point without a saturated rate integrator."""
    x500 = get_vehicle("x500")
    pos = tuple((x * (1.0 if x > 0 else 0.75), y, z) for x, y, z in x500.rotor_positions)
    p = x500.copy(name="x500_short_rear", rotor_positions=pos)
    cfg = _ctrl_cfg(position_center=(0.0, 0.0, 0.0), position_range=(2.0, 2.0, 2.0))
    dt = 0.005
    fc = FlightController(p, cfg, 1, dt, "cpu")
    iface = ACTION_INTERFACES.get("position")(fc)
    sim = ReferenceQuadrotorSim(QuadrotorModel(p, 1, "cpu"), dt)
    sim.state.pos[:] = torch.tensor([[0.0, 0.0, 1.0]])
    sim.model.reset(None, p.hover_omega)
    fc.reset(torch.arange(1), sim.state.quat, torch.zeros(1, 3))
    a = torch.tensor([[0.0, 0.0, 0.5, 0.0]])  # hold (0, 0, 1)
    for _ in range(int(4.0 / dt)):
        s = sim.state
        thrust = iface(a, s.quat, torch.cat([s.pos, s.quat, s.lin_vel, s.ang_vel], -1), s.ang_vel)
        sim.step(fc.mixer.thrust_to_omega(thrust))
    assert (sim.state.pos[0] - torch.tensor([0.0, 0.0, 1.0])).norm().item() < 0.02
