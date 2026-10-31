import math

import numpy as np
import torch

from quadsim.core import math as qm
from quadsim.deploy.interface import (
    ActionDecoder,
    ObservationBuilder,
    enu_to_ned,
    flu_to_frd,
    px4_odometry_to_sim,
    q_enu_flu_to_ned_frd,
    q_ned_frd_to_enu_flu,
    q_rotate,
    q_to_rot,
)


def _rand_q(n, rng):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_frame_round_trips():
    rng = np.random.default_rng(0)
    q = _rand_q(100, rng)
    q2 = q_ned_frd_to_enu_flu(q_enu_flu_to_ned_frd(q))
    assert np.allclose(q2 * np.sign((q2 * q).sum(-1, keepdims=True)), q, atol=1e-12)
    v = rng.normal(size=(100, 3))
    assert np.allclose(enu_to_ned(enu_to_ned(v)), v)
    assert np.allclose(flu_to_frd(flu_to_frd(v)), v)


def test_quaternion_frame_consistency():
    """R_NED_FRD v_FRD == enu_to_ned(R_ENU_FLU v_FLU) for the same physical body vector."""
    rng = np.random.default_rng(1)
    q = _rand_q(100, rng)
    v_flu = rng.normal(size=(100, 3))
    lhs = q_rotate(q_enu_flu_to_ned_frd(q), flu_to_frd(v_flu))
    rhs = enu_to_ned(q_rotate(q, v_flu))
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_level_heading_north_conventions():
    # Facing north (ENU yaw = +90 deg) and level: PX4 reports yaw 0 (heading north) and identity-like FRD
    q_enu = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    q_ned = q_enu_flu_to_ned_frd(q_enu)
    r = q_to_rot(q_ned)
    assert np.allclose(r, np.eye(3), atol=1e-12)


def test_px4_odometry_conversion():
    pos, q, vel, w = px4_odometry_to_sim([1.0, 2.0, -3.0], [1.0, 0, 0, 0], [0.5, 0.0, -1.0], [0.1, 0.2, 0.3])
    assert np.allclose(pos, [2.0, 1.0, 3.0]) and np.allclose(vel, [0.0, 0.5, 1.0])
    assert np.allclose(w, [0.1, -0.2, -0.3])
    # PX4 identity attitude = nose north, level -> sim yaw +90 deg
    assert abs(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)) - math.pi / 2) < 1e-12


def test_state_obs_matches_torch_env_formula():
    rng = np.random.default_rng(2)
    n = 20
    q = _rand_q(n, rng)
    vel = rng.normal(size=(n, 3))
    gyro = rng.normal(size=(n, 3))
    contract = {"action": {"dim": 4, "hover_action": [0.1, 0, 0, 0], "control_mode": "ctbr"},
                "observation": {"layout": [["last_action", 4, ""]]}}
    ob = ObservationBuilder(contract)
    o = ob.state_obs(np.zeros((n, 3)), q, vel, gyro)
    # torch reference identical to QuadrotorEnv._state_obs
    qt = torch.tensor(q, dtype=torch.float64)
    rot = qm.quat_to_rotmat(qt)
    ref = torch.cat([qm.quat_rotate_inverse(qt, torch.tensor(vel)), rot[..., :2].reshape(-1, 6), torch.tensor(gyro),
                     torch.tensor([0.1, 0, 0, 0], dtype=torch.float64).expand(n, 4)], -1).numpy()
    assert np.allclose(o, ref, atol=1e-12)


def test_action_decoder_ctbr_and_px4():
    contract = {"action": {"control_mode": "ctbr", "collective_range": [0.0, 1.0], "max_total_thrust_N": 34.19,
                           "max_body_rate_rad_s": [6.0, 6.0, 3.0], "max_velocity_m_s": [3, 3, 2],
                           "max_yaw_rate_rad_s": 2.0}}
    dec = ActionDecoder(contract)
    d = dec.decode(np.array([0.0, 0.5, -0.5, 1.0]))
    assert abs(d["collective_N"] - 17.095) < 1e-9
    assert np.allclose(d["body_rates_flu"], [3.0, -3.0, 3.0])
    m = dec.to_px4(np.array([0.0, 0.5, -0.5, 1.0]))
    assert np.allclose([m["roll"], m["pitch"], m["yaw"]], [3.0, 3.0, -3.0])
    assert np.allclose(m["thrust_body"], [0, 0, -0.5])


def test_action_decoder_velocity_px4():
    contract = {"action": {"control_mode": "velocity", "collective_range": [0, 1], "max_total_thrust_N": 1.0,
                           "max_body_rate_rad_s": [1, 1, 1], "max_velocity_m_s": [3.0, 3.0, 2.0],
                           "max_yaw_rate_rad_s": 2.0}}
    dec = ActionDecoder(contract)
    q_yaw90 = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])  # heading +y (north in ENU)
    m = dec.to_px4(np.array([1.0, 0.0, 0.5, 0.5]), q_yaw90)
    # forward 3 m/s while facing north -> NED velocity (3, 0, -1); +yaw rate (CCW) -> negative NED yawspeed
    assert np.allclose(m["velocity"], [3.0, 0.0, -1.0], atol=1e-12)
    assert abs(m["yawspeed"] + 1.0) < 1e-12
