import math

import numpy as np
import torch

from quadsim.core import math as qm
from quadsim.deploy.interface import (
    POSE_FRAME_NED,
    VELOCITY_FRAME_BODY_FRD,
    VELOCITY_FRAME_NED,
    ActionDecoder,
    ObservationBuilder,
    enu_to_ned,
    flu_to_frd,
    px4_odometry_to_sim,
    px4_topic,
    px4_vehicle_odometry_to_sim,
    q_conj,
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


def test_px4_topic_versioning():
    assert px4_topic("/fmu/out/vehicle_status", 1) == "/fmu/out/vehicle_status_v1"  # PX4 v1.16 VehicleStatus
    assert px4_topic("/fmu/out/vehicle_odometry", 0) == "/fmu/out/vehicle_odometry"
    assert px4_topic("/fmu/in/vehicle_rates_setpoint") == "/fmu/in/vehicle_rates_setpoint"


def test_px4_vehicle_odometry_frames_and_invalid():
    rng = np.random.default_rng(3)
    q = _rand_q(1, rng)[0]
    pos, vel_ned, w = [1.0, 2.0, -3.0], np.array([0.5, -0.2, 0.1]), [0.1, 0.2, 0.3]
    ref = px4_odometry_to_sim(pos, q, vel_ned, w)
    out = px4_vehicle_odometry_to_sim(pos, q, vel_ned, w, POSE_FRAME_NED, VELOCITY_FRAME_NED)
    assert all(np.allclose(a, b) for a, b in zip(out, ref))
    # body-FRD velocity is rotated into NED before conversion
    vel_frd = q_rotate(q_conj(q), vel_ned)
    out = px4_vehicle_odometry_to_sim(pos, q, vel_frd, w, POSE_FRAME_NED, VELOCITY_FRAME_BODY_FRD)
    assert np.allclose(out[2], ref[2], atol=1e-12)
    # unusable: heading-free FRD pose frame, unknown velocity frame, NaN fields (PX4's "invalid")
    assert px4_vehicle_odometry_to_sim(pos, q, vel_ned, w, 2, VELOCITY_FRAME_NED) is None
    assert px4_vehicle_odometry_to_sim(pos, q, vel_ned, w, POSE_FRAME_NED, 0) is None
    assert px4_vehicle_odometry_to_sim(pos, [float("nan"), 0, 0, 0], vel_ned, w) is None
    assert px4_vehicle_odometry_to_sim(pos, q, [float("nan")] * 3, w) is None


def test_thrust_curve_inverse_and_calibration():
    from quadsim.deploy.interface import ThrustCurve

    lin = ThrustCurve()
    u = np.linspace(0, 1, 11)
    assert np.allclose(lin.inverse(u), u, atol=1e-9) and abs(lin.calibrate(0.7295, 0.592) - 0.7295 / 0.592) < 1e-12
    gz = ThrustCurve([0.0225, 0.255, 0.7225])  # (0.15 + 0.85 u)^2
    assert np.allclose(gz(u), (0.15 + 0.85 * u) ** 2)
    assert np.allclose(gz.inverse(gz(u)), u, atol=1e-8)
    assert gz.inverse(-1.0) == 0.0 and gz.inverse(2.0) == 1.0  # clipped to the achievable range
    try:
        ThrustCurve([0.0, 1.0, -1.0])  # decreasing above u = 0.5
        raise AssertionError("non-monotonic curve accepted")
    except ValueError:
        pass


def test_action_decoder_thrust_curve_makes_thrust_linear():
    from quadsim.deploy.interface import ThrustCurve

    t_max = 34.19432
    contract = {"action": {"control_mode": "ctbr", "collective_range": [0.0, 1.0], "max_total_thrust_N": t_max,
                           "max_body_rate_rad_s": [6.0, 6.0, 3.0], "max_velocity_m_s": [3, 3, 2],
                           "max_yaw_rate_rad_s": 2.0}}
    dec, gz = ActionDecoder(contract), ThrustCurve([0.0225, 0.255, 0.7225])
    hover_frac = 2.0643076923076924 * 9.81 / t_max
    s = gz.calibrate(0.7289, hover_frac)
    a0 = np.linspace(-0.6, 0.9, 16)
    acts = np.stack([a0, 0 * a0, 0 * a0, 0 * a0], -1)
    collective = dec.decode(acts)["collective_N"]
    u = -dec.to_px4(acts, thrust_scale=s, thrust_curve=gz)["thrust_body"][..., 2]
    # thrust the gz x500 produces at u (T_ref = 4 k_f omega_max^2 = t_max) is proportional to the commanded collective
    assert np.allclose(t_max * gz(u), s * collective, rtol=1e-7)
    # the linear map is exact only at hover: local gain of the real curve there is ~1.6x the linear map's
    u_lin = -dec.to_px4(acts, thrust_scale=0.7289 / hover_frac)["thrust_body"][..., 2]
    gain_ratio = (2 * 0.85 * (0.15 + 0.85 * 0.7289)) / (gz(0.7289) / 0.7289)
    assert 1.55 < gain_ratio < 1.65
    # linear map saturates (u = 1) above 1 / thrust_scale = 81 % of T_max; the curve keeps the full range
    assert u_lin[-1] == 1.0 and np.all(np.diff(u_lin) >= 0) and u[-1] < 1.0 and np.all(np.diff(u) > 0)


def test_sinusoid_reference_shapes_and_derivatives():
    from quadsim.deploy.interface import SinusoidReference

    start = np.array([1.0, -2.0, 2.5])
    circ = SinusoidReference.shape("circle", 1.5, 0.2, z_amp=0.3, start=start)
    t = np.linspace(0, 10, 2001)
    p, v, a = circ(t)
    assert np.allclose(p[0], start)
    c_xy = circ.center[:2]
    assert np.allclose(np.linalg.norm(p[:, :2] - c_xy, axis=-1), 1.5)  # constant radius
    dt = t[1] - t[0]
    assert np.allclose(np.gradient(p, dt, axis=0)[1:-1], v[1:-1], atol=1e-3)
    assert np.allclose(np.gradient(v, dt, axis=0)[1:-1], a[1:-1], atol=1e-3)
    f8 = SinusoidReference.shape("figure8", 2.0, 0.1, start=start)
    p8, _, _ = f8(t)
    assert np.allclose(p8[0], start) and abs(np.ptp(p8[:, 0]) - 4.0) < 1e-3 and abs(np.ptp(p8[:, 1]) - 2.0) < 1e-2


def test_track_and_waypoint_task_obs():
    rng = np.random.default_rng(5)
    q = _rand_q(1, rng)[0]
    pos = np.array([0.3, -0.2, 1.8])
    ref_h = rng.normal(size=(6, 3))
    v0 = rng.normal(size=3)
    o = ObservationBuilder.track_task(pos, q, ref_h, v0)
    ref = np.concatenate([q_rotate(q_conj(q), r - pos) for r in ref_h] + [q_rotate(q_conj(q), v0)])
    assert o.shape == (21,) and np.allclose(o, ref)
    wp0, wp1 = np.array([10.0, 0.0, 2.0]), np.array([10.0, 1.0, 2.0])
    o = ObservationBuilder.waypoint_task(pos, np.array([1.0, 0, 0, 0]), wp0, wp1, 0.6)
    assert np.isclose(np.linalg.norm(o[:3]), 5.0)  # clipped to max_obs_distance
    assert np.allclose(o[3:6], [0.0, 1.0, 0.0]) and o[6] == 0.6


def test_waypoint_sequence_logic():
    from quadsim.deploy.interface import WaypointSequence

    s = WaypointSequence([[0, 0, 1], [1, 0, 1], [2, 0, 1]], reach_radius=0.3)
    assert s.remaining_frac() == 1.0 and np.allclose(s.next(), [1, 0, 1])
    assert not s.update(np.array([0.0, 0.31, 1.0]))
    assert s.update(np.array([0.0, 0.29, 1.0])) and s.idx == 1 and np.isclose(s.remaining_frac(), 2 / 3)
    assert s.update(np.array([1.0, 0.0, 1.0])) and s.idx == 2 and np.allclose(s.next(), [2, 0, 1])
    assert s.update(np.array([2.0, 0.0, 1.0])) and s.done and s.idx == 2
    assert not s.update(np.array([2.0, 0.0, 1.0]))  # finished: nothing more to reach


def test_ring_from_scan_resampling():
    from quadsim.deploy.interface import ring_from_scan

    n = 360  # 1 deg LaserScan from -pi, CCW
    ang = -np.pi + np.arange(n) * (2 * np.pi / n)
    ranges = 2.0 + np.cos(ang)  # nearest at the back, farthest ahead
    ranges[(np.abs(ang - np.pi / 2) < 1e-6)] = np.inf  # no return at +90 deg (left)
    ring = ring_from_scan(ranges, -np.pi, 2 * np.pi / n, num_rays=32, range_max=8.0)
    az = 2 * np.pi * np.arange(32) / 32
    exp = 2.0 + np.cos(az)
    exp[8] = 8.0  # ray 8 = +90 deg reads max range
    assert np.allclose(ring, exp, atol=0.01)  # nearest 1-deg beam: |d angle| <= 0.5 deg -> |d range| < 0.009 m
    assert np.allclose(ring_from_scan(np.full(16, 20.0), 0.0, np.pi / 8, 32, 8.0), 8.0)  # clipped to range_max
