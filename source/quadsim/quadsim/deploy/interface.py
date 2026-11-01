"""Framework-free (NumPy) deployment interface: frame conversions, observation builder, action decoder.

Simulation conventions: world z-up (ENU-like), body FLU, quaternion wxyz ``q_WB``.
PX4 conventions:        world NED, body FRD, quaternion wxyz ``q_NED_FRD`` (VehicleOdometry.q).

This module reproduces, outside Isaac Lab, exactly the observation the policy saw in simulation
(see ``QuadrotorEnv._state_obs`` and the tasks' ``_task_obs``) and decodes normalized actions into
PX4 offboard set-points. It is unit-tested against the simulator's own observations
(tests/core/test_deploy_interface.py and scripts/validation/m6_deploy_consistency.py).
"""

from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------------------- quaternions (wxyz)


def q_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def q_conj(q):
    return np.concatenate([q[..., :1], -q[..., 1:]], -1)


def q_rotate(q, v):
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def q_to_rot(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def yaw_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ----------------------------------------------------------------------------- frames
_S = 1.0 / math.sqrt(2.0)
Q_NED_ENU = np.array([0.0, _S, _S, 0.0])  # rotates ENU vectors into NED
Q_FLU_FRD = np.array([0.0, 1.0, 0.0, 0.0])  # rotates FRD vectors into FLU (180 deg about x)


def enu_to_ned(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 1], v[..., 0], -v[..., 2]], -1)


ned_to_enu = enu_to_ned  # the map is an involution


def flu_to_frd(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], -v[..., 1], -v[..., 2]], -1)


frd_to_flu = flu_to_frd


def q_enu_flu_to_ned_frd(q):
    """q_WB (ENU, FLU) -> q (NED, FRD) as published by PX4 VehicleOdometry."""
    return q_mul(q_mul(np.broadcast_to(Q_NED_ENU, q.shape), q), np.broadcast_to(Q_FLU_FRD, q.shape))


def q_ned_frd_to_enu_flu(q):
    return q_mul(q_mul(np.broadcast_to(q_conj(Q_NED_ENU), q.shape), q), np.broadcast_to(q_conj(Q_FLU_FRD), q.shape))


def px4_odometry_to_sim(pos_ned, q_ned_frd, vel_ned, ang_vel_frd):
    """PX4 VehicleOdometry (NED/FRD, velocity in NED) -> simulation conventions (ENU/FLU)."""
    return (ned_to_enu(pos_ned), q_ned_frd_to_enu_flu(np.asarray(q_ned_frd, dtype=np.float64)),
            ned_to_enu(vel_ned), frd_to_flu(ang_vel_frd))


# VehicleOdometry frame enums (px4_msgs, PX4 v1.14 - v1.16)
POSE_FRAME_NED = 1
VELOCITY_FRAME_NED, VELOCITY_FRAME_BODY_FRD = 1, 3


def px4_vehicle_odometry_to_sim(pos, q, vel, ang_vel, pose_frame: int = POSE_FRAME_NED,
                                velocity_frame: int = VELOCITY_FRAME_NED):
    """Like :func:`px4_odometry_to_sim`, honouring ``VehicleOdometry.pose_frame / velocity_frame``.

    Returns ``None`` when the message cannot be used: unsupported frame (the policy needs a gravity- and
    heading-referenced NED pose) or any NaN field (PX4 marks invalid estimates with NaN).
    """
    pos, q, vel, ang_vel = (np.asarray(x, dtype=np.float64) for x in (pos, q, vel, ang_vel))
    if pose_frame != POSE_FRAME_NED or velocity_frame not in (VELOCITY_FRAME_NED, VELOCITY_FRAME_BODY_FRD):
        return None
    if not all(np.all(np.isfinite(x)) for x in (pos, q, vel, ang_vel)):
        return None
    if velocity_frame == VELOCITY_FRAME_BODY_FRD:
        vel = q_rotate(q / np.linalg.norm(q), vel)  # FRD body -> NED
    return px4_odometry_to_sim(pos, q, vel, ang_vel)


def px4_topic(base: str, message_version: int = 0) -> str:
    """uXRCE-DDS topic name of a PX4 message: PX4 >= 1.16 appends ``_v<N>`` for message versions N > 0
    (e.g. ``VehicleStatus`` v1 -> ``/fmu/out/vehicle_status_v1``; see uxrce_dds_client/utilities.hpp)."""
    return f"{base}_v{int(message_version)}" if message_version else base


# ----------------------------------------------------------------------------- observation
class ObservationBuilder:
    """Builds the policy observation from onboard estimates, following ``contract.json``."""

    def __init__(self, contract: dict):
        self.c = contract
        self.a_dim = contract["action"]["dim"]
        self.has_last_action = any(x[0] == "last_action" for x in contract["observation"]["layout"])
        self.last_action = np.asarray(contract["action"]["hover_action"], dtype=np.float32)

    def state_obs(self, pos_w, q_wb, vel_w, gyro_b):
        r = q_to_rot(q_wb)
        vel_b = q_rotate(q_conj(q_wb), vel_w)
        parts = [vel_b, r[..., :, :2].reshape(r.shape[:-2] + (6,)), gyro_b]
        if self.has_last_action:
            parts.append(np.broadcast_to(self.last_action, np.shape(vel_b)[:-1] + (self.a_dim,)))
        return np.concatenate(parts, -1)

    # ---- task parts
    @staticmethod
    def hover_task(pos_w, q_wb, target_pos, target_yaw):
        rel_b = q_rotate(q_conj(q_wb), target_pos - pos_w)
        e = np.arctan2(np.sin(target_yaw - yaw_of(q_wb)), np.cos(target_yaw - yaw_of(q_wb)))
        return np.concatenate([rel_b, np.sin(e)[..., None], np.cos(e)[..., None]], -1)

    @staticmethod
    def track_task(pos_w, q_wb, ref_pos_h, ref_vel_now):
        """``TrackingEnv._task_obs``: reference positions at the look-ahead instants (H, 3) relative to the vehicle and
        the current reference velocity, both in the estimated body frame -> (3 H + 3,)."""
        q_inv = q_conj(q_wb)
        rel = np.asarray(ref_pos_h, dtype=np.float64) - np.asarray(pos_w)[..., None, :]
        rel_b = q_rotate(np.broadcast_to(q_inv[..., None, :], rel.shape[:-1] + (4,)), rel)
        return np.concatenate([rel_b.reshape(rel_b.shape[:-2] + (-1,)), q_rotate(q_inv, ref_vel_now)], -1)

    @staticmethod
    def waypoint_task(pos_w, q_wb, wp_current, wp_next, remaining_frac, max_obs_distance=5.0):
        """``WaypointEnv._task_obs``: current waypoint relative to the vehicle and next waypoint relative to the current
        one (body frame, clipped to ``max_obs_distance``), fraction of waypoints remaining -> (7,)."""
        q_inv = q_conj(q_wb)

        def clip(rel):
            d = np.linalg.norm(rel, axis=-1, keepdims=True)
            return rel * np.minimum(max_obs_distance / np.maximum(d, 1e-6), 1.0)

        rel0 = clip(q_rotate(q_inv, np.asarray(wp_current) - pos_w))
        rel1 = clip(q_rotate(q_inv, np.asarray(wp_next) - np.asarray(wp_current)))
        rem = np.asarray(remaining_frac, dtype=np.float64).reshape(rel0.shape[:-1] + (1,))
        return np.concatenate([rel0, rel1, rem], -1)

    @staticmethod
    def nav_task(pos_w, q_wb, goal, range_ring=None, range_max=8.0):
        rel_b = q_rotate(q_conj(q_wb), goal - pos_w)
        d = np.linalg.norm(rel_b, axis=-1, keepdims=True)
        parts = [rel_b / np.maximum(d, 1e-6), np.minimum(d, 10.0) / 10.0]
        if range_ring is not None:
            parts.append(np.asarray(range_ring) / range_max)
        return np.concatenate(parts, -1)

    def build(self, state_parts, task_part):
        return np.concatenate([state_parts, task_part], -1).astype(np.float32)

    def update_last_action(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)


# ----------------------------------------------------------------------------- task references / sensing
class SinusoidReference:
    """``p(t) = c + sum_k A_k sin(w_k t + phi_k)`` per axis (``TrackingEnv.reference``); amp/freq/phase (K, 3), freq in
    rad/s, world ENU."""

    def __init__(self, amp, freq, phase, center):
        self.amp, self.freq, self.phase = (np.asarray(x, dtype=np.float64).reshape(-1, 3) for x in (amp, freq, phase))
        self.center = np.asarray(center, dtype=np.float64)

    def __call__(self, t):
        """t (...) -> pos, vel, acc (..., 3)."""
        tt = np.asarray(t, dtype=np.float64)[..., None, None]
        arg = self.freq * tt + self.phase
        pos = (self.amp * np.sin(arg)).sum(-2) + self.center
        vel = (self.amp * self.freq * np.cos(arg)).sum(-2)
        acc = (-self.amp * self.freq ** 2 * np.sin(arg)).sum(-2)
        return pos, vel, acc

    @classmethod
    def shape(cls, kind: str, radius: float, freq_hz: float, z_amp: float = 0.0, start=(0.0, 0.0, 2.0)):
        """Circle / figure-8 of ``TrackingEnv._sample_reference`` (vertical motion on the last harmonic), shifted
        so that ``p(0) = start``."""
        w = 2.0 * np.pi * freq_hz
        amp, freq, phase = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
        if kind == "circle":
            amp[0, :2], freq[0, :2], phase[0, 0] = radius, w, np.pi / 2
        elif kind == "figure8":
            amp[0, 0], amp[0, 1], freq[0, 0], freq[0, 1] = radius, 0.5 * radius, w, 2 * w
        else:
            raise ValueError(kind)
        amp[-1, 2], freq[-1, 2] = z_amp, w
        ref = cls(amp, freq, phase, np.zeros(3))
        ref.center = np.asarray(start, dtype=np.float64) - ref(0.0)[0]
        return ref


class WaypointSequence:
    """Waypoint index logic of ``WaypointEnv``: advance when closer than ``reach_radius``; finished after the last."""

    def __init__(self, waypoints, reach_radius: float = 0.3):
        self.wps = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
        self.reach_radius = reach_radius
        self.idx, self.done = 0, False

    def update(self, pos) -> bool:
        """Returns True when a waypoint was reached at this call."""
        if self.done or np.linalg.norm(self.wps[self.idx] - pos) >= self.reach_radius:
            return False
        if self.idx == len(self.wps) - 1:
            self.done = True
        else:
            self.idx += 1
        return True

    def current(self):
        return self.wps[self.idx]

    def next(self):
        return self.wps[min(self.idx + 1, len(self.wps) - 1)]

    def remaining_frac(self) -> float:
        return (len(self.wps) - self.idx) / len(self.wps)


def ring_from_scan(ranges, angle_min: float, angle_increment: float, num_rays: int = 32, range_max: float = 8.0):
    """LaserScan (angles CCW about +z of a yaw-aligned FLU frame) -> the training range ring: rays at azimuths
    ``2 pi k / num_rays`` (``lidar_ray_dirs``), nearest beam, non-finite / out-of-range returns read as
    ``range_max``."""
    r = np.asarray(ranges, dtype=np.float64)
    beam_az = angle_min + angle_increment * np.arange(len(r))
    az = 2.0 * np.pi * np.arange(num_rays) / num_rays
    diff = np.abs((az[:, None] - beam_az[None, :] + np.pi) % (2.0 * np.pi) - np.pi)
    out = r[np.argmin(diff, axis=1)]
    return np.where(np.isfinite(out) & (out > 0.0), np.minimum(out, range_max), range_max)


# ----------------------------------------------------------------------------- actions
class ThrustCurve:
    """Normalized thrust curve ``c(u) = T(u) / T_ref`` of the vehicle's actuators, ``u`` = PX4 normalized thrust.

    Polynomial with ascending coefficients, required to be increasing on [0, 1]. ``[0, 1]`` is the linear map.
    PX4 SITL gz ``x500`` (ESC map omega = 150 + 850 u, thrust ~ omega^2): ``c(u) = (0.15 + 0.85 u)^2`` =
    ``[0.0225, 0.255, 0.7225]``. For a real vehicle, fit it on a thrust stand.
    """

    def __init__(self, coeffs=(0.0, 1.0)):
        self.p = np.polynomial.Polynomial(np.asarray(coeffs, dtype=np.float64))
        u = np.linspace(0.0, 1.0, 201)
        if not np.all(self.p.deriv()(u) > 0):
            raise ValueError(f"thrust curve {list(coeffs)} is not increasing on [0, 1]")
        self.lo, self.hi = float(self.p(0.0)), float(self.p(1.0))

    def __call__(self, u):
        return self.p(np.asarray(u, dtype=np.float64))

    def inverse(self, y):
        """u in [0, 1] with c(u) = y (clipped to [c(0), c(1)]); vectorized bisection, |err| < 1e-9."""
        y = np.clip(np.asarray(y, dtype=np.float64), self.lo, self.hi)
        a, b = np.zeros_like(y), np.ones_like(y)
        for _ in range(32):
            m = 0.5 * (a + b)
            below = self.p(m) < y
            a, b = np.where(below, m, a), np.where(below, b, m)
        return np.where(y <= self.lo, 0.0, np.where(y >= self.hi, 1.0, 0.5 * (a + b)))

    def calibrate(self, u_hover: float, sim_hover_fraction: float) -> float:
        """``thrust_scale`` that maps the simulated hover thrust to the measured hover command ``u_hover``."""
        return float(self(u_hover)) / sim_hover_fraction


LINEAR_THRUST = ThrustCurve()


class ActionDecoder:
    """Normalized action -> physical set-points (simulation frames) and PX4 offboard messages."""

    def __init__(self, contract: dict):
        a = contract["action"]
        self.mode = a["control_mode"]
        self.lo, self.hi = a["collective_range"]
        self.t_max = a["max_total_thrust_N"]
        self.max_rate = np.asarray(a["max_body_rate_rad_s"])
        self.max_vel = np.asarray(a["max_velocity_m_s"])
        self.max_yaw_rate = a["max_yaw_rate_rad_s"]
        self.pos_center = np.asarray(a.get("position_center_m", (0.0, 0.0, 0.0)))
        self.pos_range = np.asarray(a.get("position_range_m", (0.0, 0.0, 0.0)))

    def decode(self, action, q_wb=None) -> dict:
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        if self.mode == "ctbr":
            thrust = (self.lo + (a[..., 0] + 1) * 0.5 * (self.hi - self.lo)) * self.t_max
            return {"collective_N": thrust, "body_rates_flu": a[..., 1:4] * self.max_rate}
        if self.mode == "velocity":
            v_h = a[..., :3] * self.max_vel
            yaw = yaw_of(q_wb)
            c, s = np.cos(yaw), np.sin(yaw)
            v_w = np.stack([c * v_h[..., 0] - s * v_h[..., 1], s * v_h[..., 0] + c * v_h[..., 1], v_h[..., 2]], -1)
            return {"velocity_world": v_w, "yaw_rate": a[..., 3] * self.max_yaw_rate}
        if self.mode == "position":  # set-point in the task's local frame (z-up), relative to its origin
            return {"position_local": self.pos_center + a[..., :3] * self.pos_range,
                    "yaw_rate": a[..., 3] * self.max_yaw_rate}
        raise NotImplementedError(f"decoding '{self.mode}' actions (supported: ctbr, velocity, position)")

    def to_px4(self, action, q_wb=None, thrust_scale: float = 1.0, thrust_curve: ThrustCurve = LINEAR_THRUST) -> dict:
        """PX4 offboard set-point fields.

        CTBR -> ``VehicleRatesSetpoint``: roll/pitch/yaw rates in FRD and ``thrust_body = [0, 0, -T_n]``
        with ``T_n = c^-1(thrust_scale * collective / T_max)`` (PX4 normalized thrust, ``c`` = ``thrust_curve``).
        With the default linear curve ``T_n = thrust_scale * collective / T_max``: ``thrust_scale`` then absorbs the
        real thrust curve only at hover. Calibrate ``thrust_scale = c(u_hover) / (m g / T_max_sim)`` with PX4's
        hover-thrust estimate ``u_hover`` (:meth:`ThrustCurve.calibrate`, docs/SIM2REAL.md).
        Velocity -> ``TrajectorySetpoint``: velocity NED, yawspeed (NED: positive clockwise from above).
        Position -> ``TrajectorySetpoint``: position NED (task-local origin = PX4 local origin), yawspeed.
        """
        d = self.decode(action, q_wb)
        if self.mode == "ctbr":
            t_n = np.clip(thrust_curve.inverse(thrust_scale * d["collective_N"] / self.t_max), 0.0, 1.0)
            rates = flu_to_frd(d["body_rates_flu"])
            return {"msg": "VehicleRatesSetpoint", "roll": rates[..., 0], "pitch": rates[..., 1],
                    "yaw": rates[..., 2], "thrust_body": np.stack([0 * t_n, 0 * t_n, -t_n], -1)}
        if self.mode == "position":
            return {"msg": "TrajectorySetpoint", "position": enu_to_ned(d["position_local"]), "yawspeed": -d["yaw_rate"],
                    "velocity": np.full(3, np.nan), "acceleration": np.full(3, np.nan)}
        v = enu_to_ned(d["velocity_world"])
        return {"msg": "TrajectorySetpoint", "velocity": v, "yawspeed": -d["yaw_rate"],
                "position": np.full(3, np.nan), "acceleration": np.full(3, np.nan)}
