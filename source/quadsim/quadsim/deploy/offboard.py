"""Framework-free core of the PX4 offboard bridge (``ros2/quadsim_px4_bridge``): one policy step.

The ROS 2 node only moves messages; every decision lives here so it is unit-tested without ROS
(tests/core/test_offboard.py):

* heartbeat gating: PX4 detects offboard loss only from the ``OffboardControlMode`` heartbeat
  (commander ``offboardCheck``: heartbeat older than ``COM_OF_LOSS_T``). The heartbeat is therefore sent only while
  the policy can command (fresh odometry, fresh exteroception if the task needs it, inside the geofence);
  otherwise PX4's failsafe takes over instead of holding the last set-point.
* while PX4 (not the policy) flies, the policy's ``last_action`` input is held at the hover action, the value it
  saw at every episode start in training, instead of commands that were never applied; task state (tracking time,
  waypoint index) starts at the hand-over.
* task observations mirror the training environments (hover, tracking, waypoint, navigation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .interface import ActionDecoder, ObservationBuilder, SinusoidReference, ThrustCurve, WaypointSequence

LOG_COLUMNS = ("t", "odom_stamp", "odom_rx", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy",
               "wz", "goal_x", "goal_y", "goal_z", "goal_yaw", "thrust_scale", "a0", "a1", "a2", "a3")
"""Bridge step log (:meth:`OffboardPolicy.log_row`): wall time of the step, odometry ``timestamp_sample`` (agent
clock) and receive time [s], ENU/FLU state used for the observation, task anchor (hover goal / current reference
point / current waypoint / nav goal), goal yaw, thrust scale, normalized action."""

# ----------------------------------------------------------------------------- tasks


class HoverTask:
    """``HoverEnv``: hold ``goal`` (ENU) with heading ``goal_yaw``."""

    def __init__(self, goal=(0.0, 0.0, 1.5), goal_yaw: float = 0.0):
        self.goal = np.asarray(goal, dtype=np.float64)
        self.goal_yaw = float(goal_yaw)

    def reset(self):
        pass

    def anchor(self, t):
        return self.goal

    def obs(self, b: ObservationBuilder, pos, q, t):
        return b.hover_task(pos, q, self.goal, self.goal_yaw)


class TrackTask:
    """``TrackingEnv``: follow ``reference`` (time from the hand-over), look-ahead ``horizon_s``."""

    def __init__(self, reference: SinusoidReference, horizon_s=(0.0, 0.1, 0.2, 0.4, 0.6, 1.0)):
        self.reference = reference
        self.horizon = np.asarray(horizon_s, dtype=np.float64)
        self.goal_yaw = 0.0

    def reset(self):
        pass

    def anchor(self, t):
        return self.reference(t)[0]

    def obs(self, b: ObservationBuilder, pos, q, t):
        p_h, _, _ = self.reference(t + self.horizon)
        _, v0, _ = self.reference(t)
        return b.track_task(pos, q, p_h, v0)


class WaypointTask:
    """``WaypointEnv``: fly through ``waypoints`` (reach radius, index logic and clipping as in training)."""

    def __init__(self, waypoints, reach_radius: float = 0.3, max_obs_distance: float = 5.0):
        self.waypoints = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
        self.reach_radius, self.max_obs_distance = reach_radius, max_obs_distance
        self.goal_yaw = 0.0
        self.reset()

    def reset(self):
        self.seq = WaypointSequence(self.waypoints, self.reach_radius)

    def anchor(self, t):
        return self.seq.current()

    def obs(self, b: ObservationBuilder, pos, q, t):
        self.seq.update(pos)
        return b.waypoint_task(pos, q, self.seq.current(), self.seq.next(), self.seq.remaining_frac(),
                               self.max_obs_distance)


class NavTask:
    """``NavEnv`` (range-ring variant): reach ``goal`` using the latest 360 deg range ring (set :attr:`ring`)."""

    def __init__(self, goal, range_max: float = 8.0, use_ring: bool = True):
        self.goal = np.asarray(goal, dtype=np.float64)
        self.range_max, self.use_ring = range_max, use_ring
        self.ring = None
        self.goal_yaw = 0.0

    def reset(self):
        pass

    def anchor(self, t):
        return self.goal

    def obs(self, b: ObservationBuilder, pos, q, t):
        return b.nav_task(pos, q, self.goal, self.ring if self.use_ring else None, self.range_max)


# ----------------------------------------------------------------------------- policy step


@dataclass
class StepResult:
    heartbeat: bool
    """publish ``OffboardControlMode`` this step"""
    setpoint: dict | None
    """PX4 set-point fields (:meth:`ActionDecoder.to_px4`) to publish, or None"""
    action: np.ndarray | None
    """normalized policy action (None when the policy did not run)"""
    reason: str
    """ok | no_odometry | stale_odometry | stale_sensor | geofence | not_offboard"""


class OffboardPolicy:
    def __init__(self, contract: dict, policy: Callable[[np.ndarray], np.ndarray], task="hover",
                 goal=(0.0, 0.0, 1.5), goal_yaw: float = 0.0, thrust_scale: float = 1.0,
                 max_odom_age_s: float = 0.05, geofence_radius_m: float = 3.0, thrust_curve=(0.0, 1.0),
                 max_sensor_age_s: float = 0.2):
        self.contract = contract
        self.policy = policy  # (1, obs_dim) float32 -> (1, act_dim)
        self.obs_builder = ObservationBuilder(contract)
        self.decoder = ActionDecoder(contract)
        self.hover_action = np.asarray(contract["action"]["hover_action"], dtype=np.float32)
        self.mode = contract["action"]["control_mode"]
        tp = contract.get("task_params", {})
        if task == "hover":
            task = HoverTask(goal, goal_yaw)
        elif task == "nav":
            task = NavTask(goal, range_max=tp.get("range_max", 8.0))
        elif isinstance(task, str):
            raise ValueError(f"task {task!r} needs a task object (TrackTask / WaypointTask)")
        self.task = task
        self.thrust_scale = float(thrust_scale)
        self.thrust_curve = ThrustCurve(thrust_curve)
        self.max_odom_age_s = float(max_odom_age_s)
        self.max_sensor_age_s = float(max_sensor_age_s)
        self.geofence_radius_m = float(geofence_radius_m)
        self.t_engage = None
        self.t_task = 0.0

    # hover / nav goal (ROS parameter callback) -------------------------------------------------
    @property
    def goal(self):
        return self.task.anchor(self.t_task)

    @goal.setter
    def goal(self, value):
        self.task.goal = np.asarray(value, dtype=np.float64)

    @property
    def goal_yaw(self):
        return self.task.goal_yaw

    @goal_yaw.setter
    def goal_yaw(self, value):
        self.task.goal_yaw = float(value)

    def heartbeat_flags(self) -> dict:
        """``OffboardControlMode`` fields for this policy's action interface."""
        return {"body_rate": self.mode == "ctbr", "velocity": self.mode == "velocity"}

    def step(self, odom, odom_age_s: float, offboard: bool, now: float = 0.0, sensor_age_s: float = 0.0) -> StepResult:
        """``odom`` = (pos_enu, q_enu_flu, vel_enu, gyro_flu) or None; ``offboard`` = PX4 nav_state is OFFBOARD;
        ``now`` [s] = monotonic time (task time starts at the hand-over); ``sensor_age_s`` = age of the
        exteroceptive input (range ring) for tasks that use it."""
        if odom is None:
            return StepResult(False, None, None, "no_odometry")
        if not odom_age_s <= self.max_odom_age_s:  # also catches NaN
            return StepResult(False, None, None, "stale_odometry")
        if isinstance(self.task, NavTask) and self.task.use_ring and \
                (self.task.ring is None or not sensor_age_s <= self.max_sensor_age_s):
            return StepResult(False, None, None, "stale_sensor")
        pos, q, vel, gyro = odom
        if not offboard:
            self.t_engage = None
            self.task.reset()
            self.obs_builder.update_last_action(self.hover_action)
            fence_ok = np.linalg.norm(pos - self.task.anchor(0.0)) <= self.geofence_radius_m
            return StepResult(bool(fence_ok), None, None, "not_offboard" if fence_ok else "geofence")
        if self.t_engage is None:
            self.t_engage = now
        self.t_task = now - self.t_engage
        task = self.task.obs(self.obs_builder, pos, q, self.t_task)  # may advance task state (waypoints)
        if not np.linalg.norm(pos - self.task.anchor(self.t_task)) <= self.geofence_radius_m:
            return StepResult(False, None, None, "geofence")
        obs = self.obs_builder.build(self.obs_builder.state_obs(pos, q, vel, gyro), task)[None]
        action = np.clip(np.asarray(self.policy(obs), dtype=np.float32).reshape(-1), -1.0, 1.0)
        self.obs_builder.update_last_action(action)
        sp = self.decoder.to_px4(action, q, thrust_scale=self.thrust_scale, thrust_curve=self.thrust_curve)
        return StepResult(True, sp, action, "ok")

    def log_row(self, t: float, odom_stamp: float, odom_rx: float, odom, action) -> list[float]:
        """One row of the step log, ordered as :data:`LOG_COLUMNS`."""
        pos, q, vel, gyro = odom
        return [t, odom_stamp, odom_rx, *pos, *vel, *q, *gyro, *self.goal, self.goal_yaw, self.thrust_scale,
                *action]
