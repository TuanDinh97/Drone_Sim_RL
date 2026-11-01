"""Framework-free core of the PX4 offboard bridge (``ros2/quadsim_px4_bridge``): one policy step.

The ROS 2 node only moves messages; every decision lives here so it is unit-tested without ROS
(tests/core/test_offboard.py):

* heartbeat gating: PX4 detects offboard loss only from the ``OffboardControlMode`` heartbeat
  (commander ``offboardCheck``: heartbeat older than ``COM_OF_LOSS_T``). The heartbeat is therefore sent only while
  the policy can command (fresh odometry inside the geofence); otherwise PX4's failsafe takes over instead of
  holding the last set-point.
* while PX4 (not the policy) flies, the policy's ``last_action`` input is held at the hover action, the value it
  saw at every episode start in training, instead of commands that were never applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .interface import ActionDecoder, ObservationBuilder, ThrustCurve

LOG_COLUMNS = ("t", "odom_stamp", "odom_rx", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy",
               "wz", "goal_x", "goal_y", "goal_z", "goal_yaw", "thrust_scale", "a0", "a1", "a2", "a3")
"""Bridge step log (:meth:`OffboardPolicy.log_row`): wall time of the step, odometry ``timestamp_sample`` (agent
clock) and receive time [s], ENU/FLU state used for the observation, goal, thrust scale, normalized action."""


@dataclass
class StepResult:
    heartbeat: bool
    """publish ``OffboardControlMode`` this step"""
    setpoint: dict | None
    """PX4 set-point fields (:meth:`ActionDecoder.to_px4`) to publish, or None"""
    action: np.ndarray | None
    """normalized policy action (None when the policy did not run)"""
    reason: str
    """ok | no_odometry | stale_odometry | geofence | not_offboard"""


class OffboardPolicy:
    def __init__(self, contract: dict, policy: Callable[[np.ndarray], np.ndarray], task: str = "hover",
                 goal=(0.0, 0.0, 1.5), goal_yaw: float = 0.0, thrust_scale: float = 1.0,
                 max_odom_age_s: float = 0.05, geofence_radius_m: float = 3.0, thrust_curve=(0.0, 1.0)):
        self.contract = contract
        self.policy = policy  # (1, obs_dim) float32 -> (1, act_dim)
        self.obs_builder = ObservationBuilder(contract)
        self.decoder = ActionDecoder(contract)
        self.hover_action = np.asarray(contract["action"]["hover_action"], dtype=np.float32)
        self.mode = contract["action"]["control_mode"]
        self.task = task
        self.goal = np.asarray(goal, dtype=np.float64)
        self.goal_yaw = float(goal_yaw)
        self.thrust_scale = float(thrust_scale)
        self.thrust_curve = ThrustCurve(thrust_curve)
        self.max_odom_age_s = float(max_odom_age_s)
        self.geofence_radius_m = float(geofence_radius_m)

    def heartbeat_flags(self) -> dict:
        """``OffboardControlMode`` fields for this policy's action interface."""
        return {"body_rate": self.mode == "ctbr", "velocity": self.mode == "velocity"}

    def step(self, odom, odom_age_s: float, offboard: bool) -> StepResult:
        """``odom`` = (pos_enu, q_enu_flu, vel_enu, gyro_flu) or None; ``offboard`` = PX4 nav_state is OFFBOARD."""
        if odom is None:
            return StepResult(False, None, None, "no_odometry")
        if not odom_age_s <= self.max_odom_age_s:  # also catches NaN
            return StepResult(False, None, None, "stale_odometry")
        pos, q, vel, gyro = odom
        if not np.linalg.norm(pos - self.goal) <= self.geofence_radius_m:
            return StepResult(False, None, None, "geofence")
        if not offboard:
            self.obs_builder.update_last_action(self.hover_action)
            return StepResult(True, None, None, "not_offboard")
        state = self.obs_builder.state_obs(pos, q, vel, gyro)
        if self.task == "hover":
            task = self.obs_builder.hover_task(pos, q, self.goal, self.goal_yaw)
        else:
            task = self.obs_builder.nav_task(pos, q, self.goal)
        obs = self.obs_builder.build(state, task)[None]
        action = np.clip(np.asarray(self.policy(obs), dtype=np.float32).reshape(-1), -1.0, 1.0)
        self.obs_builder.update_last_action(action)
        sp = self.decoder.to_px4(action, q, thrust_scale=self.thrust_scale, thrust_curve=self.thrust_curve)
        return StepResult(True, sp, action, "ok")

    def log_row(self, t: float, odom_stamp: float, odom_rx: float, odom, action) -> list[float]:
        """One row of the step log, ordered as :data:`LOG_COLUMNS`."""
        pos, q, vel, gyro = odom
        return [t, odom_stamp, odom_rx, *pos, *vel, *q, *gyro, *self.goal, self.goal_yaw, self.thrust_scale,
                *action]
