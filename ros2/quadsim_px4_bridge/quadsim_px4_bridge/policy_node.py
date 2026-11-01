"""ROS 2 node: quadsim ONNX policy as a PX4 offboard controller (PX4 v1.14+ uXRCE-DDS topics).

STATUS: flown against PX4 v1.16.2 SITL (gz x500) with the hover policy, see docs/PROGRESS.md
("PX4 SITL cross-check"); not flown on hardware. All numerics (frames, observation layout, action decoding)
live in ``quadsim.deploy.interface`` and the per-step decisions in ``quadsim.deploy.offboard``; both are
unit-tested (tests/core) and validated in simulation (scripts/validation/m6_deploy_consistency.py).

Topics (PX4 >= 1.16 appends ``_v<N>`` to versioned messages, e.g. ``/fmu/out/vehicle_status_v1``)
  in : /fmu/out/vehicle_odometry (px4_msgs/VehicleOdometry, NED/FRD)
       /fmu/out/vehicle_status   (px4_msgs/VehicleStatus)
  out: /fmu/in/offboard_control_mode, /fmu/in/vehicle_rates_setpoint (CTBR) or /fmu/in/trajectory_setpoint (velocity)

Safety: the OffboardControlMode heartbeat is published only while the node can command (fresh, finite
odometry inside the geofence). PX4 detects offboard loss from that heartbeat alone (COM_OF_LOSS_T), so a
stale estimate or a fence violation hands control back to PX4's failsafe.

Thrust map: ``thrust_curve`` = ascending polynomial coefficients of the vehicle's normalized thrust curve
(default ``[0, 1]`` = linear; gz x500: ``[0.0225, 0.255, 0.7225]``), see ``quadsim.deploy.interface.ThrustCurve``.

Tasks (``task``, must match the policy): ``hover`` (``goal_enu``, ``goal_yaw_enu``); ``track`` (``track_shape`` circle |
figure8, ``track_radius``, ``track_freq_hz``, ``track_z_amp``, reference starting at ``track_start_enu`` at the
hand-over); ``waypoint`` (``waypoints_enu`` = flattened x, y, z list); ``nav`` (``goal_enu`` + a
``sensor_msgs/LaserScan`` on ``scan_topic``, resampled to the training range ring; velocity interface).

Runtime parameters (all except ``policy_dir``, ``task``, limits and ``log_path``) can be changed while flying, e.g.
``ros2 param set /quadsim_policy goal_enu "[1.0, 0.0, 2.5]"``; task definitions are rebuilt on change.
``log_path`` (optional) writes one CSV row per policy step with the policy's inputs and outputs (``LOG_COLUMNS``).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import onnxruntime as ort
import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleOdometry, VehicleRatesSetpoint, VehicleStatus
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import LaserScan
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from quadsim.deploy.interface import (SinusoidReference, ThrustCurve, px4_topic, px4_vehicle_odometry_to_sim,
                                     ring_from_scan)
from quadsim.deploy.offboard import LOG_COLUMNS, HoverTask, NavTask, OffboardPolicy, TrackTask, WaypointTask

TASK_PARAMS = ("goal_enu", "goal_yaw_enu", "track_shape", "track_radius", "track_freq_hz", "track_z_amp",
               "track_start_enu", "waypoints_enu", "waypoint_reach_radius")


def topic(base: str, msg_type) -> str:
    return px4_topic(base, getattr(msg_type, "MESSAGE_VERSION", 0))


class PolicyNode(Node):
    def __init__(self):
        super().__init__("quadsim_policy")
        self.declare_parameters("", [
            ("policy_dir", ""), ("task", "hover"), ("goal_enu", [0.0, 0.0, 1.5]), ("goal_yaw_enu", 0.0),
            ("thrust_scale", 1.0), ("thrust_curve", [0.0, 1.0]), ("max_odom_age_s", 0.05), ("geofence_radius_m", 3.0),
            ("log_path", ""), ("track_shape", "circle"), ("track_radius", 1.0), ("track_freq_hz", 0.2),
            ("track_z_amp", 0.0), ("track_start_enu", [0.0, 0.0, 2.0]), ("waypoints_enu", [0.0, 0.0, 2.0]),
            ("waypoint_reach_radius", -1.0), ("scan_topic", "/quadsim/scan"), ("max_sensor_age_s", 0.2),
        ])
        get = lambda n: self.get_parameter(n).value  # noqa: E731
        self.params = {n: get(n) for n in TASK_PARAMS}
        d = get("policy_dir")
        contract = json.load(open(os.path.join(d, "contract.json")))
        self.task_params = contract.get("task_params", {})  # task constants of training (horizon, radii, ...)
        sess = ort.InferenceSession(os.path.join(d, "policy.onnx"), providers=["CPUExecutionProvider"])
        self.task_name = get("task")
        self.core = OffboardPolicy(contract, lambda obs: sess.run(None, {"obs": obs})[0], task=self._build_task(),
                                   thrust_scale=get("thrust_scale"), max_odom_age_s=get("max_odom_age_s"),
                                   geofence_radius_m=get("geofence_radius_m"), thrust_curve=get("thrust_curve"),
                                   max_sensor_age_s=get("max_sensor_age_s"))
        self.scan_t = -1e9
        if self.task_name == "nav":
            self.create_subscription(LaserScan, get("scan_topic"), self.on_scan, 1)
        self.log = None
        if get("log_path"):
            self.log = open(get("log_path"), "w", buffering=1)  # line-buffered
            self.log.write(",".join(LOG_COLUMNS) + "\n")
        self.add_on_set_parameters_callback(self.on_params)
        # PX4 publishes /fmu/out/* best-effort + transient-local; a volatile best-effort reader matches both
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(VehicleOdometry, topic("/fmu/out/vehicle_odometry", VehicleOdometry), self.on_odom,
                                 qos)
        self.create_subscription(VehicleStatus, topic("/fmu/out/vehicle_status", VehicleStatus), self.on_status, qos)
        self.pub_mode = self.create_publisher(
            OffboardControlMode, topic("/fmu/in/offboard_control_mode", OffboardControlMode), qos)
        self.pub_rates = self.create_publisher(
            VehicleRatesSetpoint, topic("/fmu/in/vehicle_rates_setpoint", VehicleRatesSetpoint), qos)
        self.pub_traj = self.create_publisher(
            TrajectorySetpoint, topic("/fmu/in/trajectory_setpoint", TrajectorySetpoint), qos)
        self.odom, self.odom_t, self.offboard = None, 0.0, False
        self.odom_stamp, self.odom_rx = 0.0, 0.0
        self.create_timer(1.0 / contract["policy_rate_hz"], self.step)
        self.get_logger().info(f"policy {d} task={self.task_name} mode={self.core.mode} "
                               f"rate={contract['policy_rate_hz']} Hz thrust_scale={self.core.thrust_scale}")

    def _build_task(self):
        p = self.params
        if self.task_name == "hover":
            return HoverTask(p["goal_enu"], p["goal_yaw_enu"])
        if self.task_name == "track":
            ref = SinusoidReference.shape(p["track_shape"], p["track_radius"], p["track_freq_hz"], p["track_z_amp"],
                                          p["track_start_enu"])
            return TrackTask(ref, self.task_params.get("horizon_s", (0.0, 0.1, 0.2, 0.4, 0.6, 1.0)))
        if self.task_name == "waypoint":
            tp = self.task_params
            reach = p["waypoint_reach_radius"] if p["waypoint_reach_radius"] > 0 else tp.get("reach_radius", 0.3)
            return WaypointTask(np.asarray(p["waypoints_enu"], dtype=np.float64).reshape(-1, 3), reach,
                                tp.get("max_obs_distance", 5.0))
        if self.task_name == "nav":
            return NavTask(p["goal_enu"], range_max=self.task_params.get("range_max", 8.0))
        raise ValueError(f"unknown task {self.task_name!r}")

    def on_params(self, params) -> SetParametersResult:
        c = self.core
        for p in params:
            if p.name in TASK_PARAMS:
                if p.name in ("goal_enu", "track_start_enu") and len(p.value) != 3:
                    return SetParametersResult(successful=False, reason=f"{p.name} needs 3 values")
                if p.name == "waypoints_enu" and len(p.value) % 3:
                    return SetParametersResult(successful=False, reason="waypoints_enu needs 3 N values")
                self.params[p.name] = p.value
                if p.name == "goal_enu":
                    c.goal = np.asarray(p.value, dtype=np.float64)  # keep the task state (in-flight goal change)
                elif p.name == "goal_yaw_enu":
                    c.goal_yaw = float(p.value)
                else:
                    try:
                        c.task = self._build_task()
                    except ValueError as e:
                        return SetParametersResult(successful=False, reason=str(e))
            elif p.name == "thrust_scale":
                c.thrust_scale = float(p.value)
            elif p.name == "thrust_curve":
                try:
                    c.thrust_curve = ThrustCurve(p.value)
                except ValueError as e:
                    return SetParametersResult(successful=False, reason=str(e))
            elif p.name in ("policy_dir", "task", "max_odom_age_s", "geofence_radius_m", "log_path", "scan_topic",
                            "max_sensor_age_s"):
                return SetParametersResult(successful=False, reason=f"{p.name} is fixed at start-up")
        self.get_logger().info(f"goal={c.goal.tolist()} yaw={c.goal_yaw:.3f} thrust_scale={c.thrust_scale} "
                               f"thrust_curve={c.thrust_curve.p.coef.tolist()}")
        return SetParametersResult(successful=True)

    def on_odom(self, m: VehicleOdometry):
        odom = px4_vehicle_odometry_to_sim(m.position, m.q, m.velocity, m.angular_velocity, m.pose_frame,
                                           m.velocity_frame)
        if odom is None:
            self.get_logger().warn(f"unusable odometry (pose_frame={m.pose_frame}, velocity_frame="
                                   f"{m.velocity_frame} or NaN fields)", throttle_duration_sec=1.0)
            return
        self.odom = odom
        self.odom_t = time.monotonic()
        self.odom_stamp, self.odom_rx = m.timestamp_sample * 1e-6, time.time()

    def on_scan(self, m: LaserScan):
        tp = self.task_params
        self.core.task.ring = ring_from_scan(m.ranges, m.angle_min, m.angle_increment, tp.get("range_rays", 32),
                                             tp.get("range_max", 8.0))
        self.scan_t = time.monotonic()

    def on_status(self, m: VehicleStatus):
        self.offboard = m.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def _us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def step(self):
        now = time.monotonic()
        r = self.core.step(self.odom, now - self.odom_t, self.offboard, now=now, sensor_age_s=now - self.scan_t)
        if r.reason in ("stale_odometry", "stale_sensor", "geofence"):
            self.get_logger().warn(f"{r.reason}: heartbeat and set-points stopped (PX4 failsafe takes over)",
                                   throttle_duration_sec=1.0)
        if r.heartbeat:
            self.pub_mode.publish(OffboardControlMode(timestamp=self._us(), **self.core.heartbeat_flags()))
        sp = r.setpoint
        if sp is None:
            return
        if self.log is not None:
            row = self.core.log_row(time.time(), self.odom_stamp, self.odom_rx, self.odom, r.action)
            self.log.write(",".join(f"{x:.6f}" for x in row) + "\n")
        if sp["msg"] == "VehicleRatesSetpoint":
            msg = VehicleRatesSetpoint(timestamp=self._us(), roll=float(sp["roll"]), pitch=float(sp["pitch"]),
                                       yaw=float(sp["yaw"]))
            msg.thrust_body = [float(x) for x in sp["thrust_body"]]
            self.pub_rates.publish(msg)
        else:
            msg = TrajectorySetpoint(timestamp=self._us())
            msg.position = [float("nan")] * 3
            msg.velocity = [float(x) for x in sp["velocity"]]
            msg.yaw = float("nan")
            msg.yawspeed = float(sp["yawspeed"])
            self.pub_traj.publish(msg)


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.log is not None:
            node.log.close()


if __name__ == "__main__":
    main()
