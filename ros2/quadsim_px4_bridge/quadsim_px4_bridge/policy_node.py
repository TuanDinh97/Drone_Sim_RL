"""ROS 2 node: quadsim ONNX policy as a PX4 offboard controller (PX4 v1.14+ uXRCE-DDS topics).

STATUS: interface code only — not yet flown against PX4 SITL or hardware (see docs/SIM2REAL.md).
All numerics (frames, observation layout, action decoding) live in ``quadsim.deploy.interface`` and are
unit-tested and validated in simulation (scripts/validation/m6_deploy_consistency.py).

Topics
  in : /fmu/out/vehicle_odometry (px4_msgs/VehicleOdometry, NED/FRD)
       /fmu/out/vehicle_status   (px4_msgs/VehicleStatus)
  out: /fmu/in/offboard_control_mode, /fmu/in/vehicle_rates_setpoint (CTBR) or /fmu/in/trajectory_setpoint (velocity)
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import onnxruntime as ort
import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleOdometry, VehicleRatesSetpoint, VehicleStatus
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from quadsim.deploy.interface import ActionDecoder, ObservationBuilder, px4_odometry_to_sim


class PolicyNode(Node):
    def __init__(self):
        super().__init__("quadsim_policy")
        p = self.declare_parameters("", [
            ("policy_dir", ""), ("task", "hover"), ("goal_enu", [0.0, 0.0, 1.5]), ("goal_yaw_enu", 0.0),
            ("thrust_scale", 1.0), ("max_odom_age_s", 0.05), ("geofence_radius_m", 3.0),
        ])
        get = lambda n: self.get_parameter(n).value  # noqa: E731
        d = get("policy_dir")
        self.contract = json.load(open(os.path.join(d, "contract.json")))
        self.sess = ort.InferenceSession(os.path.join(d, "policy.onnx"), providers=["CPUExecutionProvider"])
        self.obs_builder = ObservationBuilder(self.contract)
        self.decoder = ActionDecoder(self.contract)
        self.task = get("task")
        self.goal = np.asarray(get("goal_enu"), dtype=np.float64)
        self.goal_yaw = float(get("goal_yaw_enu"))
        self.thrust_scale = float(get("thrust_scale"))
        self.max_age = float(get("max_odom_age_s"))
        self.fence = float(get("geofence_radius_m"))
        self.mode = self.contract["action"]["control_mode"]
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.on_odom, qos)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.on_status, qos)
        self.pub_mode = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.pub_rates = self.create_publisher(VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", qos)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
        self.odom, self.odom_t, self.offboard = None, 0.0, False
        self.create_timer(1.0 / self.contract["policy_rate_hz"], self.step)
        self.get_logger().info(f"policy {d} mode={self.mode} rate={self.contract['policy_rate_hz']} Hz")

    def on_odom(self, m: VehicleOdometry):
        self.odom = px4_odometry_to_sim(np.array(m.position), np.array(m.q), np.array(m.velocity),
                                        np.array(m.angular_velocity))
        self.odom_t = time.monotonic()

    def on_status(self, m: VehicleStatus):
        self.offboard = m.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def _us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def step(self):
        mode = OffboardControlMode(timestamp=self._us(), body_rate=self.mode == "ctbr",
                                   velocity=self.mode == "velocity")
        self.pub_mode.publish(mode)  # heartbeat must stream before/while in offboard
        if self.odom is None or time.monotonic() - self.odom_t > self.max_age:
            return  # stale estimate: stop commanding, PX4 offboard-loss failsafe takes over
        pos, q, vel, gyro = self.odom
        if np.linalg.norm(pos - self.goal) > self.fence:
            self.get_logger().warn("geofence exceeded: not sending set-points", throttle_duration_sec=1.0)
            return
        state = self.obs_builder.state_obs(pos, q, vel, gyro)
        if self.task == "hover":
            task = self.obs_builder.hover_task(pos, q, self.goal, self.goal_yaw)
        else:
            task = self.obs_builder.nav_task(pos, q, self.goal)
        obs = self.obs_builder.build(state, task)[None]
        action = self.sess.run(None, {"obs": obs})[0][0]
        self.obs_builder.update_last_action(action)
        sp = self.decoder.to_px4(action, q, thrust_scale=self.thrust_scale)
        if not self.offboard:
            return
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
    rclpy.spin(PolicyNode())


if __name__ == "__main__":
    main()
