"""Virtual 2-D LiDAR for PX4 SITL navigation runs (ROS 2, system Python of ROS).

Publishes ``sensor_msgs/LaserScan`` on ``/quadsim/scan`` at ``rate_hz``: the 32-ray horizontal ring of
``NavEnv._update_range_ring`` (yaw-aligned, CCW from the heading, ray k at azimuth 2 pi k / 32, max range 8 m),
ray-cast analytically against the vertical cylinders of a navigation layout from the simulator ground-truth pose
(like a physical body-mounted sensor), with the training noise (2 cm Gaussian, 1 % dropouts -> max range).
The obstacles are virtual: they are not in the Gazebo world (no contact physics); collisions are evaluated from the
ground truth with the training criterion (scripts/sitl/compare_tasks.py).

Parameters: ``layout_file`` (JSON from quadsim_task_scenario.py --make_layouts), ``layout_index``, ``transform``
= [dx, dy, dyaw]: rigid map of the layout (rotation by dyaw about its start, then the start moved by dx, dy; set
by the scenario at the hand-over so that the layout start / heading coincide with the vehicle's), ``rate_hz``,
``noise``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import rclpy
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from quadsim.deploy.virtual_sensors import ray_cylinders, ring_dirs, transform_layout


class VirtualScan(Node):
    def __init__(self):
        super().__init__("quadsim_virtual_scan")
        self.declare_parameters("", [("layout_file", ""), ("layout_index", 0), ("transform", [0.0, 0.0, 0.0]),
                                     ("rate_hz", 50.0), ("noise", True), ("num_rays", 32), ("range_max", 8.0)])
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        layout = json.load(open(g("layout_file")))["layouts"][g("layout_index")]
        self.cyl = np.asarray(layout["cylinders"], dtype=np.float64)
        self.start = np.asarray(layout["start"][:2], dtype=np.float64)
        self.n, self.rmax, self.noise = g("num_rays"), g("range_max"), g("noise")
        self.rng = np.random.default_rng(0)
        self.pos = self.yaw = None
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_groundtruth", self.on_pos, qos)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude_groundtruth", self.on_att, qos)
        self.pub = self.create_publisher(LaserScan, "/quadsim/scan", 1)
        self.create_timer(1.0 / g("rate_hz"), self.tick)

    def on_pos(self, m):
        self.pos = np.array([m.y, m.x, -m.z])  # NED -> ENU

    def on_att(self, m):
        w, x, y, z = m.q
        self.yaw = math.pi / 2 - math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))  # NED yaw -> ENU yaw

    def tick(self):
        if self.pos is None or self.yaw is None:
            return
        cyl = transform_layout(self.cyl, self.start, self.get_parameter("transform").value)
        r = ray_cylinders(self.pos[:2], self.pos[2], ring_dirs(self.n, self.yaw), cyl, self.rmax)
        if self.noise:
            r = r + 0.02 * self.rng.standard_normal(self.n)
            r = np.where(self.rng.random(self.n) < 0.01, self.rmax, r)
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link_yaw_aligned"
        msg.angle_min, msg.angle_increment = 0.0, 2.0 * math.pi / self.n
        msg.angle_max = msg.angle_min + msg.angle_increment * (self.n - 1)
        msg.range_min, msg.range_max = 0.0, float(self.rmax)
        msg.ranges = [float(x) for x in np.clip(r, 0.0, self.rmax)]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = VirtualScan()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
