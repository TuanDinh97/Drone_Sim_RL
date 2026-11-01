"""PX4 SITL scenario for a quadsim policy running in ``quadsim_px4_bridge`` (ROS 2, system Python of ROS).

Sequence: wait for PX4 pre-flight checks -> arm + PX4 take-off -> PX4 Hold -> (calibrate ``thrust_scale`` from
PX4's hover-thrust estimate, docs/SIM2REAL.md) -> set the bridge goal to the current pose -> OFFBOARD (policy
flies) -> hold ``--hold`` s -> goal +1 m x (ENU) for ``--step`` s -> goal +1 m z for ``--step`` s -> PX4 Hold ->
land. The same schedule is flown in quadsim by scripts/sitl/quadsim_scenario.py.

Recorded (all stamps in the uXRCE-DDS agent's time base = this machine's clock, seconds):
odometry (EKF2), simulator ground truth (requires the SITL instrumentation of ros2/px4_sitl/setup_px4_sitl.sh),
the bridge's rate set-points, vehicle status and goal changes -> ``--out`` (.npz) + ``.json`` metadata.

Usage (after ``source /opt/ros/jazzy/setup.bash; source <ws>/install/setup.bash``):
    python3 scripts/sitl/sitl_scenario.py --out logs/sitl/run1.npz --px4_build <PX4>/build/px4_sitl_default
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time

import numpy as np
import rclpy
from px4_msgs.msg import (VehicleAngularVelocity, VehicleAttitude, VehicleCommand, VehicleCommandAck,
                          VehicleLocalPosition, VehicleOdometry, VehicleRatesSetpoint, VehicleStatus)
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from quadsim.deploy.interface import ThrustCurve, px4_odometry_to_sim, px4_topic, yaw_of

NAV_OFFBOARD, NAV_LOITER, NAV_TAKEOFF = 14, 4, 17
ARMED = 2


def topic(base, msg_type):
    return px4_topic(base, getattr(msg_type, "MESSAGE_VERSION", 0))


class Scenario(Node):
    def __init__(self, args):
        super().__init__("quadsim_sitl_scenario")
        self.args = args
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=50)
        self.lock = threading.Lock()
        self.rec = {k: [] for k in ("odom", "gt_pos", "gt_att", "gt_rate", "rates_sp", "status", "goal", "ack")}
        self.status = None
        self.odom = None
        sub = self.create_subscription
        sub(VehicleOdometry, topic("/fmu/out/vehicle_odometry", VehicleOdometry), self.on_odom, qos)
        sub(VehicleStatus, topic("/fmu/out/vehicle_status", VehicleStatus), self.on_status, qos)
        sub(VehicleCommandAck, topic("/fmu/out/vehicle_command_ack", VehicleCommandAck), self.on_ack, qos)
        sub(VehicleLocalPosition, "/fmu/out/vehicle_local_position_groundtruth", self.on_gt_pos, qos)
        sub(VehicleAttitude, "/fmu/out/vehicle_attitude_groundtruth", self.on_gt_att, qos)
        sub(VehicleAngularVelocity, "/fmu/out/vehicle_angular_velocity_groundtruth", self.on_gt_rate, qos)
        sub(VehicleRatesSetpoint, topic("/fmu/in/vehicle_rates_setpoint", VehicleRatesSetpoint), self.on_rates, qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, topic("/fmu/in/vehicle_command", VehicleCommand), qos)
        self.param_cli = self.create_client(SetParameters, f"/{args.bridge_node}/set_parameters")

    # ------------------------------------------------------------------ recording
    def _put(self, key, row):
        with self.lock:
            self.rec[key].append(row)

    def on_odom(self, m):
        self.odom = m
        self._put("odom", [time.time(), m.timestamp * 1e-6, m.timestamp_sample * 1e-6, *m.position, *m.q,
                           *m.velocity, *m.angular_velocity, m.pose_frame, m.velocity_frame, m.reset_counter])

    def on_status(self, m):
        self.status = m
        self._put("status", [time.time(), m.timestamp * 1e-6, m.nav_state, m.arming_state,
                             float(m.pre_flight_checks_pass), float(m.failsafe)])

    def on_ack(self, m):
        self._put("ack", [time.time(), m.command, m.result])

    def on_gt_pos(self, m):
        self._put("gt_pos", [time.time(), m.timestamp_sample * 1e-6, m.x, m.y, m.z, m.vx, m.vy, m.vz])

    def on_gt_att(self, m):
        self._put("gt_att", [time.time(), m.timestamp_sample * 1e-6, *m.q])

    def on_gt_rate(self, m):
        self._put("gt_rate", [time.time(), m.timestamp_sample * 1e-6, *m.xyz])

    def on_rates(self, m):
        self._put("rates_sp", [time.time(), m.timestamp * 1e-6, m.roll, m.pitch, m.yaw, *m.thrust_body])

    # ------------------------------------------------------------------ commands
    def command(self, cmd, p1=0.0, p2=0.0, p3=0.0, p4=float("nan"), p5=float("nan"), p6=float("nan"),
                p7=float("nan")):
        m = VehicleCommand(timestamp=int(time.time() * 1e6), command=cmd, param1=float(p1), param2=float(p2),
                           param3=float(p3), param4=float(p4), param5=float(p5), param6=float(p6), param7=float(p7),
                           target_system=1, target_component=1, source_system=1, source_component=1,
                           from_external=True)
        self.pub_cmd.publish(m)

    def set_mode(self, main, sub=0):
        self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, main, sub)

    def set_bridge(self, **kv):
        req = SetParameters.Request()
        for k, v in kv.items():
            if isinstance(v, (list, tuple)):
                val = ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                                     double_array_value=[float(x) for x in v])
            else:
                val = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(v))
            req.parameters.append(Parameter(name=k, value=val))
        if not self.param_cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"bridge node /{self.args.bridge_node} not available")
        fut = self.param_cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 5.0:
            time.sleep(0.01)
        res = fut.result()
        if res is None or not all(r.successful for r in res.results):
            raise RuntimeError(f"setting bridge parameters {kv} failed: {res}")
        if "goal_enu" in kv:
            self._put("goal", [time.time(), *kv["goal_enu"]])

    # ------------------------------------------------------------------ helpers
    def wait(self, cond, timeout, what):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cond():
                return time.time() - t0
            time.sleep(0.02)
        raise TimeoutError(f"timeout ({timeout} s) waiting for {what}")

    def nav(self):
        return None if self.status is None else self.status.nav_state

    def enu_pose(self):
        m = self.odom
        pos, q, _, _ = px4_odometry_to_sim(np.array(m.position), np.array(m.q), np.array(m.velocity),
                                           np.array(m.angular_velocity))
        return pos, float(yaw_of(q))

    def px4_shell(self, *cmd):
        b = self.args.px4_build
        out = subprocess.run([os.path.join(b, "bin", cmd[0]), *cmd[1:]], cwd=os.path.join(b, "rootfs"),
                             capture_output=True, text=True, timeout=10)
        return out.stdout + out.stderr


def run(node: Scenario, args) -> dict:
    meta = {"args": vars(args), "events": {}}
    ev = meta["events"]
    log = node.get_logger().info
    node.wait(lambda: node.odom is not None and node.status is not None, 120, "PX4 odometry + status")
    node.wait(lambda: node.status.pre_flight_checks_pass, 180, "pre-flight checks (EKF ready)")
    log("pre-flight checks pass -> arm + take-off")
    for _ in range(20):
        node.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        time.sleep(0.2)
        node.command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF)
        time.sleep(0.3)
        if node.status.arming_state == ARMED and node.nav() in (NAV_TAKEOFF, NAV_LOITER):
            break
    node.wait(lambda: node.status.arming_state == ARMED, 5, "arming")
    ev["armed"] = time.time()
    node.wait(lambda: node.nav() == NAV_LOITER, 60, "take-off complete (Hold)")
    time.sleep(args.settle)
    ev["hold_reached"] = time.time()
    if node.args.px4_build:
        meta["px4_hover_thrust_estimate"] = node.px4_shell("px4-listener", "hover_thrust_estimate")
        meta["px4_param_MPC_THR_HOVER"] = node.px4_shell("px4-param", "show", "MPC_THR_HOVER")
    curve = ThrustCurve(args.thrust_curve)
    if args.thrust_scale == "auto":
        m = re.search(r"hover_thrust:\s*([0-9.]+)", meta.get("px4_hover_thrust_estimate", ""))
        if m is None:
            raise RuntimeError("could not read hover_thrust_estimate (pass --thrust_scale <value>)")
        meta["u_hover"] = float(m.group(1))
        ts = curve.calibrate(meta["u_hover"], args.sim_hover_fraction)
    else:
        ts = float(args.thrust_scale)
    meta["thrust_scale"], meta["thrust_curve"] = ts, list(args.thrust_curve)
    pos, yaw = node.enu_pose()
    g0 = [round(float(x), 2) for x in pos]
    meta["goal0_enu"], meta["goal_yaw_enu"] = g0, yaw
    node.set_bridge(goal_enu=g0, goal_yaw_enu=yaw, thrust_scale=ts, thrust_curve=list(args.thrust_curve))
    log(f"thrust_scale={ts:.4f} goal={g0} yaw={yaw:.3f} -> OFFBOARD")
    time.sleep(1.0)  # heartbeat streaming with the new goal
    for _ in range(25):
        node.set_mode(6)
        time.sleep(0.2)
        if node.nav() == NAV_OFFBOARD:
            break
    node.wait(lambda: node.nav() == NAV_OFFBOARD, 3, "OFFBOARD")
    t0 = ev["offboard"] = time.time()
    schedule = [(args.hold, [0.0, 0.0, 0.0])]
    if args.step > 0:
        schedule += [(args.step, [1.0, 0.0, 0.0]), (args.step, [1.0, 0.0, 1.0])]
    t = 0.0
    for i, (dur, off) in enumerate(schedule):
        if i > 0:
            node.set_bridge(goal_enu=[g + o for g, o in zip(g0, off)])
            ev[f"goal_{i}"] = time.time()
        t += dur
        while time.time() - t0 < t:
            if node.nav() != NAV_OFFBOARD:
                ev["left_offboard"] = time.time()
                meta["left_offboard_nav_state"] = int(node.nav())
                log(f"vehicle left OFFBOARD (nav_state {node.nav()}) at {time.time() - t0:.2f} s")
                break
            time.sleep(0.02)
        if "left_offboard" in ev:
            break
    if args.fence_test > 0 and "left_offboard" not in ev:
        # goal beyond the bridge's geofence: the bridge must stop its heartbeat -> PX4 offboard-loss failsafe
        node.set_bridge(goal_enu=[g0[0] + args.fence_test, g0[1], g0[2]])
        ev["fence_goal"] = time.time()
        try:
            dt = node.wait(lambda: node.nav() != NAV_OFFBOARD, 10.0, "PX4 leaving OFFBOARD after the fence violation")
            ev["left_offboard"] = time.time()
            meta["left_offboard_nav_state"] = int(node.nav())
            meta["fence_failsafe_latency_s"] = dt
            log(f"fence test: PX4 left OFFBOARD after {dt:.2f} s (nav_state {node.nav()})")
        except TimeoutError as e:
            meta["fence_test_error"] = str(e)
        time.sleep(3.0)  # observe the failsafe mode
    ev["handback"] = time.time()
    log("schedule done -> PX4 Hold -> land")
    for _ in range(10):
        node.set_mode(4, 3)
        time.sleep(0.2)
        if node.nav() == NAV_LOITER:
            break
    time.sleep(2.0)
    node.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
    try:
        node.wait(lambda: node.status.arming_state != ARMED, 60, "landing + disarm")
        ev["disarmed"] = time.time()
    except TimeoutError as e:
        meta["landing_error"] = str(e)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--hold", type=float, default=30.0)
    p.add_argument("--step", type=float, default=15.0, help="duration of each 1 m step segment (0: hover only)")
    p.add_argument("--settle", type=float, default=8.0, help="s in PX4 Hold before the hand-over")
    p.add_argument("--thrust_scale", default="auto", help="'auto' (PX4 hover-thrust estimate) or a value")
    p.add_argument("--sim_hover_fraction", type=float, default=0.592, help="m g / T_max of the training preset")
    p.add_argument("--thrust_curve", type=float, nargs="+", default=[0.0, 1.0],
                   help="bridge thrust curve coefficients (default linear; gz x500: 0.0225 0.255 0.7225)")
    p.add_argument("--px4_build", default="", help="PX4 build dir (for px4-listener / px4-param)")
    p.add_argument("--bridge_node", default="quadsim_policy")
    p.add_argument("--fence_test", type=float, default=0.0,
                   help="after the schedule move the goal this far [m] (beyond the bridge geofence), time the failsafe")
    args = p.parse_args()
    rclpy.init()
    node = Scenario(args)
    ex = SingleThreadedExecutor()  # rclpy's MultiThreadedExecutor dropped ~35 % of the 300 msg/s here
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    meta, err = {}, None
    try:
        meta = run(node, args)
    except Exception as e:  # keep the recording for diagnosis
        err = repr(e)
        node.get_logger().error(err)
    meta["error"] = err
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with node.lock:
        np.savez_compressed(args.out, **{k: np.asarray(v, dtype=np.float64) for k, v in node.rec.items()})
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[sitl] wrote {args.out} error={err}", flush=True)
    ex.shutdown()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
