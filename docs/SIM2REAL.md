# Sim-to-Real guide

This document describes what the platform guarantees for transfer, what has been verified in simulation,
and the steps that remain before a real flight. The hover policy has been flown as a PX4 offboard controller
against **PX4 v1.16.2 SITL** (Gazebo `x500`) through the ROS 2 bridge and compared with quadsim
(docs/PROGRESS.md, "PX4 SITL cross-check"). **No policy from this repository has been flown on hardware.**

## 1. Design principles implemented

| Principle | Implementation | Verified |
|---|---|---|
| Deployable observations only | `policy` obs = estimated velocity (body), attitude (R[:, :2]), gyro, last action, task info. Everything else (true state, rotor speeds, wind, DR parameters) is `critic`-only (asymmetric actor-critic). | by construction + deploy-in-the-loop test |
| Realistic actuation | first-order rotor dynamics with asymmetric τ, ω saturation, mixer desaturation, onboard PX4-like rate PID at 200 Hz using the *noisy* gyro | M2 (PhysX vs reference), unit tests |
| Latency | action delay 0–2 physics steps (0–10 ms), odometry delay 0–2 steps, optional IMU delay, all randomized per episode | unit test (exact delay) |
| Sensor noise | IMU (white + bias drift + turn-on bias), odometry noise / VIO drift, depth noise (σ∝z², dropouts, quantization) | unit tests (statistics) |
| Domain randomization | mass, inertia, CoM, k_f per rotor, k_m, τ_motor, ω_max (battery sag), drag, payload, sensor-noise scale | DR ablation (docs/PROGRESS.md) |
| Interface contract | `contract.json` exported with every policy: obs layout, action semantics & scaling, rates, frames, training assumptions, vehicle parameters | M6 |
| Numerically identical deployment | NumPy observation builder + ONNX policy reproduce simulator obs/actions to 1e-6 in closed loop | M6 (`scripts/validation/m6_deploy_consistency.py`) |

## 2. Frames and conventions

| | Simulation (quadsim) | PX4 |
|---|---|---|
| World | z-up (ENU-like; x = forward of the arena) | NED |
| Body | FLU | FRD |
| Quaternion | wxyz, `q_WB` | wxyz, `q` of VehicleOdometry (FRD→NED) |
| Rotor order | 0 FR (CCW), 1 RL (CCW), 2 FL (CW), 3 RR (CW) | Quad-X: identical numbering 1..4 |

Conversions are in `quadsim.deploy.interface` (`px4_odometry_to_sim`, `q_enu_flu_to_ned_frd`, `enu_to_ned`,
`flu_to_frd`) and are unit-tested (`tests/core/test_deploy_interface.py`), including the check that the
same physical body vector is mapped consistently through both quaternion conventions and that PX4 yaw 0
(nose north) equals simulation yaw +90°.

## 3. Action interfaces

* **CTBR** (`Quadsim-Hover/Track/Waypoint`): collective thrust + body rates → PX4 `VehicleRatesSetpoint`
  (`roll, pitch, yaw` in FRD, `thrust_body = [0, 0, −T_n]`). PX4's own rate controller replaces the
  simulated one. The simulated rate loop uses PX4-like gains; the real gains should be similar or the
  rate-loop bandwidth should be identified and matched (see §5).
* **Velocity** (`Quadsim-Nav-*`): velocity in the yaw-aligned frame + yaw rate → PX4 `TrajectorySetpoint`
  (NED velocity, yawspeed). PX4's position/velocity controller replaces the simulated cascade — and that matters:
  in SITL PX4's default velocity loop (`MPC_XY_VEL_P_ACC`=1.8) is ~2× slower than the simulated one (P = 3.0) and the
  range-navigation policy reached the goal on 2/6 layouts (quadsim 87 %); with PX4 configured like the training cascade
  (`MPC_XY_VEL_P_ACC=3.0 MPC_XY_VEL_D_ACC=0 MPC_Z_VEL_I_ACC=1.0 MPC_TILTMAX_AIR=34.4`) 4/6. **Configure (or identify and
  train with) the autopilot's velocity loop** before deploying a velocity policy.

### Thrust calibration (CTBR)
PX4's normalized thrust maps nonlinearly to Newtons. Set `thrust_scale` in the ROS 2 node so that the
simulated hover command maps to PX4's hover thrust:
`thrust_scale = u_hover / (m g / T_max_sim)` (for the X500 preset `m g / T_max = 0.592`).
Use the **hover-thrust estimate** of a PX4-controlled hover (`listener hover_thrust_estimate`, done automatically by
`scripts/sitl/sitl_scenario.py`), not the parameter `MPC_THR_HOVER`: in SITL the x500 default is 0.60 while the
estimate is 0.73 (scale 1.23 instead of 1.01). The linear map is exact only at hover: for gz `x500`
(`T = 4 k_f (150 + 850 u)^2`) the local thrust gain around hover is 1.6x the linear map's, and in SITL 3 of 5 runs
with the linear map developed a ~4 Hz heave limit cycle. **Preferred: the thrust-curve map** — bridge parameter
`thrust_curve` = ascending polynomial coefficients of `c(u) = T(u)/T_ref` (gz x500: `[0.0225, 0.255, 0.7225]`), with
`thrust_scale = c(u_hover) / (m g / T_max_sim)` (`ThrustCurve.calibrate`; 1.0016 in SITL); 0 of 4 SITL runs
oscillated. For hardware, measure the curve on a thrust stand (and consider adding it to the contract).

## 4. Deployment steps (ROS 2 + PX4 v1.14+, uXRCE-DDS)

1. Export: `python scripts/rl/export.py --headless --task Quadsim-Hover-v0 --checkpoint <run>`.
2. Copy `<run>/exported/{policy.onnx, contract.json}` to the companion computer; install `onnxruntime`,
   `numpy` and the `quadsim` package (only `quadsim.deploy` is imported at runtime — no Isaac Sim needed).
3. Build `ros2/quadsim_px4_bridge` in a ROS 2 workspace that contains `px4_msgs` generated from the **same PX4
   tree** that is flashed (PX4 ≥ 1.16 versions some topics, e.g. `VehicleStatus` → `/fmu/out/vehicle_status_v1`;
   the bridge derives topic names from `px4_msgs`' `MESSAGE_VERSION`).
4. **PX4 SITL first** — verified procedure (Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic, Micro-XRCE-DDS Agent v2,
   PX4 v1.16.2; the bridge uses only rclpy APIs that also exist in Humble):
   * `bash ros2/px4_sitl/setup_px4_sitl.sh` — clones/builds PX4 SITL and the ROS 2 workspace (px4_msgs + bridge)
     under `third_party/`; adds SITL-only ground-truth topics to PX4's `dds_topics.yaml` (evaluation only).
   * `POLICY_DIR=<run>/exported bash scripts/sitl/run_sitl.sh logs/sitl/run.npz` — agent, PX4 + gz `x500`
     (headless, fresh parameters), bridge, and `scripts/sitl/sitl_scenario.py`: PX4 take-off → Hold → thrust
     calibration → OFFBOARD (policy) → hold / 1 m steps → Hold → land; records everything. Evaluate with
     `scripts/sitl/compare.py`.
   * `run_sitl.sh ... --thrust_curve 0.0225 0.255 0.7225` uses the thrust-curve map (recommended, §3).
   * Other tasks: `TASK=track|waypoint|nav POLICY_DIR=... bash scripts/sitl/run_sitl.sh out.npz --task <task> ...`
     (circle / figure-8 reference from the hand-over pose, waypoint course, navigation layout with the virtual 2-D
     LiDAR `scripts/sitl/virtual_scan.py`: `SCAN_LAYOUTS=layouts.json SCAN_INDEX=i`). quadsim counterparts:
     `scripts/sitl/quadsim_task_scenario.py`; comparison: `scripts/sitl/compare_tasks.py`.
   * SITL estimator: `run_sitl.sh` sets `EKF2_GPS_DELAY=0` because gz's GNSS has no latency (with the 110 ms default,
     right for real receivers, EKF2 leads the truth by 0.1 s in motion). Do **not** copy this to hardware.
   * Manual use: `ros2 launch quadsim_px4_bridge policy.launch.py policy_dir:=<dir> thrust_scale:=<s>`, with
     `PYTHONPATH` containing `onnxruntime` and `source/quadsim`; arm and switch to OFFBOARD from QGC or with
     `VehicleCommand`. The goal can be changed in flight: `ros2 param set /quadsim_policy goal_enu "[x, y, z]"`.
   * Pitfalls met: the x500 airframe sets `NAV_DLL_ACT=2`, so arming fails without a GCS (headless runs set it to 0);
     ROS 2's vendored `gz` on `PATH` cannot start PX4's Gazebo Harmonic world (run PX4 with `/opt/ros` removed from
     `PATH`); a surviving `gz sim` server is silently re-used by the next PX4 instance.
5. Hardware: start tethered / in a net, low geofence radius, RC kill switch, offboard-loss failsafe set
   to *Land* or *Position hold*.

The node sends the `OffboardControlMode` heartbeat **only** while it can command (odometry younger than
`max_odom_age_s`, finite, NED frame, inside `geofence_radius_m`). PX4 decides offboard loss from that heartbeat
alone (`COM_OF_LOSS_T`), so a stale estimate or a fence violation hands control to PX4's failsafe (verified in SITL,
docs/PROGRESS.md). Before the hand-over the policy's `last_action` input is held at the hover action.

## 5. Open gaps before real flight (prioritized)

1. **PX4 SITL** (x500) — done for the hover policy (docs/PROGRESS.md). Open points found there:
   (a) the policy observes EKF2 odometry whose slowly varying position error (cm level with GNSS + baro) is not
   in the training noise model (white noise + small random walk); (b) with a linear thrust map the policy's lightly
   damped ~4 Hz heave mode became a limit cycle in 3/5 runs → use the measured thrust curve, consider randomizing the
   curve shape in training; (c) the PX4 rate loop differs from the simulated
   one (pitch: same delay, smaller τ) and is roll/pitch-asymmetric because the x500 airframe's `CA_ROTOR*` geometry
   differs from the gz model;
   identify the real vehicle's rate-loop response (FOPDT fit, `quadsim.deploy.closed_loop_metrics.fit_fopdt`) and
   randomize/match it in training; (d) odometry is 5–16 ms old at the policy step (median per run, p95 up to 20 ms),
   at and above the trained 0–10 ms. (a), (b) (thrust-map gain), (c) (rate-gain) and (d) are covered by the v1 training
   distribution (`Quadsim-{Hover,Track,Waypoint}-v1`, docs/PROGRESS.md). Tracking, waypoint and navigation policies
   were also flown in SITL: tracking transfers (estimate-referenced RMSE as in quadsim); the waypoint course is completed
   but 25–35 % slower (cause not isolated); the velocity-interface navigation policy needs a matched velocity loop
   (§3). Depth/RGB policies have not been run in SITL (no rendered gz camera matched to the training camera yet).
2. **System identification of the real vehicle**: mass, inertia (bifilar pendulum), thrust and torque
   coefficients and motor time constant (thrust stand), rotor drag (flight logs). Update the preset and
   retrain; keep DR ranges centred on the identified values.
3. **Latency measurement** on the real pipeline (camera → estimator → policy → PX4) and matching the
   training delay distribution; increase the delay range if needed.
4. **Depth camera gap**: the depth policy is trained on analytic ray-cast depth with a stereo noise model.
   Validate against recorded RealSense/OAK-D frames (holes near edges, speckle, IR interference) and
   extend the noise model, or add an image-level domain-randomization stage.
5. **RGB policies** need texture/lighting randomization that the current procedural scenes lack.
