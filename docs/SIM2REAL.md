# Sim-to-Real guide

This document describes what the platform guarantees for transfer, what has been verified in simulation,
and the steps that remain before a real flight. **No policy from this repository has been flown on hardware
or against PX4 SITL yet.**

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
  (NED velocity, yawspeed). PX4's position/velocity controller replaces the simulated cascade.

### Thrust calibration (CTBR)
PX4's normalized thrust maps nonlinearly to Newtons. Set `thrust_scale` in the ROS 2 node so that the
simulated hover command maps to PX4's estimated hover thrust:
`thrust_scale = MPC_THR_HOVER / (m g / T_max_sim)` (for the X500 preset `m g / T_max = 0.592`).
A better approach (roadmap) is to identify the thrust curve on a thrust stand and add it to the contract.

## 4. Deployment steps (ROS 2 + PX4 v1.14+, uXRCE-DDS)

1. Export: `python scripts/rl/export.py --headless --task Quadsim-Hover-v0 --checkpoint <run>`.
2. Copy `<run>/exported/{policy.onnx, contract.json}` to the companion computer; install `onnxruntime`,
   `numpy` and the `quadsim` package (only `quadsim.deploy` is imported at runtime — no Isaac Sim needed).
3. Build `ros2/quadsim_px4_bridge` in a ROS 2 workspace that contains `px4_msgs` matching the PX4 version.
4. **PX4 SITL first** (gz `x500`, whose parameters equal the `x500` preset): start the uXRCE-DDS agent,
   `ros2 launch quadsim_px4_bridge policy.launch.py`, arm and switch to offboard from QGC.
5. Hardware: start tethered / in a net, low geofence radius, RC kill switch, offboard-loss failsafe set
   to *Land* or *Position hold*.

The node stops publishing set-points when odometry is older than `max_odom_age_s` or the vehicle leaves
the geofence, so PX4's offboard-loss failsafe takes over.

## 5. Open gaps before real flight (prioritized)

1. **Run the node against PX4 SITL** (x500) — not done. Expected issues: rate-loop gain mismatch, thrust
   curve, estimator latency of EKF2 (typically 5–20 ms; the policy was trained with 0–10 ms).
2. **System identification of the real vehicle**: mass, inertia (bifilar pendulum), thrust and torque
   coefficients and motor time constant (thrust stand), rotor drag (flight logs). Update the preset and
   retrain; keep DR ranges centred on the identified values.
3. **Latency measurement** on the real pipeline (camera → estimator → policy → PX4) and matching the
   training delay distribution; increase the delay range if needed.
4. **Depth camera gap**: the depth policy is trained on analytic ray-cast depth with a stereo noise model.
   Validate against recorded RealSense/OAK-D frames (holes near edges, speckle, IR interference) and
   extend the noise model, or add an image-level domain-randomization stage.
5. **RGB policies** need texture/lighting randomization that the current procedural scenes lack.
