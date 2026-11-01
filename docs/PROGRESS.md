# Progress log (verified results only)

Legend: ✅ verified by an executed test (artifact linked) · 🟡 implemented, partially verified · ⬜ not started

## Milestone 1 — Isaac Sim / Isaac Lab foundation ✅ (2026-11-01)
- Stack: Isaac Sim 5.1.0 (pip) + Isaac Lab v2.3.2 + torch 2.7.0+cu128 on RTX 5060 8 GB (sm_120).
- `scripts/validation/m1_isaac_smoke_test.py`: headless boot 2.8 s; 64 rigid bodies free fall 0.5 s at dt=1/400:
  |z − z_symplectic| = 9.5e-7 m, |z − z_analytic| = 6.1e-3 m (= expected O(dt) bias g·t·dt/2). → `docs/results/m1_smoke_test.json`
- Finding: PhysX rigid bodies integrate with symplectic (semi-implicit) Euler.
- Finding: Isaac Lab stock `Isaac-Quadcopter-Direct-v0` cannot run here: its Crazyflie USD download from NVIDIA's CDN
  did not complete within 625 s (network). Hence all quadsim assets are procedural.

## Milestone 2 — Quadrotor dynamics ✅ (2026-11-01)
- Pure-PyTorch core (`quadsim.core`): presets X500 (PX4 SITL gz model) and Crazyflie 2.x (Förster 2015),
  exact first-order asymmetric motor model, thrust/yaw-moment, rotor drag + rolling moment, body drag,
  rotor gyroscopic & spin-up torques, ground effect, wind (mean + Gauss–Markov turbulence + 1-cos gusts),
  push disturbances, mixer with PX4-style desaturation, rate PID / attitude P / position-velocity cascade, Lee SE(3) controller,
  RK4 + semi-implicit reference integrators.
- Unit tests `tests/core` (27 tests, CPU, ~10 s): ✅ all pass. Include torque sign conventions, exact motor response,
  Euler-equation conservation (Dzhanibekov tumbling), first-order convergence of semi-implicit → RK4,
  closed-loop cascade step (<5 cm), Crazyflie recovery from 46° tilt + 3.7 rad/s spin, SE(3) circle tracking RMSE < 0.1 m.
- Isaac integration: procedural USD quadrotor (explicit MassAPI), body wrench written via `permanent_wrench_composer`
  each physics step (200 Hz), PhysX integrates rigid body + contacts.
- `scripts/validation/m2_dynamics_validation.py` (both presets ✅ all pass):

| Test | X500 | Crazyflie | criterion |
|---|---|---|---|
| T1 PhysX mass / inertia vs preset | 0 / 0 rel. err | 0 / 0 | < 1e-4 |
| T2 hover drift (2 s) | 0.0 mm | 0.0 mm | < 5 mm |
| T3 thrust step accel. RMS err / peak | 0.019 % | 0.026 % | < 2 % |
| T4 yaw accel. vs τ/I_zz | 0.004 % | 0.008 % | < 3 % |
| T5 open-loop 1.5 s pos. err vs semi-implicit ref. | 0.40 mm | 2.1 mm | < 1 cm |
| T5 att. err vs semi-implicit ref. | 0.006° | 0.066° | < 0.5° |
| T5 pos. err vs RK4 (integration error of PhysX @200 Hz) | 3.3 mm | 8.9 mm | (info) |
| T6 closed-loop velocity cascade PhysX vs ref. (3 s) | 12 mm | 9 mm | < 3 cm |

  Artifacts: `docs/results/m2_dynamics_{x500,crazyflie}.json`, `docs/figures/m2_dynamics_*.pdf`.

## Milestone 3 — Sensors, visualization, recording ✅ (2026-11-01)
- Sensor models (`quadsim.core.sensors`): IMU (white noise density, turn-on bias, Gauss–Markov bias drift,
  scale/misalignment, sample-and-hold rate, latency), barometer (ISA), magnetometer, GPS, odometry/state-estimate
  model (noise, VIO-like drift, delay; profiles perfect/mocap/vio/gps_ins), depth-camera noise (stereo σ∝z²,
  dropouts, quantization, range limits), RGB photometric augmentation, analytic ray casting (ground, cylinders,
  yaw-boxes) for range finders / LiDAR rings / synthetic depth, wind and push disturbances.
- Unit tests `tests/core/test_sensors.py` (13): ✅ IMU noise std at 200 Hz within 5 % of σ_c·√f, turn-on bias,
  exact latency (3 steps), free-fall specific force = 0, ISA (100 m → 100129 Pa), exact ray geometry,
  depth-noise statistics, wind turbulence σ. **Bug found by the recording plot and fixed:** no-return pixels were
  receiving σ∝z² noise (regression test added).
- Rendered depth (RTX TiledCamera) vs analytic ray-cast depth, `scripts/validation/m3_depth_consistency.py`
  (64 random poses, 96×72, HFOV 87°, noise off), ✅:
  | camera pitch | median abs err | p90 | hit/miss agreement | self-occluded px |
  |---|---|---|---|---|
  | 0° (default mount (0.15, 0, −0.06) m) | 7.2 mm | 2.4 cm | 99.6 % | 0 % |
  | +15° | 9.0 mm | 2.3 cm | 99.5 % | 5.6 % (front rotors) |
  Finding: the first camera mount (0.12, 0, 0.02) saw the X500 rotor discs (3 % of pixels); default moved.
- Tasks registered (all pass the smoke test, finite obs, consistent spaces; `docs/results/env_smoke_test.json`):
  Hover, Hover-Motor, Track, Waypoint, Nav-Range, Nav-Depth (ray-cast), Nav-Depth-Render, Nav-RGB, Nav-RGBD.
- Synchronized HDF5 recorder (`quadsim.data.recorder`) + `scripts/data/record.py`: round-trip verified
  (shapes, monotonic time, dt, last-sample equality, embedded config) for tracking (state) and rendered-depth nav
  (images as uint16 mm). Plotting: `scripts/data/plot_recording.py` → `docs/figures/m3_recording_*.png`.
- Visualization: procedural target/reference markers, chase camera, `scripts/tools/view.py` (GUI / MP4).
  ✅ Headless MP4 of trained hover and depth-navigation policies (`outputs/videos/`, frames in
  `docs/figures/video_*_frames.png`) and interactive GUI (full `isaaclab.python.kit`, 500 steps, no errors) after
  the upstream patch below, on an idle GPU.
- Known transient: the very first launch with `--enable_cameras` crashed once in `libslang.so` while compiling
  shaders (empty shader cache); every later launch succeeded.

## Milestone 4 — State-based RL ✅ (hover, tracking, waypoints; multi-seed hover in session 3)
- PPO (rsl_rl 3.1.2), asymmetric actor-critic: actor sees only deployable obs (noisy, delayed odometry + gyro +
  last action + task), critic additionally sees true state, rotor speeds, wind, disturbance, DR parameters.
  Control interface CTBR (collective thrust + body rates → onboard PX4-like rate PID at 200 Hz), policy 50 Hz.
  Training-time randomization: mass ±10 %, inertia ±15 %, CoM ±1 cm, k_f ±10 %/rotor, k_m ±15 %, motor τ ×[0.7,1.6],
  ω_max ×[0.92,1], drag ×[0.5,1.5], payload −5…+10 % weight, sensor-noise scale ×[0.5,1.5], action delay 0–2
  physics steps, odometry delay 0–2 steps.
- Hover: 4096 envs, 600 iterations (79 M steps), ~190 k steps/s alone on the RTX 5060, 3.9 GB VRAM.
  Evaluation (deterministic policy, 1000 episodes, same seeds for both controllers, `scripts/rl/evaluate.py`):

  | condition | controller | success¹ | crash | final pos. err | final yaw err |
  |---|---|---|---|---|---|
  | nominal (training dist.) | **PPO** | **99.9 ± 0.2 %** | 0.1 % | **1.38 ± 0.26 cm** | 4.4° |
  | nominal | cascade PID | 88.7 ± 2.0 % | 0.5 % | 6.1 ± 0.7 cm | 1.5° |
  | + wind 0–3 m/s, turbulence, gusts, pushes ≤0.3 mg (unseen) | **PPO** | **97.3 ± 1.0 %** | 0.1 % | 2.9 cm | 4.6° |
  | same | cascade PID | 78.6 ± 2.5 % | 0.7 % | 8.3 cm | 1.8° |

  ¹ success = |e_p| < 10 cm and |v| < 0.2 m/s at the end of the 8 s episode starting from up to 57° tilt, 1.5 m/s,
  3 rad/s and 1.5 m offset. ± = 95 % CI. Results: `docs/results/eval_hover_*.json`.
  Limitation: the PPO policy's final yaw error (4.4°) is worse than the baseline's (weak yaw reward).

- Tracking: 4096 envs, 1500 iterations (197 M steps). Evaluation, 500 episodes each, same seeds
  (`docs/results/eval_track_*.json`):

  | reference set | controller | RMSE | max err | crash |
  |---|---|---|---|---|
  | aggressive (≤4 m/s, accel. up to ~15 m/s²) | **PPO** | **12.6 ± 1.5 cm** | 40.6 cm | 0.2 % |
  | aggressive | cascade PID (vel+acc FF) | 46.6 ± 3.7 cm | 103 cm | 3.2 % |
  | aggressive | SE(3) geometric (v2) | 49.2 ± 3.9 cm | 98 cm | 1.8 % |
  | aggressive | SE(3) v1 (torque-level law mapped to rates at 50 Hz) | 114 cm | 222 cm | **53 %** |
  | moderate (≤2 m/s, ≤0.35 Hz) | **PPO** | **5.8 ± 0.2 cm** | 29 cm | 0 % |
  | moderate | cascade PID | 17.4 ± 1.3 cm | 50 cm | 0 % |
  | moderate | SE(3) geometric (v2) | 22.7 ± 1.4 cm | 53 cm | 0.2 % |

  The first SE(3) baseline was a design error (high-gain attitude law executed at the 50 Hz policy rate through
  the delay chain); it was replaced by Lee's position law + onboard attitude/rate loops. Baseline gains are
  defaults, not tuned. Part of the aggressive set exceeds the baselines' tilt limits (0.6–0.7 rad).
- Waypoint: 4096 envs, 1500 iterations. 500 episodes (`docs/results/eval_waypoint_nominal.json`):
  PPO **99.4 % success**, 5 waypoints in **6.6 s**, 0.6 % crash; cascade 92.8 %, 10.4 s, 2.2 % crash.
- DR ablation (hover; identical policy trained without DR/noise/latency; 1000 episodes per row;
  `docs/results/eval_dr_ablation_*.json`):

  | eval condition | with DR | without DR |
  |---|---|---|
  | ideal (no DR/noise/delay) | 100 %, 0.22 cm, yaw 3.7° | 100 %, 0.28 cm, yaw 9.8° |
  | training randomization | 99.9 %, 1.38 cm | 99.9 %, 2.05 cm |
  | harsh OOD (mass ×0.8–1.25, motor τ ×1.5–2.5, k_f ×0.8–1, noise ×1.5–2, 10–20 ms delay, wind) | 78.5 ± 2.5 %, crash 3.9 %, 16.4 cm | 77.2 ± 2.6 %, crash 4.9 %, 18.6 cm |

  Honest reading: DR lowers errors everywhere but the harsh-condition success difference is **not significant**;
  CTBR + onboard rate loop already absorbs much parameter variation. Single seed per arm.
- SAC (skrl 1.4.3, `scripts/rl/train_sac.py`), hover, 256 envs, 30 k vectorized steps (7.7 M transitions, 21 min):
  pipeline ✅ executes and learns (episode return −11.2 → +1.7, still rising), but the policy is **not competitive**:
  0 % crash, 0 % success, 1.13 m final error (PPO reaches return ≈ 11). Under-trained (UTD ratio 1/256), untuned.

## Milestone 5 — Vision / navigation RL ✅ (criteria met; moderate depth-policy performance)
Full-density evaluation (curriculum off, 4–24 obstacles, mean ≈ 12; 500 episodes unless noted;
`docs/results/eval_nav_*.json`):

| policy input | controller | success | collision | time to goal |
|---|---|---|---|---|
| range ring (360°, 32 rays) | potential field (paired run / independent run) | 49.8 % / 39.3 % | 5.2 % / 4.4 % | 10.2 s |
| range ring | PPO, reward v1 (prox. 0.2, crash 10, γ 0.99) | 52.9 % | 46.3 % | 3.7 s |
| range ring | **PPO, reward v2** (prox. 2.0, crash 20, γ 0.995) | **83.0 ± 3.3 %** | 16.6 % | 4.2 s |
| depth 64×48, analytic (training sensor) | PPO v2, CNN, 900 it. | 61.1 ± 4.3 % | 36.9 % | 4.3 s |
| depth, **zero-shot on RTX-rendered depth** (300 ep.) | same policy | 66.7 ± 5.3 % | 32.7 % | 4.2 s |

- Reward v1 → v2 is a documented reward iteration (v1 policy flew at the speed limit and accepted collisions).
- The potential field rarely collides but is trapped in local minima (42–52 % time-outs). Its two runs differ by more
  than the per-run CI: layouts are not paired across runs (shared RNG stream consumed differently).
- Depth policy: analytic → rendered depth shows no degradation (within CI) — supports training on the 12× faster
  analytic sensor. Its gap to the 360° range policy is plausibly the 87° field of view without memory (unverified).
- Depth run stopped at iteration 900 of 1500 (curriculum reached 24 obstacles at it. 501; success flat over the
  last 300 it.) for time reasons; more training may help.
- RGB-D: rendered RGB + depth → CNN PPO executes end to end (10-iteration run, 64 envs). No performance claim.

## Milestone 6 — Sim-to-Real interfaces ✅ (bridge flown in PX4 SITL; all tasks deployable)
- Export (`scripts/rl/export.py`): TorchScript + ONNX with normalizer and clamp; ONNX vs PyTorch max |Δ| = 3.6e-7 ✅.
  Machine-readable interface contract (`contract.json`, example `docs/results/contract_hover.json`).
- `quadsim.deploy.interface` (NumPy only): ENU/FLU ↔ NED/FRD, PX4 odometry → sim conventions, observation
  builder, action decoder → PX4 `VehicleRatesSetpoint` / `TrajectorySetpoint`. 7 unit tests ✅ (incl. frame
  consistency of a physical body vector through both quaternion conventions).
- Deployment pipeline in the loop (`scripts/validation/m6_deploy_consistency.py`, hover, 512 episodes) ✅:
  ONNX policy fed by the NumPy builder flies the simulator; max obs error 9.5e-7, max action error 1.0e-6,
  success 99.8 % (identical to PyTorch).
- ROS 2 package `ros2/quadsim_px4_bridge` (offboard node, launch, config): ✅ flown against PX4 v1.16.2 SITL with the
  hover policy after fixing four bugs (see "PX4 SITL cross-check" below); not flown on hardware.

## Milestone 7 — Docker, benchmark, documentation ✅
- `docker/Dockerfile` on the official `nvcr.io/nvidia/isaac-lab:2.3.2`; compose services test/train/gui/tensorboard.
- NGC refuses direct connections from this network (HTTP 403); the Docker daemon ignores the shell proxy.
  `scripts/tools/pull_image_via_proxy.py` pulls via the registry API through the user's proxy with sha256
  verification and `docker load`s the image (8.41 GB compressed).
- `docker compose run --rm test` ✅: unit tests and the Isaac Sim GPU smoke test pass inside the container.
- Benchmark (idle GPU, env stepping only; `docs/results/benchmark.jsonl`): Hover 16 k / 63 k / 237 k / 439 k env-steps/s
  at 256 / 1024 / 4096 / 8192 envs (3.9 GB at 8192); Nav-Depth (analytic) 83 k @ 2048 (4.4 GB); RTX-rendered depth
  6.8 k @ 256 (5.6 GB); RGB-D 3.8 k @ 128.
- M2 regression re-run with the upstream patch active: all pass.
- README, `docs/SIM2REAL.md`, LaTeX report `report/report.pdf` (14 pages; tables generated from `docs/results`).

## PX4 SITL cross-check (roadmap "PX4 SITL/HITL loop") ✅ hover policy (2026-11-01)
**Result: the exported hover policy flies PX4 v1.16.2 SITL (Gazebo `x500`) as an offboard controller through
`ros2/quadsim_px4_bridge`.** All 10 full-schedule runs with the fixed bridge (60 s in OFFBOARD: hold, 1 m x step,
1 m z step) completed without leaving OFFBOARD or triggering a failsafe, followed by a normal hand-back and landing.
PX4 received 3004–3006 rate set-points per 60 s (max gap 21 ms, none > 30 ms). The original bridge did **not** fly
(bugs below). Recommended configuration: thrust-curve map (finding 2).

Setup (reproducible: `ros2/px4_sitl/setup_px4_sitl.sh`, `scripts/sitl/run_sitl.sh`; native on the host, no Docker):
PX4 v1.16.2 SITL, airframe 4001 `gz_x500`, Gazebo Harmonic 8.14 (headless, lockstep, real-time factor 1.000),
Micro-XRCE-DDS Agent (UDP), ROS 2 Jazzy (the host's install; the bridge only uses rclpy APIs that also exist in Humble),
`px4_msgs` generated from the same PX4 tree. PX4 parameters: airframe defaults + `NAV_DLL_ACT=0` (no GCS).
SITL-only instrumentation: gz ground-truth topics exported over uXRCE-DDS (evaluation only). Policy
`quadsim_hover/2026-11-01_02-30-46_ctbr_s42` (model_599, ONNX), 50 Hz, CTBR → `VehicleRatesSetpoint`.
Scenario (`scripts/sitl/sitl_scenario.py`): PX4 take-off to 2.5 m → PX4 Hold → `thrust_scale` from PX4's hover-thrust
estimate → goal := current pose → OFFBOARD (policy) → 30 s hold → goal +1 m x (15 s) → +1 m z (15 s) → PX4 Hold →
land. quadsim counterpart (`scripts/sitl/quadsim_scenario.py`, 64 envs, seed 7): same schedule, every env starts at
the goal at rest (= hand-over). Both are evaluated by the same code (`quadsim.deploy.closed_loop_metrics`, unit-tested):
"truth" = simulator ground truth, "estimate" = what the policy sees (quadsim odometry model / PX4 EKF2).
Comparison: `scripts/sitl/compare.py` → `docs/results/sitl_comparison.json` (per-run values),
figure `scripts/sitl/plot_comparison.py`. Raw runs (`.npz`, bridge step logs, PX4 `.ulg`) in `logs/sitl/` (git-ignored).

| metric | quadsim: nominal (n=64 envs) | quadsim: training DR (n=64 envs) | PX4 SITL, linear thrust map (n=3 runs) | PX4 SITL, thrust-curve map (n=4 runs) | PX4 SITL, linear + CA_ROTOR fix (n=2 runs) | PX4 SITL, uncalibrated (scale 1.0) (n=1 runs) |
|---|---|---|---|---|---|---|
| hover RMS 3-D error, ground truth [cm] | 1.59 ± 0.46 | 1.76 ± 0.62 | 4.12 ± 1.9 | 6 ± 1.1 | 4.71 ± 1.1 | 7.74 |
| hover RMS horizontal / vertical, ground truth [cm] | 1.27 ± 0.44 / 0.868 ± 0.4 | 1.32 ± 0.52 / 1.05 ± 0.59 | 2.95 ± 1.1 / 2.83 ± 1.6 | 3.97 ± 0.73 / 4.32 ± 1.5 | 3.95 ± 1 / 2.55 ± 0.38 | 2.46 / 7.34 |
| hover RMS 3-D error w.r.t. onboard estimate [cm] | 1.88 ± 0.021 | 2.05 ± 0.49 | 1.02 ± 0.13 | 0.828 ± 0.04 | 1.01 ± 0.22 | 1.55 |
| estimate vs ground truth position RMS (hover) [cm] | 2.28 ± 0.31 | 2.42 ± 0.58 | 3.84 ± 2.1 | 5.81 ± 1.2 | 4.51 ± 1.1 | 6.38 |
| x step (truth): rise time 10-90 % [s] | 0.652 ± 0.011 | 0.647 ± 0.016 | 0.553 ± 0.025 | 1.03 ± 0.46 | 1.01 ± 0.45 | 0.6 |
| x step (truth): overshoot [%] | 1.34 ± 1.2 | 1.55 ± 1.3 | 3.47 ± 0.67 | 2.49 ± 2.5 | 2.99 ± 2.4 | 4.62 |
| x step (truth): settling time (5 cm) [s] | 1.05 ± 0.029 | 1.05 ± 0.04 | 1.87 ± 0.081 | 7.28 ± 5.1 | 10.7 ± 3.7 | 1.87 |
| z step (truth): rise time 10-90 % [s] | 0.633 ± 0.011 | 0.689 ± 0.054 | 0.92 ± 0.099 | 1.06 ± 0.13 | 0.83 ± 0.01 | 2.04 |
| z step (truth): overshoot [%] | 1.6 ± 1.2 | 1.25 ± 1.3 | 2.31 ± 1.8 | 0 | 3.6 ± 0.72 | 0 |
| z step (truth): settling time (5 cm) [s] | 0.962 ± 0.036 | 1.2 ± 1 | 1.52 ± 0.16 | 4.13 ± 2.6 | 1.44 ± 0.071 | nan |
| x step w.r.t. estimate: rise [s] / overshoot [%] / settling [s] | 0.646 ± 0.014 / 3.86 ± 0.39 / 1.07 ± 0.022 | 0.643 ± 0.022 / 3.9 ± 1.1 / 2.86 ± 3.8 | 0.536 ± 0.0065 / 1.89 ± 0.22 / 1.84 ± 0.019 | 0.55 ± 0.01 / 1.49 ± 0.1 / 1.86 ± 0.02 | 0.54 ± 0.004 / 1.68 ± 0.03 / 1.86 ± 0.023 | 0.584 / 2.69 / 1.89 |
| z step w.r.t. estimate: rise [s] / overshoot [%] / settling [s] | 0.625 ± 0.015 / 4.1 ± 0.29 / 0.998 ± 0.028 | 0.685 ± 0.056 / 3.86 ± 1 / 4.09 ± 5 | 0.643 ± 0.014 / 1.34 ± 0.49 / 1.07 ± 0.047 | 0.636 ± 0.012 / 0.44 ± 0.017 / 1.02 ± 0.017 | 0.624 ± 0.024 / 1.03 ± 0.65 / 1.07 ± 0.027 | 0.744 / 0 / 1.24 |
| rate cmd -> true rate: delay (roll / pitch) [ms] | 13.2 ± 4.3 / 13.2 ± 4.3 | 16.2 ± 6.3 / 15.6 ± 5.3 | 22.7 ± 0.94 / 12.7 ± 1.9 | 31 ± 1 / 10 ± 1.4 | 15 ± 5 / 10 ± 2 | 28 / 8 |
| rate cmd -> true rate: time constant (roll / pitch) [ms] | 53.1 ± 1.9 / 52 ± 6.9e-15 | 45.5 ± 11 / 48.8 ± 9.7 | 46.9 ± 1.7 / 39.1 ± 1.5 | 38 / 42 ± 1.4 | 38.2 ± 3 / 41.3 ± 3.2 | 44.5 / 38 |
| rate cmd -> true rate: gain (roll / pitch) | 1.02 ± 0.021 / 1.01 ± 0.0066 | 0.841 ± 0.19 / 0.907 ± 0.16 | 0.995 ± 0.007 / 0.968 ± 0.0093 | 0.959 ± 0.0065 / 1.02 ± 0.018 | 1.04 ± 0.024 / 0.993 ± 0.027 | 0.945 / 0.981 |
| rate cmd -> true rate: VAF (roll / pitch) | 0.972 ± 0.0062 / 0.986 ± 0.0017 | 0.838 ± 0.14 / 0.906 ± 0.12 | 0.955 ± 0.0014 / 0.97 ± 0.0067 | 0.959 ± 0.00051 / 0.985 ± 0.002 | 0.97 ± 0.0094 / 0.974 ± 0.0098 | 0.94 / 0.984 |
| hover collective action a0: std / dominant frequency [Hz] | 0.174 ± 0.018 / 3.56 ± 0.42 | 0.176 ± 0.038 / 3.45 ± 0.55 | 0.218 ± 0.14 / 4 ± 0.2 | 0.00465 ± 0.00016 / 0.813 ± 0.27 | 0.178 ± 0.17 / 2.25 ± 1.5 | 0.0222 / 7.5 |
| hover vertical velocity RMS, truth [cm/s] | 5.39 ± 1.1 | 5.41 ± 1.4 | 10.1 ± 6.2 | 1.02 ± 0.036 | 8.79 ± 7.8 | 1 |
| hover body-rate RMS, truth (roll / pitch) [rad/s] | 0.172 ± 0.013 / 0.175 ± 0.017 | 0.186 ± 0.047 / 0.189 ± 0.046 | 0.0731 ± 0.034 / 0.0782 ± 0.043 | 0.0259 ± 0.0022 / 0.0184 ± 0.0021 | 0.0589 ± 0.034 / 0.0789 ± 0.06 | 0.0272 / 0.257 |
| hover body-rate dominant frequency (roll / pitch) [Hz] | 3.11 ± 1.9 / 5.05 ± 1.4 | 3.07 ± 1.9 / 4.79 ± 1.5 | 2.92 ± 1.4 / 2.92 ± 1.4 | 0.813 ± 0.11 / 0.75 | 2.25 ± 1.5 / 2.25 ± 1.5 | 0.75 / 7.5 |
| estimate attitude lag vs truth (roll / pitch) [ms] | 4.8 ± 4 / 4.7 ± 4 | 6.05 ± 4 / 6 ± 4.1 | 3 / 2 | 3.25 ± 0.43 / 2 | 3 / 2 | 4 / 2 |
| odometry age at policy step: median / p95 [ms] | – | – | 11.9 ± 0.94 / 15.9 ± 1 | 7.99 ± 1.8 / 12 ± 1.7 | 12.5 ± 3.2 / 16.6 ± 3.2 | 13.8 / 17.9 |

± = std over runs (SITL) or over the 64 envs (quadsim). Hover window 10–30 s after the hand-over. "w.r.t. estimate"
rows measure what the policy regulates (control performance); "truth" rows include the estimator's error. Rate
response: output-error fit of `rate(s)/cmd(s) = K e^{-Ls}/(τs+1)` over the whole flight (closed-loop data; VAF =
variance accounted for); SITL truth rate = gz ground truth at 50 Hz. Columns: linear map = `thrust_scale` only
(SIM2REAL §3, the documented procedure); thrust-curve map = exact inverse of the gz motor curve (finding 2);
CA_ROTOR fix = linear map + PX4 allocation geometry set to the gz rotor positions (finding 4).

![sim vs SITL](figures/sitl_vs_quadsim.png)

Findings
1. **Control performance transfers; the estimator does not match the training model.** Referenced to the estimate
   the policy regulates, SITL (thrust-curve map) matches or beats quadsim: hover RMS 0.83 cm (quadsim 1.9 cm), z step
   rise 0.64 s / settling 1.02 s (0.63 s / 1.00 s), x step rise 0.55 s (0.65 s). Referenced to ground truth SITL is
   worse (hover 4–6 cm vs 1.6 cm) because EKF2 (GNSS + baro, gz sensor noise) has a slowly varying position error of
   2.3–7.7 cm RMS per run (bias up to 6 cm in z), whereas quadsim's odometry model is white-noise dominated (2.3 cm).
   Truth-referenced step metrics in SITL therefore mostly measure EKF2 drift (e.g. x settling 7–11 s w.r.t. truth vs
   1.86 s w.r.t. the estimate). The smooth EKF2 estimate also makes the policy's commands much quieter than in quadsim
   (figure, rows 2–3), which is why the control error is smaller in SITL.
2. **Heave limit cycle with the linear thrust map; removed by the thrust-curve map.** With `thrust_scale` only, 3 of 5
   calibrated runs (default_r1/r3, geom_r1) showed a persistent 3.75–4 Hz vertical oscillation from the hand-over to the
   end: collective action std 0.30–0.36 (swinging −0.4…+0.7), vertical velocity 0.14–0.17 m/s RMS (truth and EKF2),
   coupled roll/pitch-rate oscillation 0.09–0.15 rad/s at ~5 Hz; position RMS w.r.t. the estimate stayed ≈ 1 cm. The
   other 2 runs were quiet (a0 std ≤ 0.023, 0.010–0.014 m/s). Cause (supported, not proven): the linear map is exact
   only at hover; around hover the gz motor curve (`T = T_max (0.15 + 0.85 u)^2`) has 1.6× the thrust gain the policy
   was trained with, so the vertical loop's lightly damped ~3.5–4 Hz mode, which is also visible in quadsim as
   noise-driven heave (vertical velocity 5.4 cm/s RMS, collective chatter at 3.6 Hz), becomes a limit cycle. With the
   exact inverse of the curve (`ThrustCurve`, bridge parameter `thrust_curve`) **0 of 4 runs** oscillated (a0 std
   0.004–0.005, vertical velocity 1.0 cm/s, body rates ≤ 0.03 rad/s). 0/4 vs 3/5 is suggestive but small
   (Fisher one-sided p ≈ 0.12); what selects the quiet or the oscillating regime under the linear map is not identified
   (not odometry age). For hardware: identify the thrust curve (thrust stand) and use the curve map; consider
   randomizing the thrust-curve shape in training.
3. **Thrust calibration.** PX4's hover-thrust estimate is 0.7295 ± 0.0001 over 10 runs (analytic value for the gz
   motor model 0.729), while the x500 default `MPC_THR_HOVER` is 0.60 → linear `thrust_scale` = 0.7295 / 0.592 = 1.232
   (`MPC_THR_HOVER` would give 1.01). With the curve map the calibrated scale is 1.0016, i.e. the gz x500 and the
   quadsim preset agree on hover thrust within 0.2 %. The linear map also saturates (u = 1) once the policy asks for
   more than 81 % of T_max (unit test). With `thrust_scale = 1.0` (−19 % thrust at the hover command, outside the
   training randomization) the policy still flies: it raises its collective (a0 0.43) and holds −1.3 cm below the goal
   w.r.t. the estimate; z step rise 0.74 s; this run showed a 7.5–9.5 Hz pitch-rate oscillation (0.29 rad/s RMS, not
   investigated).
4. **Rate loop: pitch as fast as quadsim's, roll slower because of the allocation geometry.** Command → body-rate fits
   (truth body rates from central differences of the ground-truth attitude; an earlier version of this analysis used
   PX4's backward-differenced Euler-angle "ground-truth angular velocity" and overstated all SITL delays by ~10 ms —
   corrected): pitch delay 8–14 ms, τ 38–44 ms in SITL vs 13 ms / 53 ms in quadsim (13 ms = 0–10 ms action delay +
   motor lag + PID), i.e. PX4's pitch loop is slightly *faster*. Roll is slower (22–32 ms delay with the default
   geometry) because PX4's x500 airframe `CA_ROTOR*` = (0.13, 0.20–0.22) m differ from the gz rotor positions
   (±0.174, ±0.174) m, so the allocator under-actuates roll. **Verified:** with `CA_ROTOR*` = gz geometry, roll delay
   20 ms (quiet run) / 10 ms (oscillating run) vs 24–32 / 22 ms with the default geometry under the same conditions;
   pitch unchanged. Fitted delays depend on the excitation (closed-loop identification, read as ±10 ms).
5. **x step settles slower in SITL** (1.86 ± 0.02 s vs 1.07 s w.r.t. the estimate, all SITL configurations): after a
   faster rise the position dips back to ≈ 0.9 m at ≈ 1.5 s before settling (figure, top left); quadsim settles
   monotonically. Cause not isolated (the pitch loop is not slower; candidates: EKF2's velocity dynamics during
   accelerations, PX4's rate-loop damping / filters, the different τ).
6. **Hover collective and command chatter in quadsim.** In nominal quadsim the policy hovers with mean a0 ≈ 0.150
   instead of the static hover action 0.184 (−2.9 % commanded thrust). Replaying its command sequence through the x500
   motor model offline shows that the asymmetric motor lag (τ_up 12.5 ms < τ_down 25 ms) rectifies the chattering
   command (a0 std 0.17) into +2.2 % mean thrust (symmetric lag: −0.1 %), i.e. most of the gap. In SITL with quiet
   commands the policy hovers at a0 = 0.179–0.182 ≈ the hover action; the oscillating runs at 0.11–0.12.
7. **Latency.** Odometry (EKF2 `VehicleOdometry`, 100 Hz) is 3.8–8.0 ms old on arrival at the bridge (median per run;
   p95 8–12 ms) and 5–16 ms old at the policy step (median per run; p95 9–20 ms, max 23 ms) because the 50 Hz policy
   timer is not synchronized with the odometry stream — at and above the upper end of the trained 0–10 ms odometry
   delay. EKF2's attitude output lags the gz truth by only 2–4 ms (quadsim: 4.8 ms mean, 0–10 ms by construction).
   Policy period 20.0 ms (p99 ≤ 20.4 ms, max 21 ms).

Bridge bugs found and fixed (each reproduced first)
- **Versioned topics:** PX4 v1.16 publishes `VehicleStatus` (message version 1) on `/fmu/out/vehicle_status_v1`; the node
  subscribed to `/fmu/out/vehicle_status`, never saw OFFBOARD and never published a set-point. Reproduced with the
  original node (`logs/sitl/orignode_r1`): PX4 accepted OFFBOARD and stayed in it for 8 s while the node sent **0**
  rate set-points; the vehicle flew open-loop on the last attitude-controller set-point (3.5 m drift, up to 1.2 m/s)
  and no failsafe triggered. Fix: topic names derived from `px4_msgs` `MESSAGE_VERSION` (`px4_topic`).
- **Heartbeat not gated (safety):** the `OffboardControlMode` heartbeat was published before the stale-odometry and
  geofence checks, but PX4 detects offboard loss from that heartbeat alone (`offboardCheck.cpp`), so the documented
  "stop publishing → PX4 failsafe" could never happen (the original-node run above is this failure mode). Fix:
  heartbeat only while the node can command. Verified (`logs/sitl/fence_r1`): goal moved 4 m away (fence 3 m) → PX4
  flagged the failsafe and left OFFBOARD after 1.03 s (`COM_OF_LOSS_T` = 1 s) and, without RC, landed at 0.7 m/s.
- **Packaging:** missing `setup.cfg` → `ros2 launch/run` could not find `policy_node`.
- `last_action` input contained policy outputs computed (but never applied) while PX4 was flying → held at the hover
  action until the hand-over; odometry with NaN fields or an unsupported pose/velocity frame is rejected; body-FRD
  velocity is rotated to NED.
- New: thrust-curve map (`thrust_curve`), runtime goal / thrust changes via ROS parameters, optional per-step CSV log,
  launch arguments. The per-step logic lives in the ROS-free `quadsim.deploy.offboard.OffboardPolicy`; tests in
  `tests/core/test_offboard.py`, `test_deploy_interface.py`, `test_closed_loop_metrics.py` (64 core tests pass).
  A column-order bug in the new step log (q/velocity swapped vs the header) was found during the analysis and fixed;
  the analysis did not use the affected columns and the existing logs were rewritten in the corrected order.

Tooling findings: ROS 2's vendored `gz` on `PATH` cannot start PX4's Gazebo world (PX4 is run with `/opt/ros` removed
from `PATH`); a `gz sim` server surviving a run is silently re-used by the next PX4 instance with the old (crashed)
vehicle → processes are killed by process group and `run_sitl.sh` refuses to start next to a running instance; rclpy's
`MultiThreadedExecutor` dropped ~35 % of 300 msg/s in the recorder (PX4 itself received every set-point) →
`SingleThreadedExecutor`; `ls | head` under `set -o pipefail` aborted the log copy (SIGPIPE) in some runs.

Caveats: SITL only; one policy (hover, one seed); gz uses the same motor/drag parameters as the quadsim preset, so SITL
cross-checks the rigid-body engine, PX4's control/estimation stack and the interface — not the motor model; 3 + 4 + 2 + 1
SITL runs, so run-to-run statistics are coarse; rate fits use closed-loop data (VAF 0.92–0.98) and 50 Hz ground truth;
tracking / waypoint / velocity-interface policies not yet run in SITL.

## Continuation: audit, all tasks in PX4 SITL, sim-to-real v1 distribution (2026-11-01, session 3)
Goal: re-verify the reported results, take tracking / waypoint / navigation policies through PX4 SITL, and turn the
sim-to-sim gaps found in SITL into improvements of the training environment. Worktree branch
`claude/frosty-swartz-ab3133`; runs and logs in the main checkout's `logs/` (worktree `logs/*` are symlinks).

### Re-verification of reported results ✅
| item | reported | re-run (this session) | verdict |
|---|---|---|---|
| unit tests | 48 (README) / 60 | 80 pass (new tests added) | ✅ |
| env smoke test (6 non-camera tasks) | pass | pass | ✅ |
| M2 dynamics (x500) | all pass | all pass | ✅ |
| M6 deploy consistency (hover) | obs 9.5e-7, success 99.8 % | identical | ✅ |
| hover eval, 1000 ep. (PPO / cascade) | 99.9 % 1.38 cm / 88.7 % 6.1 cm | **bit-identical** (also after this session's code changes) | ✅ |
| tracking eval, 500 ep. | 12.6 ± 1.5 cm | 12.2 cm | ✅ within CI (original used 250 envs, re-run 500 → other samples; `evaluate.py` now records `num_envs`) |
| waypoint eval, 500 ep. | 99.4 ± 0.7 % | 98.8 % | ✅ within CI (same reason) |
| camera tasks, recording, resume, rendered depth | pass | REVERIFY2_PLACEHOLDER | |

### Deployment of all tasks ✅
- Tracking, waypoint and range-navigation policies exported (contract now has `task_params`: horizon, reach radius,
  ring geometry). `quadsim.deploy.offboard.OffboardPolicy` gained task objects (onboard sum-of-sines reference with
  task time from the hand-over, training waypoint switching, LaserScan → 32-ray ring with stale-sensor gating).
- Deploy-in-the-loop (`scripts/validation/m6_deploy_consistency.py`, generalized; ONNX + NumPy observation rebuilt from
  onboard quantities flies the simulator): hover / tracking / waypoint / nav-range max obs error 9.5e-7 / 7.5e-6 /
  2.9e-6 / 1.2e-6, max action error ≤ 8.4e-6; success / crash vs PyTorch within 3 points (nav: 80.8 vs 83.3 %) — all pass.

### SITL findings (methodology, estimator)
- **EKF2 GNSS delay mis-configuration in SITL.** EKF2's position led the gz truth by 0.11 s in motion: `EKF2_GPS_DELAY`
  defaults to 110 ms (right for real receivers) but gz's GNSS has no latency. With 0 ms: lead −3 ms, truth-referenced
  tracking error 14 → 7 cm (estimate-referenced unchanged). `run_sitl.sh` now sets it (`SITL_GPS_DELAY_MS`); hover runs
  of the first SITL section were stationary and unaffected.
- **Ground-truth artefacts of PX4's gz bridge**: ground-truth velocity and "angular velocity" are backward differences
  (the latter of Euler angles) → truth body rates are recomputed by central differences of the ground-truth quaternions
  (`body_rates_from_quat`); rate-loop fits in `docs/results/sitl_comparison.json` were recomputed.
- **EKF2 error statistics** (`scripts/sitl/estimator_stats.py` → `docs/results/sitl_estimator.json`, 6 hover runs):
  per-flight bias ~1.8 / 1.8 / 3.9 cm (x/y/z), slow error std 1.4 cm with 2–3 s correlation, fast error 1.3 mm,
  velocity error ~1 cm/s (0.3–2 s correlation), roll/pitch 0.2 mrad — quadsim's v0 model is mostly white
  (1 cm, 3 cm/s, 10 mrad). These define the v1 estimator model.

### Tracking, waypoints and navigation in PX4 SITL
Same reference / course / layouts in both simulators (`scripts/sitl/quadsim_task_scenario.py`, `sitl_scenario.py
--task`), thrust-curve map, `EKF2_GPS_DELAY=0`, scored by `scripts/sitl/compare_tasks.py` → `docs/results/sitl_tasks.json`.

**Tracking — circle (r 1 m, 0.2 Hz)** (SITL n=2 runs, quadsim n=32 envs; RMSE after 2 s)

| metric | quadsim | PX4 SITL |
|---|---|---|
| RMSE truth [cm] | 1.78 ± 0.37 | 6.34 ± 0.74 |
| RMSE horizontal, truth [cm] | 1.38 ± 0.33 | 5.12 ± 1.1 |
| RMSE vertical, truth [cm] | 1.07 ± 0.37 | 3.65 ± 0.25 |
| max error, truth [cm] | 3.33 ± 0.57 | 10.7 ± 0.15 |
| RMSE w.r.t. estimate [cm] | 2.47 ± 0.059 | 2.22 ± 0.24 |


**Tracking — figure-8 (r 1.5 m, 0.15 Hz)** (SITL n=2 runs, quadsim n=32 envs; RMSE after 2 s)

| metric | quadsim | PX4 SITL |
|---|---|---|
| RMSE truth [cm] | 2.67 ± 0.23 | 9.2 ± 0.4 |
| RMSE horizontal, truth [cm] | 2.34 ± 0.19 | 8.81 ± 0.15 |
| RMSE vertical, truth [cm] | 1.24 ± 0.32 | 2.51 ± 0.94 |
| max error, truth [cm] | 5.46 ± 0.65 | 16.8 ± 0.82 |
| RMSE w.r.t. estimate [cm] | 3.04 ± 0.054 | 3.42 ± 0.37 |

**Waypoints** (5-waypoint course; SITL n=2 runs, quadsim n=32 envs)

| metric | quadsim | PX4 SITL |
|---|---|---|
| course completed (policy's own logic) | 1 | 1 |
| time to finish [s] | 5.13 ± 0.017 | 6.74 ± 0.27 |
| waypoints passed within 0.3 m on truth (of 5) | 5 | 4 ± 1 |
| estimate error during the course, RMS / max [cm] | – | 15.2 ± 0.68 / 31.3 ± 0.31 |

**Navigation (range ring, velocity interface)** — per layout: quadsim success / collision rate over its envs vs the PX4 SITL outcome (time to goal [s])

| layout | obstacles | quadsim | PX4 default velocity loop | PX4 velocity loop matched to quadsim |
|---|---|---|---|---|
| 0 | 7 | 1.00 / 0.00 (4.05) | goal (4.25), min clearance 0.37 m | goal (4.02), min clearance 0.46 m |
| 1 | 15 | 0.38 / 0.62 (4.71) | collision, min clearance 0.14 m | collision, min clearance 0.12 m |
| 2 | 13 | 0.97 / 0.03 (4.23) | goal (4.38), min clearance 0.45 m | goal (4.21), min clearance 0.53 m |
| 3 | 15 | 0.97 / 0.03 (4.34) | collision, min clearance 0.25 m | goal (4.49), min clearance 0.49 m |
| 4 | 11 | 0.91 / 0.06 (4.33) | collision, min clearance 0.28 m | collision, min clearance 0.31 m |
| 5 | 15 | 1.00 / 0.00 (4.28) | collision, min clearance 0.23 m | goal (4.46), min clearance 0.53 m |
| all | | 0.87 / 0.12 | 0.33 success | 0.67 success |

- **Tracking transfers at the control level** (estimate-referenced RMSE equal within noise); truth-referenced error is
  3–4× larger in SITL because of EKF2's correlated error.
- **Waypoints**: course completed in every SITL run (also 1 extra geometry-fix run), but 6.3–7.0 s instead of 5.13 s.
  Ruled out: body drag (quadsim without body drag: 5.13 s), allocation geometry (6.29 s). At up to 7 m/s EKF2's error
  reaches 30 cm, so on truth some waypoints are passed just outside the 0.3 m radius. AIRMODE_PLACEHOLDER
- **Navigation (velocity interface) — largest gap.** With PX4's default velocity controller 2/6 layouts reached the goal
  (quadsim 87 %). Velocity-loop identification (FOPDT, lateral axis): PX4 default L = 0.21 s, τ = 0.52 s; quadsim cascade
  L = 0.14 s, τ = 0.22 s (`MPC_XY_VEL_P_ACC` 1.8 vs simulated 3.0). PX4 configured like the simulated cascade
  (`MPC_XY_VEL_P_ACC=3.0, MPC_XY_VEL_D_ACC=0, MPC_Z_VEL_I_ACC=1.0, MPC_TILTMAX_AIR=34.4`): τ = 0.25 s, **4/6 goals**;
  residual delay 0.21 vs 0.14 s (PX4 attitude/rate chain). One flight per layout → indicative only. Obstacles are virtual
  in SITL (`scripts/sitl/virtual_scan.py`, ring identical to the training caster, unit-tested; collisions from ground
  truth with the training radius); no contact physics.

### Sim-to-real training distribution v1 and multi-seed hover
`Quadsim-{Hover,Track,Waypoint}-v1` (`quadsim.tasks.apply_sim2real_v1`): EKF-like estimate errors (per-flight bias,
Gauss–Markov position/velocity errors; levels from SITL EKF2, widened), action and estimator delay 0–20 ms, noise scale
×[0.5, 2], rate-PID gain ×[0.6, 1.4], thrust-map gain around hover ×[0.8, 1.7]. All new terms are off in the v0 tasks
(v0 evaluations bit-identical, unit test for the unchanged random stream).

HOVER_V1_PLACEHOLDER

### Defects fixed in this session
`evaluate.py` did not record `num_envs` (made the tracking/waypoint re-runs look non-reproducible); NavEnv ring docstring
(true vs estimated pose); SITL harness: rclpy abort at exit (skipped ulog copy), `ls | head` SIGPIPE under pipefail, nav
analysis window used a different clock origin than the trajectories; `load_policy` could not evaluate a policy in an env
with a different critic input (now actor-only fallback); new step-log column order (q/velocity swapped) found by the
analysis and fixed.

## Upstream issue found and patched (affects GUI / video)
- Isaac Lab v2.3.2 `WrenchComposer.__init__` calls `hasattr(asset.data, "body_link_pos_w")`, which evaluates the
  property → PhysX transform query before GPU buffers exist when a viewport renders → every RigidObject fails
  ("Failed to get rigid body transforms from backend"); reproduced with a plain Isaac Lab cuboid.
  Fix: `quadsim/isaac/patches.py` (version-gated, class-level attribute check), applied on `import quadsim.tasks`.
  Viewer enables the chase camera only after the first reset (Isaac Lab's controller reads poses at start-up too).
- Headless MP4 ✅ after the patch (`docs/figures/m3_video_track_frames.png`). GUI: the patched run passed asset
  initialization but then ran out of VRAM because a 4096-env training shared the 8 GB GPU → re-test on an idle GPU.

## Performance engineering
- Analytic depth for vision RL: Warp kernel (`core/sensors/raycast_warp.py`), O(N·R) memory; 2048 envs × 3072
  rays × 24 cylinders = 1.36 ms/frame, 169 MB; matches the PyTorch reference (unit test on CPU and CUDA).

## Open issues / next steps
- Not done: hardware flights; real-vehicle system identification; depth/RGB policies in SITL (needs a rendered gz camera
  matched to the training camera); retraining the navigation policy with a randomized / PX4-identified velocity loop;
  tracking/waypoint policies retrained on v1 (task configs exist: `Quadsim-{Track,Waypoint}-v1`); multi-seed statistics
  beyond hover; tuned baselines; RGB policy training with texture/lighting randomization (see `docs/ROADMAP.md`).
- Open sim-to-sim gaps: waypoint course 25–35 % slower in SITL (cause not isolated); ~70 ms extra delay of PX4's velocity
  and attitude chain; SITL obstacles are virtual (no contact physics).
- GUI and a 4096-env training run cannot share the 8 GB GPU.
- Files created by the container in bind mounts are owned by root.
