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

## Milestone 4 — State-based RL 🟡 (hover ✅, tracking/waypoint training)
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

## Milestone 6 — Sim-to-Real interfaces 🟡
- Export (`scripts/rl/export.py`): TorchScript + ONNX with normalizer and clamp; ONNX vs PyTorch max |Δ| = 3.6e-7 ✅.
  Machine-readable interface contract (`contract.json`, example `docs/results/contract_hover.json`).
- `quadsim.deploy.interface` (NumPy only): ENU/FLU ↔ NED/FRD, PX4 odometry → sim conventions, observation
  builder, action decoder → PX4 `VehicleRatesSetpoint` / `TrajectorySetpoint`. 7 unit tests ✅ (incl. frame
  consistency of a physical body vector through both quaternion conventions).
- Deployment pipeline in the loop (`scripts/validation/m6_deploy_consistency.py`, hover, 512 episodes) ✅:
  ONNX policy fed by the NumPy builder flies the simulator; max obs error 9.5e-7, max action error 1.0e-6,
  success 99.8 % (identical to PyTorch).
- ROS 2 package `ros2/quadsim_px4_bridge` (offboard node, launch, config): 🟡 written and syntax-checked only;
  **not executed against PX4 SITL or hardware**.

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
- Not done: PX4 SITL / hardware flights, real-vehicle system identification, multi-seed statistics, tuned baselines,
  RGB policy training with texture/lighting randomization (see `docs/ROADMAP.md`).
- GUI and a 4096-env training run cannot share the 8 GB GPU.
- Files created by the container in bind mounts are owned by root.
- Files created by the container in bind mounts are owned by root.
