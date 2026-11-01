# Drone_Sim_RL — a Quadrotor Sim-to-Real RL research platform on Isaac Lab

`quadsim` is a GPU-parallel quadrotor simulation and reinforcement-learning platform built on
**NVIDIA Isaac Lab v2.3.2 / Isaac Sim 5.1.0**, designed around *verified physics* and *Sim-to-Real
transfer*: every physical model is validated against analytic solutions or an independent integrator,
policies observe only what a real vehicle can measure, and the exported policy is checked to reproduce
the simulator bit-for-bit (to 1e-6) through a framework-free deployment interface.

> Status and every number quoted here come from executed tests — see [`docs/PROGRESS.md`](docs/PROGRESS.md).
> Hardware used: Ubuntu 24.04, RTX 5060 **8 GB**, i5-14600KF. Nothing has been flown on a real vehicle yet
> (see [`docs/SIM2REAL.md`](docs/SIM2REAL.md)). The full technical report is [`report/report.pdf`](report/report.pdf).

## Verified results (simulation, RTX 5060 8 GB)

| What | Result | Evidence |
|---|---|---|
| PhysX vs independent reference integrator, 1.5 s open-loop flight (X500) | **0.4 mm** position error | `scripts/validation/m2_dynamics_validation.py` |
| Thrust step / yaw torque vs model | 0.02 % / 0.004 % error | same |
| RTX-rendered depth vs analytic GPU ray casting | **7 mm** median, 99.6 % hit agreement | `scripts/validation/m3_depth_consistency.py` |
| Exported ONNX policy + NumPy deploy interface in closed loop | obs/action error **1e-6**, identical success | `scripts/validation/m6_deploy_consistency.py` |
| Hover/recovery (57° tilt, 3 rad/s), PPO vs cascade PID | **99.9 %** vs 88.7 % success; 1.4 vs 6.1 cm | `docs/results/eval_hover_nominal.json` |
| Same under unseen wind, gusts and pushes | **97.3 %** vs 78.6 % | `docs/results/eval_hover_wind_push.json` |
| Trajectory tracking ≤4 m/s, PPO vs cascade vs SE(3) | **12.6** vs 46.6 vs 49.2 cm RMSE | `docs/results/eval_track_nominal.json` |
| Waypoint course (5 waypoints), PPO vs cascade | **99.4 %** in 6.6 s vs 92.8 % in 10.4 s | `docs/results/eval_waypoint_nominal.json` |
| Training throughput (hover, 4096 envs) | ~190 k env-steps/s, 3.9 GB VRAM | training logs |
| Obstacle navigation, 360° range ring (PPO, reward v2) vs potential field | **83.0 %** vs 39–50 % success (17 % vs 5 % collisions) | `docs/results/eval_nav_*.json` |
| Depth-camera navigation (CNN on 64×48 analytic depth) → zero-shot on RTX-rendered depth | 61.1 % → **66.7 %** (no degradation) | same |
| Env throughput, hover / analytic depth / rendered depth | 439 k @ 8192 envs / 83 k @ 2048 / 6.8 k @ 256 env-steps/s | `docs/results/benchmark.jsonl` |

Negative / neutral results are reported too: domain randomization gave lower errors but no significant
success gain on hover under harsh conditions; the off-policy SAC baseline learns but was not competitive within
the compute budget; the first navigation reward produced a fast but collision-prone policy (53 % success); the first
SE(3) baseline design was unstable at 50 Hz; the depth policy (61–67 %) remains well below the range policy
(see `docs/PROGRESS.md`). RGB/RGB-D pipelines execute end to end but no RGB policy performance is claimed.


## Features

**Physics** (pure PyTorch core `quadsim.core`, framework-agnostic, unit-tested)
- Presets: Holybro **X500** (identical to PX4 SITL gz `x500`) and **Crazyflie 2.x** (Förster 2015); any vehicle via `QuadrotorParams`.
- Per-rotor first-order motor dynamics (asymmetric rise/fall τ, exact discretization, ω saturation), thrust
  `k_f ω²`, yaw reaction `k_m T`, spin-up reaction torque, rotor gyroscopic torque, blade-flapping rotor drag
  and rolling moment, quadratic/linear body drag, Cheeseman–Bennett ground effect.
- Wind (mean + Gauss–Markov turbulence + 1-cos gusts), push disturbances, payload.
- PhysX integrates the rigid body + contacts at 200 Hz; the body wrench is recomputed every physics step.
- Classical controllers: PX4-style rate PID / attitude P / position–velocity PI cascade, Lee SE(3), mixer with desaturation.

**Sensors**: IMU (noise density, turn-on bias, bias drift, scale/misalignment, rate, latency), barometer (ISA),
magnetometer, GPS, state-estimate model (mocap / VIO / GPS-INS profiles with latency), depth camera
(RTX-rendered *or* analytic GPU ray casting with a Warp kernel — the two agree to 7 mm median), RGB / RGB-D
(RTX TiledCamera), LiDAR-like range ring, range finder.

**Reinforcement learning**: PPO (rsl_rl) with asymmetric actor-critic (privileged critic), CNN encoder for
image policies, empirical normalization, curriculum (obstacle density), domain randomization (inertial,
actuator, aerodynamic, sensor, latency), resumable training, evaluation with confidence intervals against
classical baselines under identical seeds.

**Tasks** (Gymnasium IDs)

| Task | Description | Action interface | Observation |
|---|---|---|---|
| `Quadsim-Hover-v0` | recover from up to 57°/3 rad/s perturbations and hold a pose | CTBR | state |
| `Quadsim-Hover-Motor-v0` | same, direct rotor thrusts | motor | state |
| `Quadsim-Track-v0` | track random circles / figure-8s / smooth curves (≤4 m/s) | CTBR | state + 6 future ref. points |
| `Quadsim-Waypoint-v0` | fly through 5 random waypoints | CTBR | state + next two waypoints |
| `Quadsim-Nav-Range-v0` | reach a goal through a cylinder forest | velocity | state + 32-ray range ring |
| `Quadsim-Nav-Depth-v0` | same, depth camera (analytic ray cast, fast) | velocity | state + 64×48 depth |
| `Quadsim-Nav-Depth-Render-v0` | same, RTX-rendered depth | velocity | state + depth |
| `Quadsim-Nav-RGB-v0` / `-RGBD-v0` | RTX-rendered RGB / RGB-D | velocity | state + image |

**Data & tooling**: synchronized HDF5 recording (truth, estimates, IMU, baro, actions, rotor speeds, wind,
images), plotting, GUI viewer / MP4 recording with target markers, throughput benchmark.

**Sim-to-Real**: ONNX/TorchScript export with a machine-readable interface contract, NumPy deployment
interface (ENU/FLU ↔ NED/FRD, PX4 `VehicleRatesSetpoint` / `TrajectorySetpoint`), ROS 2 PX4 offboard bridge
(`ros2/quadsim_px4_bridge`, not yet run against PX4).

## Installation

### Native (conda)
```bash
bash scripts/setup_env.sh          # creates conda env "quadsim": Isaac Sim 5.1 (pip), Isaac Lab v2.3.2, quadsim
conda activate quadsim
export OMNI_KIT_ACCEPT_EULA=YES
```
Pins are in `env/constraints.txt` (torch 2.7.0+cu128, warp 1.11.1, rsl-rl-lib 3.1.2, …). Some extras try to
upgrade torch to CUDA-13 builds — always install with `-c env/constraints.txt`.

### Docker (official Isaac Lab 2.3.2 image)
```bash
docker compose -f docker/docker-compose.yaml build
docker compose -f docker/docker-compose.yaml run --rm test      # unit tests + Isaac Sim GPU smoke test
docker compose -f docker/docker-compose.yaml run --rm train "python scripts/rl/train.py --task Quadsim-Hover-v0 --headless"
xhost +local: && docker compose -f docker/docker-compose.yaml run --rm gui
```
`logs/`, `datasets/`, `outputs/` are bind-mounted (persistent); Isaac Sim caches live in named volumes.
If `docker pull nvcr.io/...` returns *403* (NGC blocks some networks; the Docker daemon ignores your shell
proxy), fetch the base image with `python3 scripts/tools/pull_image_via_proxy.py nvcr.io/nvidia/isaac-lab:2.3.2`.

## Quick start
```bash
python -m pytest tests/core -q                                         # 48 unit tests (~15 s)
python scripts/validation/m2_dynamics_validation.py --headless        # PhysX vs reference dynamics
python scripts/rl/train.py --task Quadsim-Hover-v0 --headless         # ~10 min on an RTX 5060
python scripts/rl/evaluate.py --headless --task Quadsim-Hover-v0 --controllers policy cascade \
       --checkpoint logs/rsl_rl/quadsim_hover/<run> --episodes 1000
python scripts/tools/view.py --task Quadsim-Hover-v0 --checkpoint logs/rsl_rl/quadsim_hover/<run>   # GUI
python scripts/rl/export.py --headless --task Quadsim-Hover-v0 --checkpoint logs/rsl_rl/quadsim_hover/<run>
```
More: `scripts/data/record.py` (datasets), `scripts/tools/view.py --headless --enable_cameras --video` (MP4),
`scripts/tools/benchmark.py`, `scripts/rl/train_queue.sh` (detached sequential training).
Config overrides: append `env.<field>=<value>` / `agent.<field>=<value>`, e.g.
`env.wind.enabled=True env.dynamics.preset=crazyflie agent.max_iterations=300`.

## Repository layout
```
source/quadsim/quadsim/
  core/          pure PyTorch physics: params, model (motors/aero/wrench), dynamics (reference integrators),
                 controllers/, sensors/ (IMU, baro, odometry, cameras, ray casting + Warp kernel), disturbances
  isaac/         procedural USD quadrotor spawner, version-gated upstream patches
  tasks/         Isaac Lab DirectRLEnv tasks (base, hover, tracking, waypoint, navigation), PPO configs, baselines
  rl/            CNN actor-critic, policy loading, CLI overrides
  data/          HDF5 recorder
  deploy/        export (ONNX/TorchScript + contract) and NumPy deployment interface
scripts/         validation/ (milestone acceptance tests), rl/ (train, evaluate, export), data/, tools/
tests/core/      unit tests (no Isaac Sim needed)
ros2/            quadsim_px4_bridge ROS 2 package
docker/          Dockerfile + compose
docs/            PROGRESS (verified results), ROADMAP, SIM2REAL, figures, results (JSON)
report/          LaTeX technical report
```

## Conventions
World z-up (ENU-like), body FLU, quaternions wxyz (`q_WB`), SI units, PX4 Quad-X rotor numbering.
Policy observations contain only deployable quantities; privileged information is restricted to the critic.

## Known limitations
See `docs/PROGRESS.md` (open issues) and the report's limitations section. Most important: no hardware or
PX4-SITL flights yet; depth policies trained on analytic depth with a parametric noise model; RGB policies lack
texture/lighting randomization; baselines use default (untuned) gains; single training seed per configuration.

## License
BSD-3-Clause (same as Isaac Lab). Isaac Sim is subject to the NVIDIA Omniverse license.
