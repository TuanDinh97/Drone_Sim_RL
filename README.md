# Drone_Sim_RL — quadsim, a drone reinforcement-learning research framework on Isaac Lab

`quadsim` is a modular, configuration-driven framework for developing, training, testing and comparing drone
reinforcement-learning methods on **NVIDIA Isaac Lab v2.3.2 / Isaac Sim 5.1.0**, built around *verified physics*
and *Sim-to-Real transfer*. Researchers define their own vehicles, sensors, control interfaces, tasks, rewards,
constraints and learning algorithms through registries, configuration terms and plug-in modules, without
touching the simulation core, and evaluate them reproducibly against classical baselines, in- and out-of-distribution.

> Status and every number quoted here come from executed tests — see [`docs/PROGRESS.md`](docs/PROGRESS.md).
> Architecture: [`docs/FRAMEWORK.md`](docs/FRAMEWORK.md) · Tutorials: [`docs/TUTORIALS.md`](docs/TUTORIALS.md) ·
> Troubleshooting: [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) · Sim-to-real: [`docs/SIM2REAL.md`](docs/SIM2REAL.md) ·
> Report: [`report/report.pdf`](report/report.pdf). Hardware used: Ubuntu 24.04, RTX 5060 **8 GB**, i5-14600KF.
> Nothing has been flown on a real vehicle yet.

## What the framework provides

| Area | Provided | Extend with |
|---|---|---|
| Vehicles | X500 (= PX4 SITL gz x500), Crazyflie 2.x, 5-inch racer, X-hexarotor; motor dynamics, rotor drag, ground effect, gyroscopics, wind | YAML/JSON vehicle files (any rotor count), `VEHICLES.register`, Isaac Lab `MultirotorCfg` conversion |
| Control interfaces | motor-level, collective thrust + body rates, attitude, velocity, position (PX4-like onboard loops at 200 Hz) | `ACTION_INTERFACES.register` (example: acceleration commands) |
| Sensors | IMU, barometer, state estimate (EKF2-calibrated profile), depth (ray-cast / RTX), RGB / RGB-D, range ring, 3-D LiDAR term, rangefinder, all with noise / bias / latency / rate | observation terms with `noise_std`, `period_steps`, `delay_steps`, `history` |
| Tasks | hover, tracking, waypoints, obstacle navigation (range / depth / RGB / RGB-D), **drone racing**, **multi-agent swap**, **language-conditioned navigation**, constrained waypoint racing | subclass `QuadrotorEnv` (example: precision landing) or add reward / termination / cost / event / curriculum terms |
| Algorithms | PPO (rsl_rl, asymmetric, CNN), **SAC / TD3** (GPU-native, asymmetric), **PPO-Lagrangian** (any algorithm), **BC / DAgger**, **VLA adapters** | `ALGORITHMS.register` (example: Augmented Random Search) |
| Evaluation | identical seeds and conditions for policies and classical baselines, 95 % CIs, **benchmark suites with in-distribution / OOD splits** and generalization gap | `BENCHMARKS.register(Suite(...))` |
| Data | synchronized HDF5 recording, **episode-structured demonstrations with instructions**, **LeRobot v3.0 export (verified with LeRobot's loader)**, dataset inspection | — |
| Deployment | ONNX/TorchScript export + interface contract, NumPy runtime, ROS 2 PX4 offboard bridge, PX4 v1.16 SITL pipeline | optional; nothing in training depends on PX4 / ROS 2 |
| Resources | GPU-memory-aware `num_envs`, Docker image, 124 CPU unit tests, regression and example checks | — |

## Verified framework results (simulation, RTX 5060 8 GB)

| What | Result | Evidence |
|---|---|---|
| Core refactor (interfaces as classes, vehicle registry, terms) | **59/59** reference metrics bit-identical (PPO + cascade + SE(3) + potential field), after the refactor and with the final code | `scripts/validation/regression_evals.sh`, `docs/results/framework/regression.json` |
| Hover benchmark, nominal / mean of 4 OOD conditions (success) | PPO 99.9 / 87.6 %, **SAC 99.9 / 91.3 %**, BC-distilled 99.9 / 91.7 %, cascade 89.8 / 79.2 % | `docs/results/framework/bench_hover_algos.json` |
| Off-policy SAC / TD3 (new, asymmetric critics) vs the earlier skrl SAC on hover | **99.9 % / 99.6 %** vs 0 % success (TD3's first configuration: 0 %) | `docs/results/framework/offpolicy_hover.json` |
| Hexarotor from the generic frame generator, PPO (CTBR / 6-D motor actions) vs cascade | **99.8 % / 100 %** (1.2 / 1.4 cm) vs 96.5 % | `docs/results/framework/vehicles_interfaces.json` |
| Safe RL: waypoint racing with speed ≤ 2 m/s, budget 0.1 s/episode | PPO-Lagrangian **0.082 s** over-speed, 87.7 % success; unconstrained PPO 5.31 s, 99.4 % | `docs/results/framework/safe_rl.json` |
| Drone racing (2 laps, 7 gates, 5-inch racer), unseen random circuits | PPO **95.2 %**, 2.44 s per lap (11.8 m/s); fixed ellipse 99.6 %; **figure-8 0 %** (trained on one turning direction) → **54 %** when trained on both directions; gate-aligned cascade 0.7 % | `docs/results/framework/bench_race.json` |
| Multi-agent swap (4 drones, shared policy, centralized critic) | PPO 87.5 % group success, 0.4 % collisions, 2.0 s; tuned potential field 100 %, 6.9 s; straight lines 18 %, 45 % collisions | `docs/results/framework/bench_swarm.json` |
| RL → BC distillation (hover) | BC student 99.9 %, 1.4 cm (teacher 99.9 %, 1.4 cm) | `docs/results/framework/imitation.json` |
| Language-conditioned navigation (map + instruction), BC → DAgger | 81 % → **100 %** (expert 100 %); unseen phrasings 60 % with a bag-of-words encoder, **99.8 %** with a frozen pretrained CLIP text tower; without the instruction 12 % | same |
| Behaviour cloning with the last action as input | 15.8 % (copycat failure) vs 81 % without it | same |
| Vision-language BC (RGB camera + instruction, no map) → 1 DAgger round | 82 % → **100 %** (train and unseen phrasings) | same |
| LeRobot v3.0 export, verified by reloading with lerobot 0.4.4 | map dataset 200 episodes / 26 722 frames, RGB dataset 300 episodes / 42 199 frames: state, action, task and images exact | `docs/results/framework/imitation.json` |
| Extension examples (vehicle YAML / plugin, GNSS and 3-D LiDAR sensors, landing task, reward + curriculum, acceleration interface, ARS algorithm, VLA adapter) | **11/11** execute; trained plug-ins: acceleration interface 99.8 %, landing 94.1 % (first reward 0 %, fixed), ARS 46.5 % | `docs/results/framework/examples_check.json`, `vehicles_interfaces.json` |

## Verified platform results (simulation and PX4 SITL)

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
| Deploy pipeline in the loop for hover / tracking / waypoint / range navigation | obs error ≤ **7.5e-6**, success as PyTorch | `scripts/validation/m6_deploy_consistency.py` |
| **PX4 v1.16 SITL (gz x500)**: hover policy as offboard controller | 10/10 full runs; hover **0.8–1.0 cm** RMS w.r.t. the estimate (quadsim 1.9 cm) | `docs/results/sitl_comparison.json` |
| PX4 SITL: tracking (circle / figure-8), estimate-referenced RMSE | **2.2 / 3.4 cm** (quadsim 2.5 / 3.0 cm) | `docs/results/sitl_tasks.json` |
| PX4 SITL: waypoint course | completed in all runs, 6.7 s (quadsim 5.1 s) | same |
| PX4 SITL: range navigation (velocity interface), default / matched PX4 velocity loop | **2/6 / 4/6** goals (quadsim 87 %) | same |
| Hover, 3 seeds each, training distributions v0 / v1 / v2: SITL heave limit cycle (linear thrust map) | 1/6 / 4/6 / **0/6** runs | `docs/results/hover_v1.json` |

SITL also found four bridge bugs (incl. a missing failsafe path), a heave limit cycle caused by the linear thrust map
(fixed by an exact thrust-curve map), EKF2's correlated estimate error, PX4's slower rate and velocity loops and a SITL
GNSS-delay mis-configuration; all are quantified in `docs/PROGRESS.md` and the report.

Negative / neutral results are reported too: domain randomization gave lower errors but no significant
success gain on hover under harsh conditions; the off-policy SAC baseline learns but was not competitive within
the compute budget; the first navigation reward produced a fast but collision-prone policy (53 % success); the first
SE(3) baseline design was unstable at 50 Hz; the depth policy (61–67 %) remains well below the range policy
(see `docs/PROGRESS.md`). RGB/RGB-D pipelines execute end to end but no RGB policy performance is claimed.


## Platform features (physics, sensing, original tasks)

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
| `Quadsim-{Hover,Track,Waypoint}-v1` | same tasks, sim-to-real distribution identified in PX4 SITL (EKF-like estimate errors, 0–20 ms latency, rate-loop and thrust-map gain randomization) | CTBR | as v0 |
| `Quadsim-Hover-v2` | v1 + v0-level white estimate noise + collective-rate penalty (no SITL heave limit cycle, see PROGRESS) | CTBR | as v0 |
| `Quadsim-Hover-Hexa-v0` / `-Hexa-Motor-v0` | hover with a 6-rotor vehicle (generated frame); 6-D motor actions | CTBR / motor | state |
| `Quadsim-Hover-Sensors-v0` | hover + rangefinder (25 Hz, 20 ms) + barometer terms + energy reward (configuration only) | CTBR | state + sensors |
| `Quadsim-Race-v0` (+ `-Ellipse`, `-Figure8`) | 2 laps through 7 gates, random circuits (fixed circuits for OOD tests), 5-inch racer | CTBR | state + next two gates |
| `Quadsim-Swarm-v0` | 4-agent antipodal swap, shared policy, centralized critic | velocity | state + goal + neighbours |
| `Quadsim-Waypoint-SpeedCost-v0` / `-Safe-v0` | waypoint racing with a speed constraint (cost only / PPO-Lagrangian) | CTBR | as waypoint |
| `Quadsim-LangNav-v0` (+ `-Heldout`, `-RGB`, `-RGB-Heldout`) | fly to the pillar named in a natural-language instruction | velocity | state + instruction tokens + landmark map or RGB |

**Data & tooling**: synchronized HDF5 recording (truth, estimates, IMU, baro, actions, rotor speeds, wind,
images), plotting, GUI viewer / MP4 recording with target markers, throughput benchmark.

**Sim-to-Real**: ONNX/TorchScript export with a machine-readable interface contract, NumPy deployment
interface (ENU/FLU ↔ NED/FRD, PX4 `VehicleRatesSetpoint` / `TrajectorySetpoint`), ROS 2 PX4 offboard bridge
(`ros2/quadsim_px4_bridge`), flown with the hover policy against PX4 v1.16 SITL (gz x500); reproducible SITL
pipeline and sim-vs-SITL comparison in `ros2/px4_sitl/` and `scripts/sitl/` (see `docs/SIM2REAL.md`).

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
python -m pytest tests/core -q                                         # 124 unit tests (~15 s, no simulator)
python scripts/rl/train.py --headless --task Quadsim-Hover-v0         # PPO, ~8 min on an RTX 5060
python scripts/rl/train.py --headless --task Quadsim-Hover-v0 --algo sac --num_envs 1024 --max_iterations 15000
python scripts/rl/evaluate.py --headless --task Quadsim-Hover-v0 --controllers policy cascade \
       --checkpoint logs/rsl_rl/quadsim_hover/<run> --episodes 1000
python scripts/benchmark/run.py --suite hover --policy mine=logs/rsl_rl/quadsim_hover/<run>   # ID / OOD table
python scripts/rl/export.py --headless --task Quadsim-Hover-v0 --checkpoint logs/rsl_rl/quadsim_hover/<run>
python scripts/tools/view.py --task Quadsim-Hover-v0 --checkpoint logs/rsl_rl/quadsim_hover/<run>   # GUI / --video
```
Config overrides: append `env.<field>=<value>` / `agent.<field>=<value>`, e.g.
`env.wind.enabled=True env.dynamics.preset=examples/vehicles/hexa_550.yaml env.control.mode=position`.
Your own components: `--plugins my_module.py` on every script (see `examples/` and `docs/TUTORIALS.md`).

Imitation learning / VLA data (details in `docs/TUTORIALS.md` §9–10):
```bash
python scripts/il/collect_demos.py --headless --task Quadsim-LangNav-v0 --expert langnav_expert --episodes 2000 \
       --no_images --out datasets/il/langnav_sym_expert.h5
python scripts/il/train_bc.py --data datasets/il/langnav_sym_expert.h5 --name ln_bc --drop last_action
python scripts/rl/evaluate.py --headless --task Quadsim-LangNav-v0 --controllers policy --algo bc \
       --checkpoint logs/il/ln_bc/model.pt
bash scripts/il/setup_lerobot_env.sh && third_party/lerobot-venv/bin/python scripts/il/export_lerobot.py \
       datasets/il/langnav_sym_expert.h5 --repo_id quadsim/langnav --out datasets/lerobot/langnav
```
PX4 SITL (optional; native ROS 2 Jazzy + Gazebo Harmonic, see `docs/SIM2REAL.md`):
```bash
bash ros2/px4_sitl/setup_px4_sitl.sh
POLICY_DIR=logs/rsl_rl/quadsim_hover/<run>/exported bash scripts/sitl/run_sitl.sh logs/sitl/hover.npz \
       --thrust_curve 0.0225 0.255 0.7225
```

## Repository layout
```
source/quadsim/quadsim/
  registry.py    extension registries (vehicles, action interfaces, algorithms, baselines, benchmarks), plugins
  core/          pure PyTorch: params + vehicles (YAML, generic frames), motor/aero model, reference integrators,
                 controllers/ (cascade, SE(3), mixer, action interfaces), sensors/, disturbances
  isaac/         procedural USD spawner (any rotor count), version-gated upstream patches
  tasks/         QuadrotorEnv + configurable terms (base/), mdp.py term library, baselines, tasks: hover, tracking,
                 waypoint, navigation, racing, swarm (multi-agent), langnav (language), PPO configs
  rl/            algorithms (PPO, SAC/TD3), off-policy learner, networks, policy I/O, CLI overrides
  il/            demonstrations (HDF5), language encoders, BC / DAgger (FiLM CNN), VLA adapter
  benchmark/     benchmark suites (in-distribution / OOD)
  data/          synchronized HDF5 recorder
  deploy/        export + contract, NumPy runtime, PX4 offboard core, closed-loop metrics, virtual sensors
  utils/         hardware detection, resource-aware defaults
examples/        runnable extension examples + vehicle files
scripts/         rl/ (train, evaluate, export), benchmark/, il/ (collect, BC, export to LeRobot, inspect),
                 validation/ (milestone, regression and example checks), data/, tools/, sitl/
tests/core/      unit tests (no Isaac Sim needed)
ros2/            PX4 offboard bridge + reproducible PX4 SITL setup (optional)
docker/          Dockerfile + compose
docs/            FRAMEWORK, TUTORIALS, TROUBLESHOOTING, PROGRESS (verified results), ROADMAP, SIM2REAL, results
report/          LaTeX technical report (source + PDF)
```

## Conventions
World z-up (ENU-like), body FLU, quaternions wxyz (`q_WB`), SI units, PX4 Quad-X rotor numbering.
Policy observations contain only deployable quantities; privileged information is restricted to the critic.

## Known limitations
See `docs/PROGRESS.md` (open issues) and the report's limitations section. Most important: no hardware flights (PX4
SITL only); real-vehicle parameters not identified (presets other than X500/Crazyflie are engineering approximations);
most new-task results are single-seed; RGB policies lack texture/lighting randomization and the language encoder is a
bag-of-words model unless a pretrained text encoder is used (unseen phrasings 60 % vs 99.8 % with CLIP); no pretrained VLA was run (the adapter interface and LeRobot
export are validated, the models are not); off-policy policies leave a large final yaw error on hover; autonomous exploration and
adaptive-control tasks are not implemented (observation histories are available as terms).

## License
BSD-3-Clause (same as Isaac Lab). Isaac Sim is subject to the NVIDIA Omniverse license.
