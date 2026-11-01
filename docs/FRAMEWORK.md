# quadsim framework architecture

`quadsim` is a modular, configuration-driven research framework for drone reinforcement learning on **Isaac Lab
v2.3.2 / Isaac Sim 5.1.0**. This document describes how the framework is organised, which parts are meant to be
replaced, and what each extension point guarantees. Step-by-step recipes are in [`TUTORIALS.md`](TUTORIALS.md);
verified numbers are in [`PROGRESS.md`](PROGRESS.md).

## Design principles

1. **Physics you can check.** Vehicle, motor, aerodynamic, sensor and controller models are plain PyTorch
   (`quadsim.core`, no Isaac import), unit-tested on the CPU and validated against an independent integrator.
   Isaac Lab / PhysX integrates the rigid body and contacts; the model-side wrench is recomputed every physics step.
2. **Deployable vs privileged information.** The `policy` observation group contains only what a real vehicle can
   measure or knows (noisy, delayed state estimate, gyro, last action, task data, configured sensors). Everything
   else (true state, rotor speeds, wind, DR parameters, other agents' true states) goes to the `critic` group.
   Classical baselines read the same onboard estimate and act through the same action interface, delays and
   actuators as the policies, so comparisons are fair.
3. **Configuration over code.** A task is a configclass; the vehicle, action interface, sensors, extra
   observations / rewards / terminations / costs / events / curricula, safety constraints and domain randomization
   are configuration fields. Research code registers new components by name and selects them from the config or
   the command line (`--plugins my_module env.control.mode=my_interface`).
4. **Reproducibility.** Every run stores the exact env/agent configuration, command line and git commit; seeds
   make evaluations bit-identical (`scripts/validation/regression_evals.sh` checks 59 reference metrics after every
   core refactor). Training is resumable; datasets carry their configuration and provenance.
5. **Optional heavy dependencies.** PX4 / ROS 2, LeRobot, pretrained language models and Docker are optional;
   none is imported by the core training loop.

## Layers

```
quadsim.registry        name -> component registries (the extension points), plugin loading
quadsim.core            pure PyTorch: vehicles (params, YAML files, generic frames), motor + aero model,
                        reference integrator, controllers + action interfaces, sensors, disturbances
quadsim.isaac           procedural USD spawner (no Nucleus assets), Isaac Lab patches
quadsim.tasks           QuadrotorEnv (DirectRLEnv) + configurable terms + tasks: hover, track, waypoint,
                        nav (range / depth / RGB / RGB-D), race, swarm (multi-agent), langnav (language);
                        mdp term library; classical baselines
quadsim.rl              algorithms (PPO via rsl_rl, SAC/TD3 GPU-native, BC/VLA evaluation hooks), policy I/O
quadsim.il              demonstrations (HDF5), language encoders, behaviour cloning / DAgger, VLA adapter
quadsim.benchmark       benchmark suites (in-distribution vs OOD conditions)
quadsim.deploy          ONNX/TorchScript export + interface contract, NumPy deploy runtime, PX4 offboard core
quadsim.utils           hardware detection and resource-aware defaults
scripts/                train / evaluate / export / benchmark / collect demos / BC / LeRobot export / SITL
examples/               runnable extension examples (vehicle, sensor, task, reward+curriculum, action
                        interface, algorithm, VLA adapter)
```

Per physics step (200 Hz by default) `QuadrotorEnv` runs: sensor models (IMU, odometry/estimator, barometer) →
delayed policy command → **action interface** on the onboard flight controller (nominal model, onboard estimates)
→ mixer / allocation → motor dynamics → aerodynamic wrench + wind + pushes + payload → PhysX. The policy runs at
50 Hz (decimation 4).

## Extension points

| Registry (`quadsim.registry`) | Selected by | Interface | Built-in |
|---|---|---|---|
| `VEHICLES` | `cfg.dynamics.preset` (name **or** `.yaml`/`.json` path) | `() -> QuadrotorParams` | `x500`, `crazyflie`, `racer5`, `hexa_x` |
| `ACTION_INTERFACES` | `cfg.control.mode` | `ActionInterface`: `action_dim`, `hover_action`, `reset`, `__call__(a, quat, est, gyro) -> rotor thrusts` | `motor`, `ctbr`, `ctatt`, `velocity`, `position` (+ `acceleration` example) |
| `ALGORITHMS` | `train.py --algo`, `evaluate.py --algo`, `export.py --algo` | `Algorithm`: `default_cfg`, `train`, `load_policy`, `actor_module` (export) | `ppo` (rsl_rl), `sac`, `td3`; evaluation only: `bc` (trained by `train_bc.py`), `vla` (+ `ars` example) |
| `BASELINES` | `evaluate.py --controllers` | `ctrl(env)() -> actions` | `cascade`, `geometric`, `potential_field`, `swarm_potential`, `langnav_expert` |
| `BENCHMARKS` | `scripts/benchmark/run.py --suite` | `Suite(task, conditions, metrics, ...)` | `hover`, `track`, `waypoint`, `nav_range`, `race`, `swarm`, `langnav` |
| Gymnasium registry | `--task` | Isaac Lab DirectRLEnv + cfg entry points | 27 built-in `Quadsim-*` tasks (+ 4 in `examples/`) |

Configuration-level extension (no registry needed), `cfg.terms` (`quadsim.tasks.base.terms`), with the Isaac Lab
term convention `func(env, **params)`:

| Term | Effect | Notes |
|---|---|---|
| `ObsTermCfg` | appended to the `policy` (deployable) or `critic` (`privileged=True`) observation | `noise_std`, `period_steps` (sensor rate, sample-and-hold), `delay_steps` (latency), `scale`, `clip`; history reset per episode |
| `RewardTermCfg` | `weight * func * step_dt` added to the task reward, logged per term | |
| `TerminationTermCfg` | failure, or truncation with `time_out=True` | |
| `CostTermCfg` | constraint cost rate; episode cost in seconds of violation | `extras["cost"]`, metric `cost` |
| `EventTermCfg` | `startup` / `reset` / `interval` randomization or perturbation | custom DR, pushes, wind |
| `CurriculumTermCfg` | called before resets with the metrics of the episodes that just finished | `mdp.attribute_curriculum` (judges each level only on episodes started at that level) |

`quadsim.tasks.mdp` is a library of reusable term functions (rangefinder, barometer, accelerometer, estimated
position/velocity, ESC rotor speeds, energy, upright, box/height terminations, speed/tilt/altitude/keep-out costs,
wind/motor/velocity-kick events, attribute curriculum).

### Vehicles

`quadsim.core.vehicles` builds a `QuadrotorParams` from a registered preset, a YAML/JSON file
(`examples/vehicles/hexa_550.yaml`) or a dict: `base:` inheritance, generic symmetric frames
(`frame: {num_rotors, arm_length, layout: x|plus}`, even rotor counts; 4-rotor X frames use the PX4 Quad-X order),
`thrust_to_weight` → `k_f` (give one of the two), `inertia: auto` (box body + point-mass motors), physical validation
of every vehicle, registered factories included (positive mass, inertia triangle inequality, thrust-to-weight > 1.05
at nominal parameters, spins ±1, full-rank allocation, positive zero-torque hover thrust on every rotor, i.e. CoM
inside the rotor polygon). Numbers such as `1e-5` are accepted (YAML would read them as strings); `base:` paths are
relative to the file. Everything downstream adapts
automatically: the USD body (one rotor disc per rotor), PhysX mass and inertia, motor/aero model, mixer
(pseudo-inverse allocation with PX4-style desaturation, any rotor count and asymmetric layouts), hover action, `motor` action dimension
(= number of rotors) and observation sizes. `from_isaaclab_multirotor()` converts an Isaac Lab `isaaclab_contrib`
`MultirotorCfg` (allocation matrix + thruster model) into quadsim parameters.

Why not Isaac Lab's `Multirotor` articulation (isaaclab_contrib, added in v2.3)? It loads robots from Nucleus USDs
(unreachable from the development network), models thrusters with a first-order rotor-speed lag but without rotor
drag, ground effect or gyroscopic terms, and simulates an articulation per vehicle. quadsim keeps a single rigid body per
vehicle with the full model-side wrench (validated against a reference integrator to 0.4 mm over 1.5 s) and offers
the conversion above for interoperability.

### Action interfaces

All interfaces run on the `FlightController` (nominal-model PX4-like rate PID, attitude P, position/velocity PI,
mixer), at the physics rate, on the delayed action and the onboard state estimate:

| `control.mode` | action (normalized to [-1, 1]) | onboard loops |
|---|---|---|
| `motor` | per-rotor thrust | none |
| `ctbr` | collective thrust + body rates | rate PID |
| `ctatt` | collective thrust + roll, pitch + yaw rate | attitude + rate |
| `velocity` | velocity in the yaw-aligned frame + yaw rate | velocity PI + attitude + rate |
| `position` | position set-point in a configurable box + yaw rate | full cascade |

The rate-loop gain and thrust-map gain can be randomized per episode (`randomization.rate_gain_scale`,
`thrust_gain_scale`), which the PX4 SITL study showed to matter for transfer.

### Sensors

Built in: IMU (noise density, turn-on bias, bias random walk, scale/misalignment, rate, latency), barometer,
state estimate (white + bias + Gauss-Markov errors, latency; EKF2-calibrated profile in `apply_sim2real_v1`),
depth camera (analytic GPU ray casting or RTX TiledCamera), RGB / RGB-D (TiledCamera with augmentation), 2-D range
ring. Any further sensor is an `ObsTermCfg` (`examples/custom_sensor.py`: 5 Hz GNSS with 100 ms latency;
`mdp.lidar_scan`: 3-D LiDAR ray-casting the task's obstacle primitives; `history=H` stacks the last H samples, e.g.
action / gyro histories for implicit system identification).

### Domain randomization

`DomainRandCfg`: mass / inertia / CoM (startup, written to PhysX), per-episode thrust and torque coefficients per
rotor, motor time constants, maximum rotor speed (battery sag), drag, payload, sensor-noise scale, rate-loop gain,
thrust-map gain; `WindCfg` / `PushCfg` disturbances; action and estimate latency ranges; plus arbitrary
`EventTermCfg` randomizations. The critic observes a normalized summary of the sampled parameters.

## Tasks

| Task | Purpose | Interface | Policy observation |
|---|---|---|---|
| `Quadsim-Hover-v0/v1/v2` (+ `-Motor`, `-Hexa`, `-Hexa-Motor`, `-Sensors`) | stabilization / recovery | ctbr / motor | state + target |
| `Quadsim-Track-v0/v1` | trajectory tracking | ctbr | state + reference preview |
| `Quadsim-Waypoint-v0/v1`, `-SpeedCost-v0`, `-Safe-v0` | waypoint racing, constrained RL | ctbr | state + next waypoints |
| `Quadsim-Nav-{Range,Depth,Depth-Render,RGB,RGBD}-v0` | obstacle avoidance | velocity | state + range ring / image |
| `Quadsim-Race-v0/v1` (+ `-Ellipse`, `-Figure8`) | drone racing through gates, multi-lap (v1: both turning directions) | ctbr (racer5) | state + next two gates |
| `Quadsim-Swarm-v0` | cooperative multi-agent position swap | velocity | state + goal + neighbours |
| `Quadsim-LangNav-v0` (+ `-Heldout`, `-RGB`, `-RGB-Heldout`) | language-conditioned navigation (IL / VLA) | velocity | state + instruction tokens (+ landmark map or RGB) |

### Multi-agent design

`SwarmEnv` treats each agent as one row of the vectorized environment and groups `agents_per_group` consecutive
rows into a team that shares a local frame (`env_origins`), episode clock and resets. Consequences: any single-agent
algorithm trains a **parameter-shared, decentralized** policy; the critic observation is **centralized** (true
states and goals of the whole group, MAPPO-style); inter-agent collisions are analytic (centre distance) so they do
not depend on cross-environment PhysX contacts; a crash terminates the crashed agent and truncates its teammates.
Neighbour information in the policy observation comes from the other agents' *estimates* (radio broadcast model).

### Safe / constrained RL

Cost terms produce per-step cost rates; `SafetyCfg(lagrangian=True, cost_limit=...)` turns any algorithm into its
Lagrangian variant: the environment returns `(r - λ c) / (1 + λ)` and updates `λ ≥ 0` with a PID rule on the mean
episode cost of finished episodes (Stooke et al. 2020). The raw reward and cost remain in `extras`, the multiplier
is logged as `Safety/lambda`. (This is the reward-penalty form without a separate cost critic.)

## Learning algorithms

* **PPO** (rsl_rl 3.1.2): asymmetric actor-critic, empirical normalization, CNN encoder for the `image` group.
* **SAC / TD3** (`quadsim.rl.offpolicy`): GPU-native, asymmetric twin critics with LayerNorm, running observation
  normalization, large batches and several updates per environment step, per-environment exploration noise (TD3),
  automatic temperature (SAC), time-out bootstrapping, reward scaling to per-second units. Checkpoints contain
  networks, optimizers and normalizers. They use the state observation groups only (an `image` group is ignored)
  and log to TensorBoard (`--logger wandb` applies to PPO). (skrl 1.4.3 is still installed; `scripts/rl/train_sac.py` is the earlier
  skrl integration, which has no asymmetric critic.)
* **BC / DAgger** (`quadsim.il.bc`): state / image / language inputs split by the dataset's observation layout;
  FiLM-conditioned CNN for language-conditioned vision; DAgger by alternating `collect_demos.py --executor` and
  `train_bc.py` on the aggregated files. Instructions are encoded either by a small bag-of-words model trained with
  the policy or by a **frozen pretrained text tower** (`train_bc.py --text_encoder hf_repo:subfolder`, loaded from
  the local Hugging Face cache only; e.g. the CLIP ViT-H/14 text encoder of `stabilityai/sd-turbo`), whose
  embeddings are cached in the checkpoint and computed on the fly for unseen instructions at evaluation time.
* **VLA adapters** (`quadsim.il.vla`): any external model wrapped as `(images, instructions, proprio) -> actions`
  is evaluated with `evaluate.py --algo vla --checkpoint module:factory`.
* **New algorithms**: subclass `Algorithm` and register it (`examples/custom_algorithm.py`: Augmented Random Search
  on thousands of parallel environments).

## Evaluation and benchmarking

`scripts/rl/evaluate.py` runs complete episodes for any mix of controllers under identical seeds and reports
mean / std / 95 % CI per metric (plus a Wilson interval for rates); the metrics are defined by each task
(`_task_episode_metrics`) plus `crash`, `time_out`, `episode_length_s` and `cost`. Every environment contributes
its first `ceil(episodes / num_envs)` episodes: stopping at a global episode count instead would drop the
episodes still running at that moment (the long ones) and bias rates toward outcomes that end episodes early. `scripts/benchmark/run.py` runs a suite: each condition in a fresh
simulator process, every controller with the same seed, a table per metric, and the generalization gap
(id − mean ood) of the primary metric. Training performance (`id`) and generalization (`ood`: perturbed
dynamics, wind, sensor noise, latency, unseen layouts or phrasings) are always reported separately.

## Imitation learning and VLA data

`scripts/il/collect_demos.py` records episode-structured HDF5 demos from any expert (RL checkpoint or registered
baseline): policy and critic observations, images, actions, rewards, ground truth, estimate, instruction text,
success and per-episode metrics, plus the configuration, vehicle, observation layout and tokenizer.
`scripts/il/export_lerobot.py` converts them into a **LeRobot v3.0** dataset (lerobot 0.4.4 in an isolated
environment, `scripts/il/setup_lerobot_env.sh`) and verifies the export by reloading it with `LeRobotDataset`.

## Resources

`quadsim.utils.hardware` reads the GPU's free memory and lowers a task's default `num_envs` when it would not fit
(model calibrated on the RTX 5060 measurements; only lowers unless asked). Observed on the 8 GB RTX 5060: state
tasks 4096 envs ~3-4 GB; rendered cameras ~4-5 GB for 64-128 envs; **do not run more than two GPU simulations at a
time** (PhysX fails to allocate its scene otherwise).

## Relation to other frameworks

| framework | simulator | what quadsim adds or does differently |
|---|---|---|
| Isaac Lab `direct/quadcopter` (Crazyflie example) | Isaac Lab | full motor/aero model, onboard controllers and action interfaces, sensors with latency, DR, baselines, deployment |
| Isaac Lab `drone_arl` + `isaaclab_contrib` Multirotor (v2.3) | Isaac Lab (manager-based) | single rigid body + validated model-side wrench, procedural assets, conversion from `MultirotorCfg` |
| Aerial Gym Simulator (NTNU ARL) | Isaac Gym | runs on the maintained Isaac Lab / Isaac Sim 5 stack; asymmetric deployable/privileged observation contract, PX4 SITL validation, IL/VLA tooling |
| OmniDrones | Isaac Sim | single-machine 8 GB-GPU workflow, sim-to-real tooling, IL/VLA tooling; OmniDrones has a broader multi-agent task set |
| gym-pybullet-drones | PyBullet (CPU) | GPU-parallel training at 10^5 env-steps/s, rendered and ray-cast cameras |

This is a qualitative positioning from the projects' public descriptions and code, not a measured comparison.
