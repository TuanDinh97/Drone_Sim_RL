# Tutorials

All commands run from the repository root with the `quadsim` environment
(`~/miniconda3/envs/quadsim/bin/python`, abbreviated `python`) and `export OMNI_KIT_ACCEPT_EULA=YES`. Add
`--headless` to every simulator script unless you want the GUI. The extension examples used below (`examples/`)
are executed in the simulator by `scripts/validation/examples_check.py` (pass flags in
`logs/validation/examples_check.json`, curated copy `docs/results/framework/examples_check.json`).

## 1. Train, evaluate, export

```bash
python scripts/rl/train.py --headless --task Quadsim-Hover-v0                     # PPO, ~8 min on an RTX 5060
python scripts/rl/evaluate.py --headless --task Quadsim-Hover-v0 --controllers policy cascade \
    --checkpoint logs/rsl_rl/quadsim_hover/<run> --episodes 1000
python scripts/rl/export.py --headless --task Quadsim-Hover-v0 --checkpoint logs/rsl_rl/quadsim_hover/<run>
```

Every run directory holds `params/{env,agent}.yaml`, the command line and git commit. Resume with
`--resume <run dir>`. Override any config field from the command line: `env.wind.enabled=True`,
`env.randomization.mass_scale="(0.8,1.2)"`, `agent.algorithm.learning_rate=3e-4`.

Other algorithms use the same commands: `--algo sac` or `--algo td3` (`logs/<algo>/<experiment>/<run>`);
`evaluate.py` and `export.py` detect the algorithm from the checkpoint.

## 2. A custom drone

YAML (`examples/vehicles/hexa_550.yaml`): mass, a generic frame (`num_rotors`, `arm_length`, `layout`),
thrust-to-weight, motor time constants, drag, `inertia: auto` … then

```bash
python scripts/rl/train.py --headless --task Quadsim-Hover-v0 env.dynamics.preset=examples/vehicles/hexa_550.yaml
```

or register Python factories (`examples/custom_vehicle.py`) and use them by name with `--plugins`. The spawner,
mixer, motor model, hover action and (for `motor` control) the action dimension follow the vehicle automatically.
Physically inconsistent vehicles, from files or from registered factories, are rejected with an explanation
(inertia triangle inequality, thrust-to-weight, spin directions, allocation rank, CoM outside the rotor polygon).

## 3. A custom sensor

Sensors that are not built in are observation terms with a rate, latency and noise model
(`examples/custom_sensor.py`):

```python
self.terms.observations["gnss"] = ObsTermCfg(func=gnss_position, noise_std=0.3, period_steps=10, delay_steps=5)
```

`period_steps` = policy steps between samples (5 Hz here), `delay_steps` = latency (100 ms). Terms are deployable
(policy observation) unless `privileged=True`.

## 4. A custom task

Subclass `QuadrotorEnv` and implement the task hooks (`examples/custom_task.py`, precision landing):
`task_obs_dim/_task_obs`, `_task_privileged_obs`, `_compute_task_reward`, `_task_crash/_task_timeout` (a crash
terminates the episode; a time-out truncates it and the value is bootstrapped - the built-in tasks return reaching
the goal as a time-out unless `env.task_end_is_terminal=True`, see `docs/PROGRESS.md`),
`_sample_initial_state`, `_task_episode_metrics`; optionally `baseline_targets` so the classical cascade baseline
can fly it. Register it with `gym.register` (entry point and config classes can be objects). Train:

```bash
python scripts/rl/train.py --headless --task Quadsim-Example-Land-v0 --plugins examples/custom_task.py
```

## 5. Rewards, terminations, curricula without touching task code

`examples/custom_reward_curriculum.py` adds an energy penalty and an upright bonus, a box termination and a
curriculum that raises the initial tilt whenever at least 256 episodes started at the current level succeed more
than 80 % of the time (`Curriculum/init_tilt` in TensorBoard). Reusable term functions: `quadsim.tasks.mdp`.

## 6. A custom control interface

`examples/custom_action_interface.py` registers an `acceleration` interface (acceleration set-point → thrust vector
→ attitude → rates → mixer):

```bash
python scripts/rl/train.py --headless --task Quadsim-Hover-v0 --plugins examples/custom_action_interface.py \
    env.control.mode=acceleration
```

## 7. A custom learning algorithm

`examples/custom_algorithm.py` registers Augmented Random Search as `--algo ars`: a configclass of
hyper-parameters, `train(env, cfg, log_dir, resume)` and `load_policy(checkpoint, env)`. The training script,
logging layout, overrides, evaluation and benchmarks then work unchanged.

## 8. Benchmarks: training performance vs generalization

```bash
python scripts/benchmark/run.py --list
python scripts/benchmark/run.py --suite hover --policy ppo=logs/rsl_rl/quadsim_hover/<run> \
    --policy sac=logs/sac/quadsim_hover/<run> --tag mine
```

`logs/benchmark_suites/hover/mine/summary.md` holds one table per metric (in-distribution and OOD conditions:
wind + pushes, harsh dynamics, sensor noise, latency) and the generalization gap. Register new suites in a plugin
with `BENCHMARKS.register(name, Suite(...))`.

## 9. Imitation learning: demonstrations → BC → DAgger

```bash
# expert demonstrations (RL checkpoint or any registered baseline), successful episodes only
python scripts/il/collect_demos.py --headless --task Quadsim-LangNav-v0 --expert langnav_expert \
    --episodes 2000 --no_images --out datasets/il/langnav_sym_expert.h5
python scripts/il/train_bc.py --data datasets/il/langnav_sym_expert.h5 --name ln_bc --drop last_action
python scripts/rl/evaluate.py --headless --task Quadsim-LangNav-v0 --controllers policy --algo bc \
    --checkpoint logs/il/ln_bc/model.pt
# DAgger round: the learner flies, the expert labels every visited state
python scripts/il/collect_demos.py --headless --task Quadsim-LangNav-v0 --expert langnav_expert \
    --executor logs/il/ln_bc/model.pt --keep all --episodes 1000 --no_images --out datasets/il/ln_dagger1.h5
python scripts/il/train_bc.py --data datasets/il/langnav_sym_expert.h5 datasets/il/ln_dagger1.h5 \
    --name ln_dagger1 --drop last_action
```

Lessons learned (`docs/PROGRESS.md`): drop the `last_action` block from BC inputs (`--drop last_action`), otherwise
the cloned policy copies its previous action and fails to stop (15.8 % vs 81 % success); for instructions phrased
differently from the training data use a frozen pretrained text encoder
(`--text_encoder stabilityai/sd-turbo:text_encoder`, local cache only): unseen phrasings 60 % → 99.8 %.
RL distillation: `--expert policy --checkpoint <run>` (hover: BC reaches the PPO teacher's 99.9 %).

## 10. Language and vision: VLA-style data and policies

```bash
python scripts/il/collect_demos.py --headless --enable_cameras --task Quadsim-LangNav-RGB-v0 \
    --expert langnav_expert --episodes 2000 --num_envs 64 --out datasets/il/langnav_rgb_expert.h5
python scripts/il/train_bc.py --data datasets/il/langnav_rgb_expert.h5 --name ln_rgb_bc --images --drop last_action
python scripts/rl/evaluate.py --headless --enable_cameras --task Quadsim-LangNav-RGB-v0 --num_envs 64 \
    --controllers policy --algo bc --checkpoint logs/il/ln_rgb_bc/model.pt
```

Export for LeRobot-based VLA training (separate environment, never touches the simulator's packages):

```bash
bash scripts/il/setup_lerobot_env.sh
third_party/lerobot-venv/bin/python scripts/il/export_lerobot.py datasets/il/langnav_rgb_expert.h5 \
    --repo_id quadsim/langnav_rgb --out datasets/lerobot/langnav_rgb
```

Evaluate any external model through the VLA adapter interface (`examples/vla_color_servo.py` is a model-free
reference adapter):

```bash
python scripts/rl/evaluate.py --headless --enable_cameras --task Quadsim-LangNav-RGB-v0 --num_envs 64 \
    --controllers policy --algo vla --checkpoint examples/vla_color_servo.py:make_adapter
```

## 11. Multi-agent

`Quadsim-Swarm-v0`: groups of 4 agents (`env.agents_per_group`) share a frame and resets; one shared policy,
centralized critic. Baselines: `cascade` (straight lines, no coordination) and `swarm_potential`.

```bash
python scripts/rl/train.py --headless --task Quadsim-Swarm-v0
python scripts/rl/evaluate.py --headless --task Quadsim-Swarm-v0 --num_envs 256 \
    --controllers policy cascade swarm_potential --checkpoint logs/rsl_rl/quadsim_swarm/<run>
```

## 12. Safe RL

Add a cost term and enable the Lagrangian (`Quadsim-Waypoint-Safe-v0`: speed ≤ 2 m/s, budget 0.1 s of
over-speed flight per episode). Evaluate unconstrained policies on `Quadsim-Waypoint-SpeedCost-v0` (same cost,
no enforcement) to compare constraint violations.

## 13. Deployment and cross-simulator validation

`scripts/rl/export.py` writes `policy.onnx`, `policy.pt` and `contract.json` (exact observation layout, action
semantics, vehicle, training assumptions); `quadsim.deploy` rebuilds observations without Isaac
(`scripts/validation/m6_deploy_consistency.py`). PX4 SITL (optional, ROS 2 Jazzy + Gazebo):
`ros2/px4_sitl/setup_px4_sitl.sh`, `scripts/sitl/run_sitl.sh` (see `docs/SIM2REAL.md`).
