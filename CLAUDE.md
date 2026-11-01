# CLAUDE.md — working notes for this repository

Quadrotor Sim-to-Real RL research platform on **Isaac Lab v2.3.2 + Isaac Sim 5.1.0**.
Read `docs/PROGRESS.md` (verified state, open issues) and `docs/ROADMAP.md` (next steps) before working.

## Current state (2026-11-01)
All seven milestones meet their acceptance criteria (details and caveats in docs/PROGRESS.md). Trained runs live in
`logs/rsl_rl/` (git-ignored): hover `*_ctbr_s42`, tracking `*_track_ctbr_s42`, waypoint `*_waypoint_ctbr_s42`,
nav range `*_nav_range_v2_s42`, nav depth `*_nav_depth_v2_s42` (model_900). The hover policy flies PX4 v1.16.2
SITL (gz x500) through `ros2/quadsim_px4_bridge` (docs/PROGRESS.md "PX4 SITL cross-check"; PX4 + ROS 2 workspace in
`third_party/`, built by `ros2/px4_sitl/setup_px4_sitl.sh`, runs via `scripts/sitl/run_sitl.sh`).
Session 3 (same day): tracking, waypoint and range-navigation policies exported and flown in SITL
(`scripts/sitl/{sitl_scenario,quadsim_task_scenario,compare_tasks}.py`, results `docs/results/sitl_*.json`); hover
policies trained with 3 seeds each on v0, v1 and v2 distributions (`Quadsim-Hover-v{1,2}`, runs `*_v1_s*`, `*_v2_s*`,
`*_ctbr_s{1,2}`), summary `docs/results/hover_v1.json`. Report section `report/sitl.tex`.
Session 4 (framework): quadsim turned into a general drone-RL research framework - registries + plugins
(`quadsim.registry`: VEHICLES, ACTION_INTERFACES, ALGORITHMS, BASELINES, BENCHMARKS; `--plugins` on every script),
YAML/generic-frame vehicles (`quadsim.core.vehicles`), action interfaces as classes (`core/controllers/interfaces.py`),
configurable terms + safe RL (`tasks/base/terms.py`, `tasks/mdp.py`), GPU-native SAC/TD3 (`rl/offpolicy.py`),
new tasks (race, swarm multi-agent, langnav language-conditioned), IL/VLA (`quadsim.il`, `scripts/il/`, LeRobot v3
export in `third_party/lerobot-venv`), benchmark suites (`quadsim.benchmark`, `scripts/benchmark/run.py`),
hardware-aware num_envs, `examples/`. Architecture: docs/FRAMEWORK.md; recipes: docs/TUTORIALS.md.
Core refactors are guarded by `scripts/validation/regression_evals.sh` + `compare_evals.py` (59 metrics bit-identical).
Next steps: docs/ROADMAP.md ("Beyond the current scope") and the open issues at the end of docs/PROGRESS.md.

## Environment (verified)
- Machine: Ubuntu 24.04, **RTX 5060 8 GB** (not the 12 GB RTX 5070 of the original brief), i5-14600KF, 32 GB RAM.
- Conda env **`quadsim`** (Python 3.11): `~/miniconda3/envs/quadsim/bin/python`. Do not use `env_isaaclab` (user's old, broken env).
- Isaac Lab source: `third_party/IsaacLab` (tag v2.3.2, git-ignored, editable install). Rebuild everything with `bash scripts/setup_env.sh`.
- Pins in `env/constraints.txt` (torch 2.7.0+cu128, warp 1.11.1, rsl-rl-lib 3.1.2, skrl 1.4.3, tensordict 0.10.0). `pip` extras try to upgrade torch to cu13 builds: always install with `-c env/constraints.txt`.
- Always `export OMNI_KIT_ACCEPT_EULA=YES`. NVIDIA's asset CDN (Nucleus/S3) is effectively unreachable from this network
  (stock tasks hang >10 min loading USDs) → **all assets in this project are procedural**; never depend on Nucleus.

## Gotchas learned the hard way
- `sim.reset()` performs 2 warm-up physics steps → re-write the root state before measuring anything.
- Standalone scripts using a raw `SimulationContext` must set `SimulationContext.instance()._disable_app_control_on_stop_handle = True`
  before `app.close()`, otherwise close() blocks forever. Envs: `env.close()` handles it.
- Exit codes of Isaac scripts are unreliable (close() may swallow exceptions) → validation scripts write JSON with `pass` flags; check those.
- `pkill -f`/`pgrep -f <pattern>` match your own shell when the pattern appears in the command line (wait loops never end, kills hit yourself). Use a regex char class: `pgrep -f "[r]un_name foo"`, or kill by PID.
- rsl_rl 3.1.2's `RslRlVecEnvWrapper.get_observations()` calls `_get_observations()` only (not `_get_states()`): return
  all obs groups (`policy`, `critic`, `image`) from `_get_observations()`.
- `DirectRLEnv.physics_dt` is a read-only property.
- Mass/inertia randomization uses CPU PhysX views → done once at startup; per-episode variation via model-side params + payload force.
- PX4 SITL (native: ROS 2 Jazzy + Gazebo Harmonic on the host): run PX4 with `/opt/ros` removed from `PATH` (ROS's vendored
  `gz` cannot start PX4's world); a surviving `gz sim` server is silently re-used by the next PX4 → kill PX4 by process group;
  PX4 ≥ 1.16 versioned topics (`/fmu/out/vehicle_status_v1`); x500 airframe needs `NAV_DLL_ACT=0` to arm without a GCS;
  rclpy's `MultiThreadedExecutor` dropped ~35 % of 300 msg/s in the recorder → use `SingleThreadedExecutor`.
- SITL measurement: set `EKF2_GPS_DELAY=0` (gz GNSS has no latency; the 110 ms default makes EKF2 lead the truth by
  0.1 s — never copy to hardware); PX4's gz ground-truth velocity / angular velocity are backward differences (the latter
  of Euler angles) → differentiate the ground-truth position / quaternion centrally. Only one PX4 SITL at a time
  (`run_sitl.sh` refuses to start otherwise); queue SITL jobs.
- `git worktree`: training writes `./logs/rsl_rl` relative to the cwd → in a worktree, symlink `logs/*`, `datasets`,
  `outputs` to the main checkout so runs survive the worktree.
- The conda env's editable `quadsim` install points at the main checkout: in a worktree prefix `PYTHONPATH=$PWD/source/quadsim`.
- 8 GB GPU: at most two simulator processes at a time; a rendered-camera job (~4.3 GB at 64 envs) cannot run next to a
  4096-env training (PhysX "failed to allocate", the process then hangs holding memory -> `kill -9`). Serialize
  rendering jobs in a queue (`scripts/il/rgb_pipeline.sh`).
- Kit's `app.close()` may exit before stdout is flushed: `print(..., flush=True)` in standalone scripts.
- `quadsim.tasks` / `isaaclab.utils` need the app (pxr); keep CPU-testable code (terms, off-policy, IL) free of
  `configclass` (plain dataclasses work inside configclass env cfgs).
- Evaluating controllers in one process vs fresh processes gives slightly different (CI-equivalent) numbers; the
  benchmark runner always uses fresh processes.
- LeRobot needs NumPy 2 / pyarrow >= 21: only inside `third_party/lerobot-venv` (never in `quadsim`).

## Layout
- `source/quadsim/quadsim/registry.py` – extension registries + plugin loader.
- `source/quadsim/quadsim/core` – pure PyTorch (no Isaac import): params + vehicles, motor/aero model, reference integrator, controllers + action interfaces, sensors, wind.
- `source/quadsim/quadsim/isaac` – procedural USD spawner.
- `source/quadsim/quadsim/tasks` – Isaac Lab DirectRLEnv tasks (base env + terms; hover/track/waypoint/nav/race/swarm/langnav), `mdp.py` term library, `baselines.py`.
- `source/quadsim/quadsim/{rl,il,benchmark,deploy,utils}` – algorithms, imitation/VLA, benchmark suites, export/deploy, hardware.
- `examples/` – runnable extension examples (checked by `scripts/validation/examples_check.py`).
- `scripts/validation` – milestone acceptance tests (write `logs/validation/*.json`; curated copies in `docs/results/`).
- `tests/core` – fast CPU unit tests: `~/miniconda3/envs/quadsim/bin/python -m pytest tests/core -q`.

## Conventions
- World Z-up, body FLU, quaternions wxyz (`q_WB`), PX4 Quad-X rotor order (0 FR ccw, 1 RL ccw, 2 FL cw, 3 RR cw).
- Policy obs = deployable quantities only (noisy/delayed estimates, gyro, last action, task info); privileged data → `critic` group.
- Commit + push after each validated milestone (`git push origin main`; remote `git@github-new:TuanDinh97/Drone_Sim_RL.git`). Never force-push; never commit logs, checkpoints, datasets or third_party.
