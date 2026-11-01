# CLAUDE.md — working notes for this repository

Quadrotor Sim-to-Real RL research platform on **Isaac Lab v2.3.2 + Isaac Sim 5.1.0**.
Read `docs/PROGRESS.md` (verified state, open issues) and `docs/ROADMAP.md` (next steps) before working.

## Current state (2026-11-01)
All seven milestones meet their acceptance criteria (details and caveats in docs/PROGRESS.md). Trained runs live in
`logs/rsl_rl/` (git-ignored): hover `*_ctbr_s42`, tracking `*_track_ctbr_s42`, waypoint `*_waypoint_ctbr_s42`,
nav range `*_nav_range_v2_s42`, nav depth `*_nav_depth_v2_s42` (model_900). The hover policy flies PX4 v1.16.2
SITL (gz x500) through `ros2/quadsim_px4_bridge` (docs/PROGRESS.md "PX4 SITL cross-check"; PX4 + ROS 2 workspace in
`third_party/`, built by `ros2/px4_sitl/setup_px4_sitl.sh`, runs via `scripts/sitl/run_sitl.sh`).
Next steps: docs/ROADMAP.md ("Beyond the current scope").

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
- The conda env's editable `quadsim` install points at the main checkout: in a worktree prefix `PYTHONPATH=$PWD/source/quadsim`.

## Layout
- `source/quadsim/quadsim/core` – pure PyTorch (no Isaac import): params, motor/aero model, reference integrator, controllers, sensors, wind.
- `source/quadsim/quadsim/isaac` – procedural USD spawner.
- `source/quadsim/quadsim/tasks` – Isaac Lab DirectRLEnv tasks (base env + hover/track/waypoint/nav).
- `scripts/validation` – milestone acceptance tests (write `logs/validation/*.json`; curated copies in `docs/results/`).
- `tests/core` – fast CPU unit tests: `~/miniconda3/envs/quadsim/bin/python -m pytest tests/core -q`.

## Conventions
- World Z-up, body FLU, quaternions wxyz (`q_WB`), PX4 Quad-X rotor order (0 FR ccw, 1 RL ccw, 2 FL cw, 3 RR cw).
- Policy obs = deployable quantities only (noisy/delayed estimates, gyro, last action, task info); privileged data → `critic` group.
- Commit + push after each validated milestone (`git push origin main`; remote `git@github-new:TuanDinh97/Drone_Sim_RL.git`). Never force-push; never commit logs, checkpoints, datasets or third_party.
