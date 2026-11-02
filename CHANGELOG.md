# Changelog

Numbers quoted below are the current (re-measured) values; `docs/PROGRESS.md` holds the full verified log.

## 0.2.0 - 2026-11-02 - research framework, review and re-measurement

### Added
- Extension registries and plug-ins (`quadsim.registry`, `--plugins` on every script): vehicles (YAML/JSON files,
  generic N-rotor frames, Isaac Lab `MultirotorCfg` conversion), action interfaces, algorithms, baselines, benchmarks.
- Configurable MDP terms (observations with rate / latency / noise / history, rewards, terminations, costs, events,
  curricula), PPO-Lagrangian with a PID multiplier, optional constraint warm-up and budget curriculum.
- GPU-native SAC / TD3 with asymmetric critics; evaluation-only BC and VLA algorithms.
- Tasks: drone racing (`Quadsim-Race-v0/v1`), multi-agent swap (`Quadsim-Swarm-v0`), language-conditioned navigation
  with a map or an RGB camera (`Quadsim-LangNav-*`), constrained waypoint racing (`Quadsim-Waypoint-Safe-v0/v1/v2`),
  hexarotor hover; option `env.task_end_is_terminal`.
- Imitation learning: episode-structured demonstrations with instructions, BC / DAgger (FiLM CNN, bag-of-words or a
  frozen CLIP text encoder), LeRobot v3.0 export verified with LeRobot's loader, dataset cards.
- Remote policies over the openpi policy-server protocol (`quadsim.il.remote`, `scripts/il/serve_policy.py`).
- Benchmark suites with in- / out-of-distribution conditions; aggregate statistics with stratified bootstrap
  intervals and probability of improvement (`quadsim.benchmark.stats`, rliable definitions).
- Tools: regression evaluations, re-evaluation from result metadata, example checks, hardware-aware `num_envs`.
- Continuous integration: GitHub Actions runs the CPU unit tests with CPU-only PyTorch (`.github/workflows/tests.yml`).

### Fixed (session-5 review; details in `docs/PROGRESS.md`, "Review and corrections")
- `evaluate.py` stopped at a global episode count, which over-weights short episodes; every environment now
  contributes its first `ceil(episodes / num_envs)` episodes. **All published evaluation numbers were re-measured**;
  the largest changes: figure-8 racing 0 → 5 % (v0) and 54 → 74 % (v1), swarm straight-line baseline 18 → 36 %,
  potential-field navigation 49.8 → 39.4 %.
- Cascade baseline integrator reset per episode; initial angular velocity frame at reset; SAC double target update;
  off-policy resume; runaway curriculum; mixer for asymmetric frames; overrides after list options; vehicle-file
  parsing and validation; recorded vehicle in `env.yaml`; hardware check; export of plug-in tasks and algorithms;
  collector sources and stale benchmark files; stale first camera frame after resets; unbounded cost logging;
  multiplier wind-up; silent reward-term overrides; degenerate race circuits; LangNav layout checks; `speed_limit`
  overrides; BC on tiny datasets.

### Fixed (after the review)
- Fresh installs: `env/constraints.txt` pinned `packaging==23.0`, which makes tensordict 0.10.0 unresolvable (its
  `pyvers` requires packaging >= 25); `setup_env.sh` now restores Isaac Sim's pin after the RL stack.
- Remote policy client: compatible with websockets 17 (connection used as a context manager); received arrays are
  copied before conversion; batched servers may return action chunks.
- Docker: no root-owned bytecode in the mounted sources.

### Changed
- Racing, swarm, safe RL and LangNav results are reported over 3 seeds.
- Optional dependency groups: `remote` (websockets, msgpack) and scipy in `dev`; the tests that need them are skipped
  without them.

## 0.1.0 - 2026-11-01 - platform
- Validated quadrotor physics (PhysX vs reference integrator 0.4 mm over 1.5 s), sensors (IMU, barometer, estimate,
  analytic and RTX depth agreeing to 7 mm), hover / tracking / waypoint / navigation tasks, PPO with asymmetric
  critics, domain randomization, ONNX export with an interface contract, ROS 2 PX4 offboard bridge flown in PX4 v1.16
  SITL, Docker image, LaTeX report.
