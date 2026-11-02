# Troubleshooting

Problems met while building and validating quadsim, with their fixes.

## Installation and environment

| Symptom | Cause / fix |
|---|---|
| `pip install` upgrades torch to a cu13 build | always install with `-c env/constraints.txt` (torch 2.7.0+cu128, warp 1.11.1, rsl-rl-lib 3.1.2, ...) |
| `ResolutionImpossible`: tensordict / `pyvers` vs `packaging==23.0` | Isaac Sim 5.1 pins `packaging==23.0`, tensordict 0.10.0's `pyvers` requires >= 25: never constrain packaging while installing the RL stack; `setup_env.sh` restores 23.0 last (pip prints the known conflict; both work with 23.0) |
| Files under `source/`, `logs/`, `datasets/` owned by root after using Docker | the container runs as root; it no longer writes bytecode into the mounted sources, outputs remain root-owned: `sudo chown -R $USER logs datasets outputs` |
| Stock Isaac Lab tasks hang > 10 min loading USDs | NVIDIA's Nucleus/S3 asset CDN unreachable; quadsim only uses procedural assets (`quadsim.isaac.spawn`) - never depend on Nucleus |
| `ModuleNotFoundError: pxr` when importing `quadsim.tasks` / `isaaclab.utils` outside a simulator script | these need a running `SimulationApp`; the CPU tests only import `quadsim.core`, `quadsim.il`, `quadsim.rl.offpolicy`, `quadsim.utils` (term configs are plain dataclasses for this reason) |
| LeRobot import fails with `pyarrow requires NumPy 2.0` | LeRobot lives in its own venv (`scripts/il/setup_lerobot_env.sh`) with NumPy 2; the simulator env stays on NumPy 1.26 |
| `pip` resolves LeRobot's dependencies for > 20 min | install `lerobot --no-deps` plus the dataset dependencies only (what the setup script does) |
| Docker build cannot pull `nvcr.io` images | the Docker daemon ignores the shell proxy; use `scripts/tools/pull_image_via_proxy.py` |

## Running simulations

| Symptom | Cause / fix |
|---|---|
| `PxgCudaDeviceMemoryAllocator failed to allocate`, `Failed to create simulation view backend` | GPU memory exhausted by concurrent simulator processes (8 GB GPU: at most two at a time, e.g. one 4096-env training + one evaluation) |
| Script ends without its final prints | Kit's `app.close()` can exit before Python flushes stdout: use `print(..., flush=True)`; validation scripts write JSON files with `pass` flags instead of relying on exit codes |
| `close()` blocks forever in a standalone `SimulationContext` script | set `SimulationContext.instance()._disable_app_control_on_stop_handle = True` before `app.close()` |
| A multi-task smoke test fails on the 2nd/3rd task with a PhysX backend error | run tasks in separate processes (`--tasks` one at a time) |
| Measurements right after `sim.reset()` are off | reset performs two warm-up physics steps; re-write the root state before measuring |
| Evaluation numbers differ slightly between `evaluate.py --controllers a b` and separate runs | a controller evaluated after another in the same process starts from a different simulator/RNG state; benchmark suites run every controller in a fresh process (cascade hover: 88.7 % vs 89.8 %, within the CI) |
| One unit test fails only while a training runs | the CUDA variant of `test_raycast_warp` needs free GPU memory; run `CUDA_VISIBLE_DEVICES= pytest tests/core` (CPU only) or on an idle GPU |
| `--num_envs` silently lower than the task default | `quadsim.utils.hardware` lowered it because the free GPU memory would not fit (it depends on what other processes hold at that moment); pass `--num_envs` explicitly, always for seeds and ablations |
| `ModuleNotFoundError: No module named 'env.control'` with `--plugins my.py env.control.mode=x` (versions before the review) | argparse attached the override to the `--plugins` list; current scripts move `env.`/`agent.` items back to the overrides (older versions: put overrides before `--plugins`) |
| `PhysX ... failed to allocate` although only one simulator runs | non-simulator GPU jobs count too: an RGB behaviour-cloning training (`train_bc.py --images`) holds ~2 GB next to a 4096-env training |
| Numbers differ from results published before the review | `evaluate.py` used to stop at a global episode count, which drops the episodes still running (the long ones) and over-weights short episodes; every env now contributes its first `ceil(episodes / num_envs)` episodes, and all published numbers were re-measured (`docs/PROGRESS.md`, "Review and corrections") |

## Learning

| Symptom | Cause / fix |
|---|---|
| rsl_rl ignores the critic group | rsl_rl 3.1.2's wrapper only calls `_get_observations()`: return `policy`, `critic` (and `image`) from it |
| SAC learns but the policy is poor (skrl, early runs) | task rewards are dt-scaled (~0.02 per step) and the entropy bonus dominates; quadsim's SAC scales rewards to per-second units and uses an asymmetric critic, large batches and 4 updates per step |
| Behaviour cloning copies its previous action and never stops | causal confusion through the `last_action` observation: train with `--drop last_action` |
| Language-conditioned BC fails on unseen phrasings | out-of-vocabulary words are ignored by the bag-of-words encoder; only colour/keyword words generalize (held-out phrasing ~60 %) - use a frozen pretrained text encoder (`train_bc.py --text_encoder stabilityai/sd-turbo:text_encoder`: 99.8 %) |
| Expert demonstrations contain collisions | check the scenario sampler: rejection sampling with infeasible constraints can silently return invalid layouts (fixed in LangNav with a guaranteed-spacing sampler) |
| Multi-agent groups desynchronize | rsl_rl randomizes `episode_length_buf` at the start of training; `SwarmEnv` re-synchronizes the clock of every group each step |
| `ValueError: reward terms [...] already exist in the task's reward` | a configured reward term had the name of one of the task's own terms (it used to replace it silently); give it its own name |
| `train.py --algo bc` fails | behaviour cloning is trained by `scripts/il/train_bc.py`; `bc` and `vla` are evaluation-only algorithms |
| A curriculum climbs to its hardest level regardless of performance (versions before the review) | `attribute_curriculum` averaged the last episode of every env, mostly from easier levels; it now judges each level only on episodes that started at it |

## Shell / workflow

| Symptom | Cause / fix |
|---|---|
| A queue script fails with a syntax error on a line that looks correct | it was edited while running: bash reads scripts incrementally; never edit a running script |
| `pkill -f pattern` kills your own shell | the pattern also matches the shell's command line; use a character class (`pgrep -f "[r]un_name foo"`) or kill by PID |
| Results vanish with a git worktree | training writes `./logs` relative to the cwd: symlink `logs`, `datasets`, `outputs` to the main checkout |
| A worktree run imports the main checkout's code | the editable install points at the main checkout: prefix `PYTHONPATH=$PWD/source/quadsim` |
| A queue's later jobs behave differently from its first ones | every queued process imports the checkout's source when it starts: do not edit `source/` while a queue still has jobs to start (make fixes in a separate worktree and merge afterwards) |
| PX4 SITL: gz world does not start / an old world is reused / cannot arm | see `CLAUDE.md` (strip `/opt/ros` from `PATH`, kill PX4 by process group, `NAV_DLL_ACT=0`, `EKF2_GPS_DELAY=0` in SITL only) |
