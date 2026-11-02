# Benchmark protocol

This page defines how methods are evaluated and compared in quadsim, so that results from different papers, labs or
algorithms are comparable. The suites live in `quadsim.benchmark.suites` (`python scripts/benchmark/run.py --list`);
the statistics in `quadsim.benchmark.stats`.

## Evaluation protocol

1. **Episodes.** `scripts/rl/evaluate.py` runs complete episodes with a fixed evaluation seed (1234) for every
   controller; every environment contributes its first `ceil(episodes / num_envs)` episodes (stopping at a global
   episode count would over-weight short episodes - quadsim's results before session 5 were affected and re-measured).
   Policies are deterministic (mean action).
2. **Conditions.** A suite evaluates every controller on in-distribution (`id`) conditions (the training
   distribution) and out-of-distribution (`ood`) conditions (perturbations, unseen layouts or phrasings), each in a
   fresh simulator process with the same seed for every controller. Never tune on `ood` conditions.
3. **Fairness.** Policies observe only deployable quantities (noisy, delayed estimates, gyro, last action, task
   data, configured sensors); classical baselines read the same estimates and act through the same action
   interface, latencies and actuators.
4. **Seeds.** Train at least 3 seeds with identical configurations; pin `--num_envs` (the hardware check otherwise
   adapts it to the free GPU memory) and report the training budget (iterations or environment steps).
5. **Reporting.** For every condition: per-seed values, mean ± sample std, and a stratified bootstrap 95 % CI over
   seeds; across conditions: the IQM with its CI and, against a reference method, the probability of improvement
   P(X > Y) with its CI (Agarwal et al., NeurIPS 2021). With 3 seeds the intervals are wide; report them anyway.
   The generalization gap is `id - mean(ood)` of the primary metric.
6. **Generative and chunked policies** (flow matching, diffusion, VLAs): seed the sampling noise (the policy
   servers take `--seed`) and report the number of actions executed per predicted chunk (`QUADSIM_REMOTE_HORIZON`);
   it changes the results (SmolVLA on LangNav-RGB: 55 / 30 / 9 % crashes with 1 / 10 / 50 actions per chunk).

## Suites

| Suite | Task | Conditions (split) | Episodes / envs | Primary metric (success definition) | Baselines |
|---|---|---|---|---|---|
| `hover` | `Quadsim-Hover-v0` | nominal (id); wind + pushes, harsh dynamics, sensor noise ×2-3, latency (ood) | 1000 / 500 | success: position error < 10 cm and speed < 0.2 m/s at the end of the 8 s episode, after a 57°/3 rad/s initial perturbation | cascade |
| `track` | `Quadsim-Track-v0` | as hover, + slower references (ood) | 500 / 250 | tracking RMSE (lower is better) | cascade, SE(3) |
| `waypoint` | `Quadsim-Waypoint-v0` | as hover | 500 / 250 | success: 5 waypoints passed within 15 s | cascade |
| `nav_range` | `Quadsim-Nav-Range-v0` | 4-24 obstacles (id); dense forests (20-24), wind, latency (ood) | 500 / 250 | success: goal reached without collision | potential field |
| `race` | `Quadsim-Race-v0` | random circuits (id); fixed ellipse, figure-8, wind (ood) | 500 / 250 | success: 2 laps (14 gates in order) without crash within 20 s | gate-aligned cascade |
| `swarm` | `Quadsim-Swarm-v0` | 4-agent swap (id); wind, latency (ood) | 500 / 256 (agent rows) | group success: all 4 agents at their goals at the end of the 10 s episode, none crashed | straight lines, potential field |
| `langnav` | `Quadsim-LangNav-v0` | training phrasings (id); held-out phrasings, wind (ood) | 500 / 250 | success: within 1 m of the instructed pillar's surface below 0.5 m/s for 0.2 s (10 s episode), no wrong pillar, no collision | privileged expert |

Latency condition: 20-30 ms command delay plus 0-30 ms estimate delay (training: 0-10 ms). Vision variants
(`Quadsim-LangNav-RGB-v0`, `-Heldout`) run the same protocol with `--enable_cameras` (64 envs, 300 episodes).

## Evaluating a new method

```bash
# any registered algorithm (PPO, SAC, TD3, BC, a plug-in) - one checkpoint per seed
python scripts/benchmark/run.py --suite race --policy mine_s1=logs/<algo>/quadsim_race/<run_s1> \
    --policy mine_s2=... --policy mine_s3=... --tag mine
# an external VLA / IL model served over the openpi protocol (any Python environment or machine)
QUADSIM_POLICY_SERVER=host:8000 python scripts/rl/evaluate.py --headless --task Quadsim-LangNav-RGB-v0 \
    --enable_cameras --num_envs 16 --controllers policy --algo vla --checkpoint quadsim.il.remote:make_adapter
# a LeRobot policy (SmolVLA, pi0, ACT, ...) fine-tuned on a quadsim LeRobot export, served from the LeRobot venv
third_party/lerobot-venv/bin/python scripts/il/serve_lerobot_policy.py --policy <run>/checkpoints/last/pretrained_model \
    --dataset datasets/lerobot/langnav_rgb_v2 --max_batch 32 --seed 0 --port 8000 &
QUADSIM_REMOTE_BATCHED=1 QUADSIM_REMOTE_HORIZON=10 python scripts/rl/evaluate.py --headless --enable_cameras \
    --task Quadsim-LangNav-RGB-Heldout-v0 --num_envs 64 --episodes 300 --controllers policy --algo vla \
    --checkpoint quadsim.il.remote:make_adapter
```

`logs/benchmark_suites/<suite>/<tag>/summary.md` holds one table per metric and the generalization gap; aggregate
seeds with `quadsim.benchmark.stats` (example: `scripts/tools/collect_multiseed.py`).

## Reference results (3 seeds; per-seed values and intervals in `docs/results/framework/multiseed.json`)

| Suite / method | id | ood conditions | Notes |
|---|---|---|---|
| race, PPO v0 (counter-clockwise circuits) | 96.1 ± 1.5 % | ellipse 99.0 ± 0.7, figure-8 **4.1 ± 3.2**, wind 96.1 ± 0.5 % | 2000 iterations × 4096 envs |
| race, PPO v1 (both turning directions) | 95.7 ± 1.0 % | ellipse 97.5 ± 0.5, figure-8 **73.2 ± 6.3**, wind 95.5 ± 0.2 % | P(v1 > v0) = 1.00 on the figure-8, 0.38 over all four conditions |
| race, PPO v1, finish as a terminal state | 93.4 ± 2.7 % | ellipse 96.1 ± 2.3, figure-8 46.3 ± 26.5, wind 93.1 ± 2.5 % | P(terminal > time-out) = 0.25 over all four conditions; laps 3.0 vs 2.5 s |
| race, gate-aligned cascade | 0.6 % | 0-0.6 % | 25 % crashes |
| swarm, shared PPO (centralized critic) | 89.6 ± 2.4 % | wind 85.4 ± 7.0, latency 76.6 ± 3.9 % | 1.7 % agent collisions; 2.0 s |
| swarm, tuned potential field | 100 % | 97.7 / 96.1 % | 6.9 s (3.4× slower) |
| langnav (map), BC + 2 DAgger rounds, bag of words | 100 % | held-out phrasings 73.5 ± 15.1 % | same data with a frozen CLIP text tower: 96.9 ± 1.2 % |
| langnav (RGB, no map), BC + 1 DAgger round | 100 % | held-out phrasings 98.4 ± 2.7 % | BC alone 76.7 ± 5.6 % |
| langnav (RGB), same on demonstrations re-collected after the camera fix | 99.8 ± 0.4 % | held-out 99.2 ± 1.4 % | BC alone 80.2 ± 6.1 / 77.6 ± 16.6 % |
| langnav (RGB), SmolVLA fine-tuned on those demonstrations (1 run, 10 k steps) | 19.1 % (10 actions per chunk), 27.2 % (50) | held-out 19.4 / 22.2 % | wrong pillar 45-66 %: chance-level target choice |
| waypoint-safe, PPO-Lagrangian (budget 0.1 s) | cost 0.105 ± 0.023 s | - | course success 87.6 / 0 / 85.2 % per seed |

Single-seed reference results (hover algorithms, tracking, waypoints, navigation, vehicles) are in `README.md` and
`docs/PROGRESS.md`.
