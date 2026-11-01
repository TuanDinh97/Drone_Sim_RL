# Roadmap

Ordered by dependency; each milestone has explicit acceptance criteria that are executed before it is marked done
(see `docs/PROGRESS.md`).

| # | Milestone | Acceptance criteria | Status |
|---|---|---|---|
| 1 | Isaac Sim/Lab foundation | headless boot; PhysX free fall matches symplectic Euler < 1e-3 m | ✅ |
| 2 | Quadrotor dynamics | unit tests pass; PhysX vs reference open-loop < 1 cm / 1.5 s; inertia exact; motor/yaw models < 3 % | ✅ |
| 3 | Sensors, visualization, recording | IMU static/dynamic statistics match configured noise (Allan-type check); analytic depth vs rendered depth agree (median abs err < 2 cm); synchronized HDF5 recording round-trips; GUI launch script works | ✅ |
| 4 | State-based RL (hover, tracking, waypoints) | training runs to completion with checkpoints & resume; evaluated policy success/error metrics reported vs classical baselines (cascade PID / SE(3)) | ✅ hover, tracking, waypoints |
| 5 | Vision RL (depth / RGB / RGB-D) | depth-based obstacle-avoidance policy trained (raycast depth, many envs) and evaluated in rendered-depth envs; RGB/RGB-D pipelines execute end to end | ✅ (depth policy 61–67 % success; RGB perf. not claimed) |
| 6 | Domain randomization & Sim-to-Real interfaces | DR ablation (policy trained with vs without DR evaluated on perturbed dynamics); ONNX export reproduces PyTorch outputs < 1e-5; ROS 2 / PX4 interface package with frame conversions unit-tested | ✅ (bridge flown in PX4 SITL with the hover policy) |
| 7 | Docker, benchmarks, documentation, report | Docker image builds and runs the unit tests headless; throughput/VRAM benchmark table; README; LaTeX report compiled | ✅ |

### Research framework (session 4; details and numbers in `docs/PROGRESS.md`, architecture in `docs/FRAMEWORK.md`)

| # | Milestone | Acceptance criteria | Status |
|---|---|---|---|
| F1 | Extension core (registries, plugins, vehicles, action interfaces, configurable terms) | reference evaluations bit-identical after the refactor; a non-quadrotor vehicle trains | ✅ 59/59 bit-identical; hexarotor 99.8 % |
| F2 | Algorithms + benchmarks (SAC/TD3, unified CLI, ID/OOD suites, hardware-aware defaults) | SAC competitive with PPO on hover; suites report generalization gaps | ✅ SAC 99.9 %, TD3 99.6 % (2nd configuration) |
| F3 | Research tasks (racing, multi-agent, constrained RL) | trained + evaluated against baselines; constraint within budget | ✅ safe RL (0.082 s ≤ 0.1 s); racing / swarm in PROGRESS |
| F4 | Imitation learning / VLA (demos with instructions, BC/DAgger, LeRobot export, VLA adapter) | BC matches an RL expert; language grounding; export reloads in LeRobot | ✅ |
| F5 | Usability (examples, tutorials, troubleshooting, Docker, tests, report) | every example executes; tests pass in Docker | ✅ |

## Beyond the current scope (future work, prioritized)
0. Framework follow-ups: multi-seed results for racing / swarm / safe RL / language tasks; a cost critic and other
   constrained-RL algorithms; pretrained text/vision encoders and fine-tuning an open VLA on the exported LeRobot data;
   texture/lighting randomization for RGB; exploration and adaptive-control benchmark tasks; distributional critics and
   yaw-aware rewards for the off-policy learners.
1. Real-flight system identification (thrust stand, k_f, k_m, τ; bifilar pendulum inertia) for the target vehicle; update presets.
2. PX4 SITL/HITL loop: run exported policy as an offboard ROS 2 node against PX4 SITL (gz x500) to cross-check dynamics.
   **SITL done** for hover, tracking, waypoint and range-navigation policies (docs/PROGRESS.md, sessions 2–3), incl.
   rate- and velocity-loop identification and SITL-informed training distributions (v1 negative, v2 removes the heave
   limit cycle). Next: navigation trained with an identified/randomized autopilot velocity loop, tracking/waypoint on
   v2-like distributions, rendered gz camera for depth policies, HITL latency measurement.
3. Hardware-in-the-loop latency measurement and estimator-in-the-loop (PX4 EKF2 outputs replace the odometry noise model).
4. Rendered-camera domain randomization (textures, lighting, sim-to-real image gap), event cameras.
5. Isaac Lab 3.x migration (Isaac Sim 6.2, quaternion xyzw convention change; the core is framework-agnostic to ease this).
