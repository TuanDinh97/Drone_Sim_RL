# Roadmap

Ordered by dependency; each milestone has explicit acceptance criteria that are executed before it is marked done
(see `docs/PROGRESS.md`).

| # | Milestone | Acceptance criteria | Status |
|---|---|---|---|
| 1 | Isaac Sim/Lab foundation | headless boot; PhysX free fall matches symplectic Euler < 1e-3 m | ✅ |
| 2 | Quadrotor dynamics | unit tests pass; PhysX vs reference open-loop < 1 cm / 1.5 s; inertia exact; motor/yaw models < 3 % | ✅ |
| 3 | Sensors, visualization, recording | IMU static/dynamic statistics match configured noise (Allan-type check); analytic depth vs rendered depth agree (median abs err < 2 cm); synchronized HDF5 recording round-trips; GUI launch script works | ⬜ |
| 4 | State-based RL (hover, tracking, waypoints) | training runs to completion with checkpoints & resume; evaluated policy success/error metrics reported vs classical baselines (cascade PID / SE(3)) | ⬜ |
| 5 | Vision RL (depth / RGB / RGB-D) | depth-based obstacle-avoidance policy trained (raycast depth, many envs) and evaluated in rendered-depth envs; RGB/RGB-D pipelines execute end to end | ⬜ |
| 6 | Domain randomization & Sim-to-Real interfaces | DR ablation (policy trained with vs without DR evaluated on perturbed dynamics); ONNX export reproduces PyTorch outputs < 1e-5; ROS 2 / PX4 interface package with frame conversions unit-tested | ⬜ |
| 7 | Docker, benchmarks, documentation, report | Docker image builds and runs the unit tests headless; throughput/VRAM benchmark table; README; LaTeX report compiled | ⬜ |

## Beyond the current scope (future work, prioritized)
1. Real-flight system identification (thrust stand, k_f, k_m, τ; bifilar pendulum inertia) for the target vehicle; update presets.
2. PX4 SITL/HITL loop: run exported policy as an offboard ROS 2 node against PX4 SITL (gz x500) to cross-check dynamics.
3. Hardware-in-the-loop latency measurement and estimator-in-the-loop (PX4 EKF2 outputs replace the odometry noise model).
4. Rendered-camera domain randomization (textures, lighting, sim-to-real image gap), event cameras.
5. Isaac Lab 3.x migration (Isaac Sim 6.2, quaternion xyzw convention change; the core is framework-agnostic to ease this).
