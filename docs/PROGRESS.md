# Progress log (verified results only)

Legend: ✅ verified by an executed test (artifact linked) · 🟡 implemented, partially verified · ⬜ not started

## Milestone 1 — Isaac Sim / Isaac Lab foundation ✅ (2026-11-01)
- Stack: Isaac Sim 5.1.0 (pip) + Isaac Lab v2.3.2 + torch 2.7.0+cu128 on RTX 5060 8 GB (sm_120).
- `scripts/validation/m1_isaac_smoke_test.py`: headless boot 2.8 s; 64 rigid bodies free fall 0.5 s at dt=1/400:
  |z − z_symplectic| = 9.5e-7 m, |z − z_analytic| = 6.1e-3 m (= expected O(dt) bias g·t·dt/2). → `docs/results/m1_smoke_test.json`
- Finding: PhysX rigid bodies integrate with symplectic (semi-implicit) Euler.
- Finding: Isaac Lab stock `Isaac-Quadcopter-Direct-v0` cannot run here: its Crazyflie USD download from NVIDIA's CDN
  did not complete within 625 s (network). Hence all quadsim assets are procedural.

## Milestone 2 — Quadrotor dynamics ✅ (2026-11-01)
- Pure-PyTorch core (`quadsim.core`): presets X500 (PX4 SITL gz model) and Crazyflie 2.x (Förster 2015),
  exact first-order asymmetric motor model, thrust/yaw-moment, rotor drag + rolling moment, body drag,
  rotor gyroscopic & spin-up torques, ground effect, wind (mean + Gauss–Markov turbulence + 1-cos gusts),
  push disturbances, mixer with PX4-style desaturation, rate PID / attitude P / position-velocity cascade, Lee SE(3) controller,
  RK4 + semi-implicit reference integrators.
- Unit tests `tests/core` (27 tests, CPU, ~10 s): ✅ all pass. Include torque sign conventions, exact motor response,
  Euler-equation conservation (Dzhanibekov tumbling), first-order convergence of semi-implicit → RK4,
  closed-loop cascade step (<5 cm), Crazyflie recovery from 46° tilt + 3.7 rad/s spin, SE(3) circle tracking RMSE < 0.1 m.
- Isaac integration: procedural USD quadrotor (explicit MassAPI), body wrench written via `permanent_wrench_composer`
  each physics step (200 Hz), PhysX integrates rigid body + contacts.
- `scripts/validation/m2_dynamics_validation.py` (both presets ✅ all pass):

| Test | X500 | Crazyflie | criterion |
|---|---|---|---|
| T1 PhysX mass / inertia vs preset | 0 / 0 rel. err | 0 / 0 | < 1e-4 |
| T2 hover drift (2 s) | 0.0 mm | 0.0 mm | < 5 mm |
| T3 thrust step accel. RMS err / peak | 0.019 % | 0.026 % | < 2 % |
| T4 yaw accel. vs τ/I_zz | 0.004 % | 0.008 % | < 3 % |
| T5 open-loop 1.5 s pos. err vs semi-implicit ref. | 0.40 mm | 2.1 mm | < 1 cm |
| T5 att. err vs semi-implicit ref. | 0.006° | 0.066° | < 0.5° |
| T5 pos. err vs RK4 (integration error of PhysX @200 Hz) | 3.3 mm | 8.9 mm | (info) |
| T6 closed-loop velocity cascade PhysX vs ref. (3 s) | 12 mm | 9 mm | < 3 cm |

  Artifacts: `docs/results/m2_dynamics_{x500,crazyflie}.json`, `docs/figures/m2_dynamics_*.pdf`.

## Milestone 3 — Sensors, visualization, recording ⬜
## Milestone 4 — State-based RL ⬜
## Milestone 5 — Vision / multimodal RL ⬜
## Milestone 6 — Domain randomization & Sim-to-Real interfaces ⬜
## Milestone 7 — Docker, benchmarks, docs, report ⬜

## Open issues
- None blocking.
