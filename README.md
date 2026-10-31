# Drone_Sim_RL — Quadrotor Sim-to-Real RL platform (Isaac Lab)

Research platform for learning-based quadrotor control with an explicit Sim-to-Real focus, built on
**Isaac Lab v2.3.2 / Isaac Sim 5.1.0**. Work in progress — see `docs/PROGRESS.md` for verified results and
`docs/ROADMAP.md` for the plan. Full documentation will be added in Milestone 7.

Quick start (native):
```bash
bash scripts/setup_env.sh            # conda env "quadsim" + Isaac Sim + Isaac Lab + quadsim
conda activate quadsim && export OMNI_KIT_ACCEPT_EULA=YES
python -m pytest tests/core -q       # fast physics/controller unit tests (CPU)
python scripts/validation/m2_dynamics_validation.py --headless
```
