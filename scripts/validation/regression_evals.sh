#!/usr/bin/env bash
# Bit-exact regression of the reference evaluations (run before and after refactoring the framework core).
#   bash scripts/validation/regression_evals.sh logs/eval/regress_<tag>
#   python scripts/validation/compare_evals.py logs/eval/regress_<before> logs/eval/regress_<after>
# Covers the CTBR (hover, tracking, waypoint) and velocity (range navigation) action interfaces, the PPO policies
# and the classical baselines (cascade, SE(3), potential field). Needs the session-1 checkpoints in logs/rsl_rl/.
set -u
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/source/quadsim${PYTHONPATH:+:$PYTHONPATH}"
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
OUT="${1:?usage: regression_evals.sh <out_dir>}"
mkdir -p "$OUT"
R=logs/rsl_rl
run() { echo "[regress] $1"; shift; $PY scripts/rl/evaluate.py --headless --out_dir "$OUT" "$@" > "$OUT/$(date +%s%N).log" 2>&1; }
run hover --task Quadsim-Hover-v0 --controllers policy cascade --checkpoint $R/quadsim_hover/*_ctbr_s42 \
    --episodes 1000 --num_envs 500
run track --task Quadsim-Track-v0 --controllers policy cascade geometric --checkpoint $R/quadsim_track/*_track_ctbr_s42 \
    --episodes 500 --num_envs 250
run waypoint --task Quadsim-Waypoint-v0 --controllers policy cascade --checkpoint $R/quadsim_waypoint/*_waypoint_ctbr_s42 \
    --episodes 500 --num_envs 250
run nav --task Quadsim-Nav-Range-v0 --controllers policy potential_field --checkpoint $R/quadsim_nav_range/*_nav_range_v2_s42 \
    --episodes 500 --num_envs 250 env.curriculum=False
ls "$OUT"/*.json
