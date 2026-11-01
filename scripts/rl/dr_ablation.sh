#!/usr/bin/env bash
# Domain-randomization ablation: evaluate a DR-trained and a no-DR-trained policy under three conditions.
# Usage: bash scripts/rl/dr_ablation.sh <run_dir_with_DR> <run_dir_without_DR> [task]
set -u
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=YES
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
DR="$1"; NODR="$2"; TASK="${3:-Quadsim-Hover-v0}"
declare -A COND
COND[ideal]='env.randomization.enabled=False env.sensors.noise_enabled=False env.control.action_delay_steps=(0,0)'
COND[randomized]=''
COND[harsh]='env.randomization.mass_scale=(0.8,1.25) env.randomization.motor_tau_scale=(1.5,2.5) env.randomization.k_f_scale=(0.8,1.0) env.randomization.sensor_noise_scale=(1.5,2.0) env.control.action_delay_steps=(2,4) env.wind.enabled=True'
for c in ideal randomized harsh; do
  for p in DR noDR; do
    run=$([ "$p" = DR ] && echo "$DR" || echo "$NODR")
    $PY scripts/rl/evaluate.py --headless --task "$TASK" --controllers policy --checkpoint "$run" \
        --episodes 1000 --num_envs 500 --tag "ablation_${c}_${p}" ${COND[$c]} > "logs/eval_ablation_${c}_${p}.log" 2>&1
    grep "\[eval\] policy:" "logs/eval_ablation_${c}_${p}.log"
  done
done
