#!/usr/bin/env bash
# Final framework evaluations that need an otherwise idle GPU (run after scripts/rl/framework_evals.sh).
set -u
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/source/quadsim${PYTHONPATH:+:$PYTHONPATH}"
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
echo "[final] $(date +%T) hexarotor motor-level policy"
R=$(ls -d logs/rsl_rl/quadsim_hover_hexa/*_hexa_motor_s42 | tail -1)
$PY scripts/rl/evaluate.py --headless --task Quadsim-Hover-Hexa-Motor-v0 --controllers policy --checkpoint $R \
    --episodes 1000 --num_envs 500 --out_dir logs/eval/vehicles --tag hexa_motor_s42 2>&1 | grep "\[eval\] .*:"
echo "[final] $(date +%T) RGB + CLIP text BC"
bash scripts/il/rgb_clip.sh
echo "[final] $(date +%T) VLA adapter reference on RGB LangNav (300 episodes)"
for T in Quadsim-LangNav-RGB-v0 Quadsim-LangNav-RGB-Heldout-v0; do
  $PY scripts/rl/evaluate.py --headless --enable_cameras --num_envs 64 --task $T --controllers policy --algo vla \
      --checkpoint examples/vla_color_servo.py:make_adapter --episodes 300 --out_dir logs/eval/il \
      --tag vla_color_servo 2>&1 | grep -E "\[eval\] policy:|Error" | cut -c1-200
done
echo "[final] $(date +%T) LiDAR example"
$PY scripts/validation/examples_check.py --only sensor_lidar_nav 2>&1 | grep "\[examples\]"
$PY scripts/tools/collect_framework_results.py
$PY scripts/tools/report_tables.py > /dev/null
echo "[final] $(date +%T) done"
