#!/usr/bin/env bash
# End-phase evaluation pipeline (run on an otherwise idle GPU). Writes logs/final/*.log and logs/eval/*.json.
#   bash scripts/rl/final_evals.sh <nav_range_v2_run> <nav_depth_v2_run>
set -u
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=YES
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
RANGE="$1"; DEPTH="$2"
mkdir -p logs/final
EV="--headless --episodes 500 --num_envs 250 env.curriculum=False"

echo "[final] nav range v2 vs potential field"
$PY scripts/rl/evaluate.py --task Quadsim-Nav-Range-v0 --controllers policy potential_field --checkpoint "$RANGE" \
    --tag range_v2 $EV > logs/final/eval_range_v2.log 2>&1; grep "\[eval\]" logs/final/eval_range_v2.log

echo "[final] nav depth v2 on analytic (ray-cast) depth = training distribution"
$PY scripts/rl/evaluate.py --task Quadsim-Nav-Depth-v0 --controllers policy --checkpoint "$DEPTH" \
    --tag depth_v2_raycast $EV > logs/final/eval_depth_raycast.log 2>&1; grep "\[eval\]" logs/final/eval_depth_raycast.log

echo "[final] nav depth v2 zero-shot on RTX-rendered depth (sim-to-sim sensor gap)"
$PY scripts/rl/evaluate.py --task Quadsim-Nav-Depth-Render-v0 --enable_cameras --controllers policy --checkpoint "$DEPTH" \
    --headless --episodes 300 --num_envs 100 env.curriculum=False --tag depth_v2_rendered \
    > logs/final/eval_depth_rendered.log 2>&1; grep "\[eval\]" logs/final/eval_depth_rendered.log

echo "[final] RGB-D end-to-end training pipeline (execution check, 10 iterations)"
$PY scripts/rl/train.py --task Quadsim-Nav-RGBD-v0 --headless --enable_cameras --num_envs 64 --max_iterations 10 \
    --run_name rgbd_pipeline_check > logs/final/train_rgbd_check.log 2>&1
grep -E "Learning iteration 9/|finished|Traceback" logs/final/train_rgbd_check.log | tail -2

echo "[final] benchmark"
rm -f logs/benchmark/benchmark.jsonl
$PY scripts/tools/benchmark.py --headless --task Quadsim-Hover-v0 --num_envs 256 1024 4096 8192 > logs/final/bench1.log 2>&1
$PY scripts/tools/benchmark.py --headless --task Quadsim-Nav-Depth-v0 --num_envs 256 1024 2048 > logs/final/bench2.log 2>&1
$PY scripts/tools/benchmark.py --headless --enable_cameras --task Quadsim-Nav-Depth-Render-v0 --num_envs 64 128 256 > logs/final/bench3.log 2>&1
$PY scripts/tools/benchmark.py --headless --enable_cameras --task Quadsim-Nav-RGBD-v0 --num_envs 64 128 > logs/final/bench4.log 2>&1
grep -h "\[BENCH\]" logs/final/bench*.log | cut -c1-160

echo "[final] M2 regression (with Isaac Lab patch applied via quadsim.tasks import)"
$PY scripts/validation/m2_dynamics_validation.py --headless > logs/final/m2.log 2>&1
grep -o '"all_pass": [a-z]*' logs/validation/m2_dynamics_x500.json
echo "[final] videos of trained policies"
HOVER=$(ls -d logs/rsl_rl/quadsim_hover/*_ctbr_s42 | tail -1)
$PY scripts/tools/view.py --headless --enable_cameras --video --task Quadsim-Hover-v0 --checkpoint "$HOVER" \
    --steps 400 --num_envs 4 > logs/final/video_hover.log 2>&1
$PY scripts/tools/view.py --headless --enable_cameras --video --task Quadsim-Nav-Depth-Render-v0 --checkpoint "$DEPTH" \
    --steps 600 --num_envs 4 env.curriculum=False > logs/final/video_nav.log 2>&1
ls -la outputs/videos

echo "[final] GUI viewer (opens a window for ~1 min)"
DISPLAY="${DISPLAY:-:0}" $PY scripts/tools/view.py --task Quadsim-Nav-Range-v0 --checkpoint "$RANGE" --steps 500 --num_envs 4 \
    env.curriculum=False > logs/final/gui.log 2>&1
echo "gui exit=$? init_failures=$(grep -c 'Failed to get rigid body' logs/final/gui.log) tracebacks=$(grep -c '^Traceback' logs/final/gui.log)"
echo "[final] done"
