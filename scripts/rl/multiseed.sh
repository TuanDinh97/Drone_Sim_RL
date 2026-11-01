#!/usr/bin/env bash
# Multi-seed replication of the framework results that were first trained with one seed (42; BC: seed 0): racing
# v0 / v1, the multi-agent swap, PPO-Lagrangian and the LangNav imitation policies (BC seeds 1, 2 on the same
# demonstrations). Same configurations as the seed-42 runs (num_envs pinned: the hardware check had lowered the seed-42
# safe run to 3456). One training at a time; state evaluations may overlap a training, rendered-camera ones run alone.
#   bash scripts/rl/multiseed.sh train > logs/multiseed_train.log 2>&1
#   bash scripts/rl/multiseed.sh eval  > logs/multiseed_eval.log 2>&1
#   python scripts/tools/collect_multiseed.py
# The published seeds 1-2 were trained with the code of commit 7008d7d (before the session-5 review fixes; the
# racing track resampling of the review changes racing training slightly) and evaluated with the corrected evaluator.
set -u
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/source/quadsim${PYTHONPATH:+:$PYTHONPATH}"
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
T="$PY scripts/rl/train.py --headless"
EV="$PY scripts/rl/evaluate.py --headless"
D=datasets/il
CLIP="--text_encoder stabilityai/sd-turbo:text_encoder"
SYM_D2="$D/langnav_sym_expert.h5 $D/langnav_sym_dagger1.h5 $D/langnav_sym_dagger2.h5"
RGB_D1="$D/langnav_rgb_expert.h5 $D/langnav_rgb_dagger1.h5"
log() { echo "[multiseed] $(date +%T) $*"; }
run() { ls -d $1 2>/dev/null | tail -1; }
bc() { local name=$1 seed=$2; shift 2
  $PY scripts/il/train_bc.py --name ${name}_s$seed --seed $seed --drop last_action "$@" \
      > logs/il/train_ms_${name}_s$seed.log 2>&1; log "bc ${name}_s$seed: $(tail -1 logs/il/train_ms_${name}_s$seed.log)"; }
mkdir -p logs/il logs/eval/il logs/eval/safe

train() {
  for s in 1 2; do
    $T --task Quadsim-Race-v1 --num_envs 4096 --max_iterations 2000 --seed $s --run_name race_v1_s$s > logs/train_ms_race_v1_s$s.log 2>&1
    $T --task Quadsim-Race-v0 --num_envs 4096 --max_iterations 2000 --seed $s --run_name race_s$s > logs/train_ms_race_s$s.log 2>&1
    $T --task Quadsim-Swarm-v0 --num_envs 4096 --max_iterations 1500 --seed $s --run_name swarm_s$s > logs/train_ms_swarm_s$s.log 2>&1
    $T --task Quadsim-Waypoint-Safe-v0 --num_envs 3456 --max_iterations 1500 --seed $s --run_name safe_v2ms_s$s \
        > logs/train_ms_safe_s$s.log 2>&1
    log "RL seed $s trained"
  done
  for s in 1 2; do  # behaviour cloning (no simulator; ~2 GB GPU for the image models)
    bc langnav_sym_bc_v2 $s --data $D/langnav_sym_expert.h5
    bc langnav_sym_dagger2 $s --data $SYM_D2
    bc langnav_sym_dagger2_clip $s --data $SYM_D2 $CLIP
    bc langnav_rgb_bc $s --data $D/langnav_rgb_expert.h5 --images --epochs 20
    bc langnav_rgb_dagger1 $s --data $RGB_D1 --images --epochs 20
    bc langnav_rgb_dagger1_clip $s --data $RGB_D1 --images --epochs 20 $CLIP
  done
}

evaluate() {
  for s in 1 2; do
    $PY scripts/benchmark/run.py --suite race --policy ppo_s$s=$(run "logs/rsl_rl/quadsim_race/*_race_s$s") \
        --policy ppo_v1_s$s=$(run "logs/rsl_rl/quadsim_race/*_race_v1_s$s") --baselines --tag seeds
    $PY scripts/benchmark/run.py --suite swarm --policy ppo_s$s=$(run "logs/rsl_rl/quadsim_swarm/*_swarm_s$s") \
        --baselines --tag seeds
    $EV --task Quadsim-Waypoint-SpeedCost-v0 --controllers policy --episodes 500 --num_envs 250 --out_dir logs/eval/safe \
        --checkpoint $(run "logs/rsl_rl/quadsim_waypoint_safe/*_safe_v2ms_s$s") --tag safe_lagrangian_s$s
    for n in langnav_sym_bc_v2 langnav_sym_dagger2 langnav_sym_dagger2_clip; do
      for task in Quadsim-LangNav-v0 Quadsim-LangNav-Heldout-v0; do
        $EV --task $task --controllers policy --algo bc --checkpoint logs/il/${n}_s$s/model.pt --episodes 500 \
            --num_envs 250 --out_dir logs/eval/il --tag ${n}_s$s
      done
    done
    for n in langnav_rgb_bc langnav_rgb_dagger1 langnav_rgb_dagger1_clip; do  # rendered camera: alone on the GPU
      for task in Quadsim-LangNav-RGB-v0 Quadsim-LangNav-RGB-Heldout-v0; do
        $EV --enable_cameras --num_envs 64 --task $task --controllers policy --algo bc --episodes 300 \
            --checkpoint logs/il/${n}_s$s/model.pt --out_dir logs/eval/il --tag ${n}_s$s
      done
    done
    log "seed $s evaluated"
  done
}

case "${1:-}" in
  train) train ;;
  eval) evaluate ;;
  *) echo "usage: $0 train|eval"; exit 1 ;;
esac
