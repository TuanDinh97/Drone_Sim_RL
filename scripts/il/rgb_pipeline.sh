#!/usr/bin/env bash
# Vision-language imitation pipeline on Quadsim-LangNav-RGB (rendered camera; run alone on an 8 GB GPU):
# expert demos -> FiLM-CNN BC -> closed-loop eval (train / held-out phrasing) -> DAgger round -> eval -> VLA adapter
# reference -> LeRobot export (+ verification with LeRobot's loader).
set -u
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/source/quadsim${PYTHONPATH:+:$PYTHONPATH}"
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
CAM="--headless --enable_cameras --num_envs 64"
D=datasets/il
mkdir -p $D logs/il
log() { echo "[rgb] $(date +%T) $*"; }
ev() {  # task ckpt tag [algo]
  $PY scripts/rl/evaluate.py $CAM --task "$1" --controllers policy --algo "${4:-bc}" --checkpoint "$2" --episodes 300 \
      --out_dir logs/eval/il --tag "$3" 2>&1 | grep -E "\[eval\] policy:" | cut -c1-220; }
log "expert demos"
$PY scripts/il/collect_demos.py $CAM --task Quadsim-LangNav-RGB-v0 --expert langnav_expert --episodes 2000 \
    --out $D/langnav_rgb_expert.h5 > logs/il/collect_rgb.log 2>&1
grep -E '"episodes"|transitions|success_rate' logs/il/collect_rgb.log | tr '\n' ' '; echo
log "BC (FiLM CNN + instruction, no last action)"
$PY scripts/il/train_bc.py --data $D/langnav_rgb_expert.h5 --name langnav_rgb_bc --images --drop last_action \
    --epochs 20 > logs/il/train_rgb_bc.log 2>&1; tail -1 logs/il/train_rgb_bc.log
ev Quadsim-LangNav-RGB-v0 logs/il/langnav_rgb_bc/model.pt langnav_rgb_bc
ev Quadsim-LangNav-RGB-Heldout-v0 logs/il/langnav_rgb_bc/model.pt langnav_rgb_bc
log "DAgger round 1"
$PY scripts/il/collect_demos.py $CAM --task Quadsim-LangNav-RGB-v0 --expert langnav_expert \
    --executor logs/il/langnav_rgb_bc/model.pt --keep all --episodes 1000 --seed 1 \
    --out $D/langnav_rgb_dagger1.h5 > logs/il/collect_rgb_dagger1.log 2>&1
grep -E '"episodes"|transitions' logs/il/collect_rgb_dagger1.log | tr '\n' ' '; echo
$PY scripts/il/train_bc.py --data $D/langnav_rgb_expert.h5 $D/langnav_rgb_dagger1.h5 --name langnav_rgb_dagger1 \
    --images --drop last_action --epochs 20 > logs/il/train_rgb_dagger1.log 2>&1; tail -1 logs/il/train_rgb_dagger1.log
ev Quadsim-LangNav-RGB-v0 logs/il/langnav_rgb_dagger1/model.pt langnav_rgb_dagger1
ev Quadsim-LangNav-RGB-Heldout-v0 logs/il/langnav_rgb_dagger1/model.pt langnav_rgb_dagger1
log "VLA adapter reference (model-free colour servo)"
ev Quadsim-LangNav-RGB-v0 examples/vla_color_servo.py:make_adapter vla_color_servo vla
log "expert on the RGB task"
$PY scripts/rl/evaluate.py $CAM --task Quadsim-LangNav-RGB-v0 --controllers langnav_expert --episodes 300 \
    --out_dir logs/eval/il --tag expert 2>&1 | grep -E "\[eval\] langnav" | cut -c1-200
log "LeRobot export"
third_party/lerobot-venv/bin/python scripts/il/export_lerobot.py $D/langnav_rgb_expert.h5 --repo_id quadsim/langnav_rgb \
    --out datasets/lerobot/langnav_rgb --max_episodes 300 --overwrite > logs/il/export_lerobot_rgb.log 2>&1
grep -E '"pass"|frames|images_identical' logs/il/export_lerobot_rgb.log | tr '\n' ' '; echo
log "done"
