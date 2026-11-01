#!/usr/bin/env bash
# Vision-language BC with a frozen pretrained CLIP text tower on the RGB LangNav data (expert + DAgger round 1),
# evaluated with training and held-out instruction phrasings. Run after scripts/il/rgb_pipeline.sh, alone on the GPU.
set -u
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/source/quadsim${PYTHONPATH:+:$PYTHONPATH}"
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
D=datasets/il
echo "[rgb-clip] $(date +%T) BC (FiLM CNN, CLIP text) on expert + DAgger-1 data"
$PY scripts/il/train_bc.py --data $D/langnav_rgb_expert.h5 $D/langnav_rgb_dagger1.h5 --name langnav_rgb_dagger1_clip \
    --images --drop last_action --epochs 20 --text_encoder stabilityai/sd-turbo:text_encoder \
    > logs/il/train_rgb_dagger1_clip.log 2>&1
tail -1 logs/il/train_rgb_dagger1_clip.log
for T in Quadsim-LangNav-RGB-v0 Quadsim-LangNav-RGB-Heldout-v0; do
  $PY scripts/rl/evaluate.py --headless --enable_cameras --num_envs 64 --task $T --controllers policy --algo bc \
      --checkpoint logs/il/langnav_rgb_dagger1_clip/model.pt --episodes 300 --out_dir logs/eval/il \
      --tag langnav_rgb_dagger1_clip 2>&1 | grep -E "\[eval\] policy:" | cut -c1-220
done
echo "[rgb-clip] $(date +%T) done"
