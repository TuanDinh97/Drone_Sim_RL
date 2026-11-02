#!/usr/bin/env bash
# Optional: isolated environment with LeRobot (dataset format v3.0) for exporting / validating quadsim demonstrations.
# Built on top of the quadsim conda env (--system-site-packages: reuses its torch 2.7 / torchvision) but with its own
# NumPy 2 / pyarrow / datasets, so the simulator environment is never modified.
#   bash scripts/il/setup_lerobot_env.sh             # datasets: export / validation
#   bash scripts/il/setup_lerobot_env.sh --policies  # + training and serving LeRobot policies (SmolVLA, ACT, ...)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
V=third_party/lerobot-venv
[ -x $V/bin/python ] || $PY -m venv --system-site-packages $V
C=$(mktemp); printf "torch==2.7.0\ntorchvision==0.22.0\nhuggingface-hub<0.36\n" > "$C"
# lerobot's full dependency set (robots, wandb, rerun, ...) is not needed for datasets: install it without deps
$V/bin/pip install --no-deps "lerobot==0.4.4"
$V/bin/pip install -c "$C" "numpy>=2,<2.3" h5py "pyarrow>=21" pandas "datasets>=4.0,<5" "av>=15,<16" jsonlines \
    deepdiff termcolor "draccus==0.10.0" accelerate
if [ "${1:-}" = "--policies" ]; then
  # lerobot.policies imports every policy family (groot needs diffusers) and the robot stack (pyserial); SmolVLA needs
  # num2words; transformers, msgpack and websockets come from the quadsim env. Versions within lerobot 0.4.4's ranges.
  $V/bin/pip install -c "$C" "numpy>=2,<2.3" "diffusers>=0.27.2,<0.36" "huggingface-hub>=0.34.2,<0.36" \
      "pyserial>=3.5,<4" "num2words>=0.5.14,<0.6"
  $V/bin/python -c "import lerobot.scripts.lerobot_train, lerobot.policies.smolvla.modeling_smolvla; print('LeRobot policies ok')"
fi
$V/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION; print('LeRobot dataset', CODEBASE_VERSION)"
