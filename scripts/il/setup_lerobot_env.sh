#!/usr/bin/env bash
# Optional: isolated environment with LeRobot (dataset format v3.0) for exporting / validating quadsim demonstrations.
# Built on top of the quadsim conda env (--system-site-packages: reuses its torch 2.7 / torchvision) but with its own
# NumPy 2 / pyarrow / datasets, so the simulator environment is never modified.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
V=third_party/lerobot-venv
[ -x $V/bin/python ] || $PY -m venv --system-site-packages $V
C=$(mktemp); printf "torch==2.7.0\ntorchvision==0.22.0\n" > "$C"
# lerobot's full dependency set (robots, wandb, rerun, ...) is not needed for datasets: install it without deps
$V/bin/pip install --no-deps "lerobot==0.4.4"
$V/bin/pip install -c "$C" "numpy>=2,<2.3" h5py "pyarrow>=21" pandas "datasets>=4.0,<5" "av>=15,<16" jsonlines \
    deepdiff termcolor "draccus==0.10.0" accelerate
$V/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION; print('LeRobot dataset', CODEBASE_VERSION)"
