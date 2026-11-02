#!/usr/bin/env bash
# Reproducible native installation of the quadsim platform.
#   Isaac Sim 5.1.0 (pip) + Isaac Lab v2.3.2 + quadsim, Python 3.11, CUDA 12.8 wheels.
# Usage: bash scripts/setup_env.sh [env_name]   (default: quadsim)
set -euo pipefail
ENV_NAME="${1:-quadsim}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_TAG="v2.3.2"
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"

if ! "$CONDA" env list | grep -qE "^${ENV_NAME}\s"; then
  "$CONDA" create -y -n "$ENV_NAME" python=3.11
fi
PY="$("$CONDA" run -n "$ENV_NAME" which python)"
C="$ROOT/env/constraints.txt"

"$PY" -m pip install --upgrade pip "setuptools<80" wheel
"$PY" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -c "$C" "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Isaac Lab (third-party, never committed)
mkdir -p "$ROOT/third_party"
if [ ! -d "$ROOT/third_party/IsaacLab" ]; then
  git clone --filter=blob:none https://github.com/isaac-sim/IsaacLab.git "$ROOT/third_party/IsaacLab"
fi
git -C "$ROOT/third_party/IsaacLab" checkout -q "$ISAACLAB_TAG"
# flatdict's sdist still imports pkg_resources: build it without isolation
"$PY" -m pip install flatdict==4.0.1 --no-build-isolation
L="$ROOT/third_party/IsaacLab/source"
"$PY" -m pip install -c "$C" -e "$L/isaaclab" -e "$L/isaaclab_assets" -e "$L/isaaclab_contrib" -e "$L/isaaclab_tasks"
"$PY" -m pip install -c "$C" -e "$L/isaaclab_rl[rsl_rl,skrl]"
# re-assert pins (some extras try to pull newer torch / warp)
"$PY" -m pip install -c "$C" torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -c "$C" warp-lang skrl tensordict starlette pytest
# this project
"$PY" -m pip install -c "$C" -e "$ROOT/source/quadsim[dev]"
# Isaac Sim pins packaging==23.0; tensordict's pyvers asks for >=25 but works with 23.0. Restored last, after
# everything that resolves tensordict (pip reports the known conflict and installs it).
"$PY" -m pip install packaging==23.0
echo "[setup] done. Activate with: conda activate $ENV_NAME ; export OMNI_KIT_ACCEPT_EULA=YES"
