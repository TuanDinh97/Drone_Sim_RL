#!/usr/bin/env bash
# Run training jobs sequentially, detached from the terminal (survives session/usage interruptions).
# Usage: nohup bash scripts/rl/train_queue.sh "<task> <run_name> [extra args...]" ... > logs/queue.log 2>&1 &
# Each job logs to logs/train_<run_name>.log; checkpoints in logs/rsl_rl/<experiment>/<timestamp>_<run_name>/.
# To resume an interrupted job: python scripts/rl/train.py --task <task> --headless --resume <run_dir>
set -u
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=YES
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
for job in "$@"; do
  read -r task run extra <<< "$job"
  echo "[queue] $(date -Is) start $task $run $extra"
  $PY scripts/rl/train.py --task "$task" --headless --run_name "$run" $extra > "logs/train_${run}.log" 2>&1
  echo "[queue] $(date -Is) done  $task $run (exit $?)"
done
