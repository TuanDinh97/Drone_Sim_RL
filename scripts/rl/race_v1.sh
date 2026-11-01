#!/usr/bin/env bash
# Racing generalization follow-up: train on circuits of both turning directions (Quadsim-Race-v1) and re-run the race
# benchmark (the v0 policy, trained on counter-clockwise circuits only, fails on the figure-8).
set -u
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/source/quadsim${PYTHONPATH:+:$PYTHONPATH}"
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
echo "[race_v1] $(date +%T) train"
$PY scripts/rl/train.py --headless --task Quadsim-Race-v1 --max_iterations 2000 --run_name race_v1_s42 \
    > logs/train_fw_race_v1.log 2>&1
grep -h "finished in" logs/train_fw_race_v1.log
R=$(ls -d logs/rsl_rl/quadsim_race/*_race_v1_s42 | tail -1)
R0=$(ls -d logs/rsl_rl/quadsim_race/*_race_s42 | tail -1)
echo "[race_v1] $(date +%T) benchmark"
$PY scripts/benchmark/run.py --suite race --policy ppo_v1=$R --baselines --tag final > logs/bench_race_v1.log 2>&1
sed -n '/^## success/,/^## lap_time/p' logs/benchmark_suites/race/final/summary.md
$PY scripts/tools/collect_framework_results.py > /dev/null
$PY scripts/tools/report_tables.py > /dev/null
echo "[race_v1] $(date +%T) done"
