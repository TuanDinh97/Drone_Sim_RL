#!/usr/bin/env bash
# Evaluations of the framework milestones, each started as soon as its training (scripts/rl/queue_framework.sh)
# has finished; state-based evaluations may overlap the next training (2 GPU processes), rendering checks run last.
#   bash scripts/rl/framework_evals.sh logs/queue_framework_2.log
set -u
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/source/quadsim${PYTHONPATH:+:$PYTHONPATH}"
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
Q="${1:-logs/queue_framework_2.log}"
waitfor() { until grep -q "done $1" "$Q" 2>/dev/null; do sleep 30; done; echo "[evals] $(date +%T) $1 trained"; }
run() { latest=$(ls -d $1 2>/dev/null | tail -1); echo "$latest"; }
EV="$PY scripts/rl/evaluate.py --headless"

waitfor race
R=$(run "logs/rsl_rl/quadsim_race/*_race_s42")
$PY scripts/benchmark/run.py --suite race --policy ppo=$R --tag final > logs/bench_race.log 2>&1
tail -8 logs/benchmark_suites/race/final/summary.md

waitfor swarm
R=$(run "logs/rsl_rl/quadsim_swarm/*_swarm_s42")
$PY scripts/benchmark/run.py --suite swarm --policy ppo=$R --tag final > logs/bench_swarm.log 2>&1
tail -6 logs/benchmark_suites/swarm/final/summary.md

waitfor land
R=$(run "logs/rsl_rl/quadsim_example_land/*_land_s42")
$EV --task Quadsim-Example-Land-v0 --plugins examples/custom_task.py --controllers policy cascade --checkpoint $R \
    --episodes 1000 --num_envs 500 --out_dir logs/eval/vehicles --tag land_s42 2>&1 | grep "\[eval\] .*:"

waitfor accel
R=$(run "logs/rsl_rl/quadsim_hover/*_accel_s42")
$EV --task Quadsim-Hover-v0 --plugins examples/custom_action_interface.py --controllers policy --checkpoint $R \
    --episodes 1000 --num_envs 500 --out_dir logs/eval/vehicles --tag accel_s42 env.control.mode=acceleration 2>&1 \
    | grep "\[eval\] .*:"

waitfor ars
R=$(run "logs/ars/quadsim_hover/*_ars_s42")
$EV --task Quadsim-Hover-v0 --plugins examples/custom_algorithm.py --controllers policy --checkpoint $R \
    --episodes 1000 --num_envs 500 --out_dir logs/eval/algos --tag ars_s42 2>&1 | grep "\[eval\] .*:"

waitfor td3b
R=$(run "logs/td3/quadsim_hover/*_td3b_s42")
$EV --task Quadsim-Hover-v0 --controllers policy --checkpoint $R --episodes 1000 --num_envs 500 \
    --out_dir logs/eval/algos --tag td3b_s42 2>&1 | grep "\[eval\] .*:"

echo "[evals] $(date +%T) final regression + rendering example checks (GPU otherwise idle)"
bash scripts/validation/regression_evals.sh logs/eval/regress_final > /dev/null 2>&1
$PY scripts/validation/compare_evals.py logs/eval/regress_before logs/eval/regress_final | tail -1
$PY scripts/validation/examples_check.py --only vla_adapter_color_servo task_landing_baseline 2>&1 | grep "\[examples\]"
$PY scripts/tools/collect_framework_results.py
echo "[evals] $(date +%T) done"
