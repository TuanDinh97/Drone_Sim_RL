#!/usr/bin/env bash
# Training queue of the framework milestones (F2 algorithms, F3 tasks). One job at a time on the 8 GB GPU.
#   bash scripts/rl/queue_framework.sh <job> [<job> ...]
#   jobs: sac td3 td3b safe hexa hexa_motor race swarm land accel ars
set -u
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/source/quadsim${PYTHONPATH:+:$PYTHONPATH}"
PY="${QUADSIM_PY:-$HOME/miniconda3/envs/quadsim/bin/python}"
T="$PY scripts/rl/train.py --headless"
for job in "$@"; do
  echo "[queue] $(date +%T) start $job"
  case $job in
    sac)        $T --task Quadsim-Hover-v0 --algo sac --num_envs 1024 --max_iterations 15000 --run_name sac_s42 ;;
    td3)        $T --task Quadsim-Hover-v0 --algo td3 --num_envs 1024 --max_iterations 15000 --run_name td3_s42 ;;
    safe)       $T --task Quadsim-Waypoint-Safe-v0 --max_iterations 1500 --run_name safe_v2ms_s42 ;;
    hexa)       $T --task Quadsim-Hover-Hexa-v0 --max_iterations 600 --run_name hexa_ctbr_s42 ;;
    hexa_motor) $T --task Quadsim-Hover-Hexa-Motor-v0 --max_iterations 1000 --run_name hexa_motor_s42 ;;
    race)       $T --task Quadsim-Race-v0 --max_iterations 2000 --run_name race_s42 ;;
    swarm)      $T --task Quadsim-Swarm-v0 --max_iterations 1500 --run_name swarm_s42 ;;
    land)       $T --task Quadsim-Example-Land-v0 --plugins examples/custom_task.py --max_iterations 400 --run_name land_s42 ;;
    accel)      $T --task Quadsim-Hover-v0 --plugins examples/custom_action_interface.py --max_iterations 600 \
                   --run_name accel_s42 env.control.mode=acceleration ;;
    ars)        $T --task Quadsim-Hover-v0 --algo ars --plugins examples/custom_algorithm.py --max_iterations 150 \
                   --run_name ars_s42 ;;
    td3b)       $T --task Quadsim-Hover-v0 --algo td3 --num_envs 1024 --max_iterations 15000 --run_name td3b_s42 \
                   agent.explore_std_range="(0.05,0.15)" agent.updates_per_step=8 agent.reward_scale=10.0 ;;
    *) echo "unknown job $job" ;;
  esac > "logs/train_fw_${job}.log" 2>&1
  echo "[queue] $(date +%T) done $job: $(grep -h 'finished in' logs/train_fw_${job}.log | tail -1)"
done
