#!/usr/bin/env bash
# One PX4 SITL experiment: uXRCE-DDS agent + PX4 v1.16 SITL (gz x500, headless, fresh parameters)
# + quadsim_px4_bridge (policy) + scripts/sitl/sitl_scenario.py, then teardown. Build first with
# ros2/px4_sitl/setup_px4_sitl.sh.
#
# Usage: bash scripts/sitl/run_sitl.sh <out.npz> [sitl_scenario.py args...]
# Env:   TP          third_party dir with PX4-Autopilot and px4_ros_ws (default <repo>/third_party)
#        POLICY_DIR  exported policy (policy.onnx + contract.json)
#        PX4_PARAMS  extra PX4 parameters "NAME=VALUE ...", applied after boot (e.g. rate-loop gains)
#        BRIDGE_SCRIPT  run this policy-node file instead of the installed bridge (e.g. an older version)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TP="${TP:-$REPO/third_party}"
OUT="$(realpath -m "$1")"; shift
POLICY_DIR="${POLICY_DIR:?set POLICY_DIR to <run>/exported}"
PX4_BUILD="$TP/PX4-Autopilot/build/px4_sitl_default"
WS="$TP/px4_ros_ws"
LOGDIR="$(dirname "$OUT")"; mkdir -p "$LOGDIR"
BASE="${OUT%.npz}"
# PX4's gz scripts must use the apt Gazebo Harmonic tools, not ROS 2's vendored gz on PATH
PX4_PATH="$(echo "$PATH" | tr : '\n' | grep -v '^/opt/ros' | paste -sd:)"

PGIDS=()
cleanup() {
  (cd "$PX4_BUILD/rootfs" && timeout 5 ../bin/px4-shutdown >/dev/null 2>&1) || true
  for g in "${PGIDS[@]}"; do kill -TERM -- "-$g" 2>/dev/null || true; done
  sleep 2
  for g in "${PGIDS[@]}"; do kill -KILL -- "-$g" 2>/dev/null || true; done
}
trap cleanup EXIT
start() {  # start <logfile> <cmd...> in its own process group
  local log="$1"; shift
  setsid "$@" >"$log" 2>&1 </dev/null &
  PGIDS+=("$!")
}

# a surviving gz server would be re-used by the new PX4 (with the old vehicle state): refuse to start
if pgrep -f "[g]z sim .*PX4-Autopilot" >/dev/null || pgrep -x px4 >/dev/null || pgrep -x MicroXRCEAgent >/dev/null; then
  echo "a gz sim / px4 / MicroXRCEAgent process is already running; stop it first"; trap - EXIT; exit 1
fi
start "$BASE.agent.log" MicroXRCEAgent udp4 -p 8888
rm -f "$PX4_BUILD"/rootfs/parameters*.bson  # airframe defaults every run
# px4 (and the gz server it spawns) in one new process group, killed as a whole by cleanup()
start "$BASE.px4.log" env PATH="$PX4_PATH" PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500 HEADLESS=1 \
  bash -c "cd '$PX4_BUILD' && exec ./bin/px4 -d"
for _ in $(seq 1 90); do
  grep -q "Startup script returned" "$BASE.px4.log" && break; sleep 1
done
grep -q "Startup script returned successfully" "$BASE.px4.log" || { echo "PX4 failed to start"; tail "$BASE.px4.log"; exit 1; }
px4cmd() { (cd "$PX4_BUILD/rootfs" && "../bin/px4-$1" "${@:2}"); }
px4cmd param set NAV_DLL_ACT 0  # no GCS in headless SITL (offboard-loss failsafe stays active)
for kv in ${PX4_PARAMS:-}; do px4cmd param set "${kv%%=*}" "${kv#*=}"; done

set +u; source /opt/ros/jazzy/setup.bash; source "$WS/install/setup.bash"; set -u
export PYTHONPATH="$WS/venv/lib/python3.12/site-packages:$REPO/source/quadsim:${PYTHONPATH:-}"
if [ -n "${BRIDGE_SCRIPT:-}" ]; then
  start "$BASE.bridge.log" python3 "$BRIDGE_SCRIPT" --ros-args -r __node:=quadsim_policy -p policy_dir:="$POLICY_DIR"
else
  start "$BASE.bridge.log" ros2 launch quadsim_px4_bridge policy.launch.py policy_dir:="$POLICY_DIR" \
    thrust_scale:=1.0 log_path:="$BASE.bridge_steps.csv"
fi
python3 "$REPO/scripts/sitl/sitl_scenario.py" --out "$OUT" --px4_build "$PX4_BUILD" "$@" 2>&1 | tee "$BASE.scenario.log"
px4cmd param show-for-airframe >"$BASE.px4_params.txt" 2>&1 || true
(cd "$PX4_BUILD/rootfs" && timeout 5 ../bin/px4-shutdown >/dev/null 2>&1) || true
sleep 3  # logger closes the .ulg
ULG="$(ls -t "$PX4_BUILD"/rootfs/log/*/*.ulg 2>/dev/null | head -1 || true)"  # SIGPIPE-safe under pipefail
[ -n "$ULG" ] && cp "$ULG" "$BASE.ulg"
echo "[run_sitl] done: $OUT"
