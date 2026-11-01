#!/usr/bin/env bash
# Reproducible native PX4 SITL + ROS 2 setup used for the quadsim <-> PX4 SITL cross-check (docs/PROGRESS.md).
#
# Host prerequisites (verified on Ubuntu 24.04): ROS 2 Jazzy, Gazebo Harmonic (gz-sim8 dev packages),
# Micro-XRCE-DDS Agent v2.x (`MicroXRCEAgent`), cmake/ninja/gcc, git. Network access goes through the shell's
# HTTP(S)_PROXY (pip and git honour it); nothing system-wide is changed.
#
# What it does
#   1. clones PX4-Autopilot v1.16.2 and px4_msgs (msg definitions copied from that exact PX4 tree)
#   2. SITL-only instrumentation: exports the gz ground-truth topics over uXRCE-DDS (no effect on control)
#   3. builds px4_sitl_default (Python deps in an isolated venv) and the ROS 2 workspace
#      (px4_msgs + quadsim_px4_bridge) with a venv that adds onnxruntime to the system Python of ROS 2
#
# Usage: bash ros2/px4_sitl/setup_px4_sitl.sh [THIRD_PARTY_DIR]   (default: <repo>/third_party, git-ignored)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TP="${1:-$REPO/third_party}"
PX4_TAG=v1.16.2
mkdir -p "$TP" && cd "$TP"

if [ ! -d PX4-Autopilot ]; then
  git clone --depth 1 --branch "$PX4_TAG" --recurse-submodules --shallow-submodules -j 8 \
    https://github.com/PX4/PX4-Autopilot.git
fi
# the shallow NuttX submodule has no tags, but px_update_git_header.py needs one (version string only)
NX=PX4-Autopilot/platforms/nuttx/NuttX/nuttx
git -C "$NX" tag | grep -q '^nuttx-' || git -C "$NX" fetch -q --depth 1 origin tag nuttx-11.0.0
YAML=PX4-Autopilot/src/modules/uxrce_dds_client/dds_topics.yaml
if ! grep -q vehicle_local_position_groundtruth "$YAML"; then
  python3 - "$YAML" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
add = """
  # quadsim SITL instrumentation: simulator ground truth (gz bridge), used only for evaluation
  - topic: /fmu/out/vehicle_local_position_groundtruth
    type: px4_msgs::msg::VehicleLocalPosition

  - topic: /fmu/out/vehicle_attitude_groundtruth
    type: px4_msgs::msg::VehicleAttitude

  - topic: /fmu/out/vehicle_angular_velocity_groundtruth
    type: px4_msgs::msg::VehicleAngularVelocity
"""
s = s.replace("\n# Create uORB::Publication\nsubscriptions:", add + "\n# Create uORB::Publication\nsubscriptions:", 1)
assert "vehicle_local_position_groundtruth" in s, "dds_topics.yaml layout changed"
open(p, "w").write(s)
PY
fi

[ -d px4-venv ] || { /usr/bin/python3 -m venv px4-venv; ./px4-venv/bin/pip install -q -r PX4-Autopilot/Tools/setup/requirements.txt; }
(cd PX4-Autopilot && PATH="$TP/px4-venv/bin:$PATH" make px4_sitl_default)

WS="$TP/px4_ros_ws"
mkdir -p "$WS/src"
if [ ! -d "$WS/src/px4_msgs" ]; then
  git clone --depth 1 --branch release/1.16 https://github.com/PX4/px4_msgs.git "$WS/src/px4_msgs"
fi
# message definitions must match the flashed PX4 exactly -> copy them from the PX4 tree
rm -f "$WS"/src/px4_msgs/msg/*.msg "$WS"/src/px4_msgs/srv/*.srv
cp PX4-Autopilot/msg/*.msg PX4-Autopilot/msg/versioned/*.msg "$WS/src/px4_msgs/msg/"
cp PX4-Autopilot/srv/*.srv "$WS/src/px4_msgs/srv/"
ln -sfn "$REPO/ros2/quadsim_px4_bridge" "$WS/src/quadsim_px4_bridge"
[ -d "$WS/venv" ] || { /usr/bin/python3 -m venv --system-site-packages "$WS/venv"; "$WS/venv/bin/pip" install -q onnxruntime pyulog; }
set +u; source /opt/ros/jazzy/setup.bash; set -u
(cd "$WS" && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release)
echo "done: PX4 $PX4_TAG SITL in $TP/PX4-Autopilot, ROS 2 workspace in $WS"
