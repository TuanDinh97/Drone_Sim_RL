"""ros2 launch quadsim_px4_bridge policy.launch.py policy_dir:=<run>/exported thrust_scale:=1.23 [params_file:=...]
   [log_path:=steps.csv] [task:=track] [geofence_radius_m:=3.0]

Empty arguments keep the values of ``params_file`` (default: config/hover.yaml).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _node(context):
    overrides = {}
    if d := LaunchConfiguration("policy_dir").perform(context):
        overrides["policy_dir"] = d
    if s := LaunchConfiguration("thrust_scale").perform(context):
        overrides["thrust_scale"] = float(s)
    if lp := LaunchConfiguration("log_path").perform(context):
        overrides["log_path"] = lp
    if task := LaunchConfiguration("task").perform(context):
        overrides["task"] = task
    if fence := LaunchConfiguration("geofence_radius_m").perform(context):
        overrides["geofence_radius_m"] = float(fence)
    params = [LaunchConfiguration("params_file").perform(context)] + ([overrides] if overrides else [])
    return [Node(package="quadsim_px4_bridge", executable="policy_node", name="quadsim_policy",
                 parameters=params, output="screen")]


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("quadsim_px4_bridge"), "config", "hover.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=cfg),
        DeclareLaunchArgument("policy_dir", default_value=""),
        DeclareLaunchArgument("thrust_scale", default_value=""),
        DeclareLaunchArgument("log_path", default_value=""),
        DeclareLaunchArgument("task", default_value=""),
        DeclareLaunchArgument("geofence_radius_m", default_value=""),
        OpaqueFunction(function=_node),
    ])
