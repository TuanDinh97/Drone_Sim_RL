from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("quadsim_px4_bridge"), "config", "hover.yaml")
    return LaunchDescription([Node(package="quadsim_px4_bridge", executable="policy_node", name="quadsim_policy",
                                   parameters=[cfg], output="screen")])
