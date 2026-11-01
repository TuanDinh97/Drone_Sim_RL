"""Gymnasium registration of the quadsim Isaac Lab tasks.

Importing this module requires a running Isaac Sim ``SimulationApp``.
"""

import gymnasium as gym

from isaaclab.utils import configclass

from quadsim.isaac.patches import apply_isaaclab_patches

apply_isaaclab_patches()

from .base import QuadrotorEnv, QuadrotorEnvCfg
from .hover.hover_env import HoverEnv, HoverEnvCfg
from .navigation.nav_env import NavEnv, NavEnvCfg
from .tracking.tracking_env import TrackingEnv, TrackingEnvCfg
from .waypoint.waypoint_env import WaypointEnv, WaypointEnvCfg

_AG = "quadsim.tasks.agents"


@configclass
class HoverMotorEnvCfg(HoverEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.control.mode = "motor"


@configclass
class NavDepthEnvCfg(NavEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.vision.mode = "depth"
        self.vision.source = "raycast"


@configclass
class NavDepthRenderEnvCfg(NavEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.vision.mode = "depth"
        self.vision.source = "render"
        self.scene.num_envs = 128


@configclass
class NavRgbEnvCfg(NavDepthRenderEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.vision.mode = "rgb"


@configclass
class NavRgbdEnvCfg(NavDepthRenderEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.vision.mode = "rgbd"


_TASKS = {
    "Quadsim-Hover-v0": (HoverEnv, HoverEnvCfg, "HoverPPORunnerCfg"),
    "Quadsim-Hover-Motor-v0": (HoverEnv, HoverMotorEnvCfg, "HoverPPORunnerCfg"),
    "Quadsim-Track-v0": (TrackingEnv, TrackingEnvCfg, "TrackPPORunnerCfg"),
    "Quadsim-Waypoint-v0": (WaypointEnv, WaypointEnvCfg, "WaypointPPORunnerCfg"),
    "Quadsim-Nav-Range-v0": (NavEnv, NavEnvCfg, "NavRangePPORunnerCfg"),
    "Quadsim-Nav-Depth-v0": (NavEnv, NavDepthEnvCfg, "NavVisionPPORunnerCfg"),
    "Quadsim-Nav-Depth-Render-v0": (NavEnv, NavDepthRenderEnvCfg, "NavVisionPPORunnerCfg"),
    "Quadsim-Nav-RGB-v0": (NavEnv, NavRgbEnvCfg, "NavVisionPPORunnerCfg"),
    "Quadsim-Nav-RGBD-v0": (NavEnv, NavRgbdEnvCfg, "NavVisionPPORunnerCfg"),
}

for _name, (_env, _cfg, _agent) in _TASKS.items():
    gym.register(
        id=_name,
        entry_point=f"{_env.__module__}:{_env.__name__}",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:{_cfg.__name__}" if _cfg.__module__ == __name__
            else f"{_cfg.__module__}:{_cfg.__name__}",
            "rsl_rl_cfg_entry_point": f"{_AG}:{_agent}",
        },
    )
