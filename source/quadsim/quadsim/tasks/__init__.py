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


def apply_sim2real_v1(cfg):
    """Training distribution "v1" = v0 + the corrections identified in the PX4 SITL cross-check (docs/PROGRESS.md):
    EKF2-like estimate errors (per-flight bias + correlated position/velocity errors instead of mostly white noise;
    levels from SITL EKF2, widened for hardware), estimator and command latency up to 20 ms (SITL: odometry 5-16 ms
    median / 23 ms max at the policy step), randomized rate-loop gain (PX4's rate loop differs: smaller time constant
    in pitch, slower roll with the airframe's allocation geometry) and thrust-map gain around hover (the linear thrust
    map on the gz x500 has 1.6x the trained local gain)."""
    o = cfg.sensors.odometry
    o.pos_noise_std, o.pos_drift_rw, o.vel_noise_std, o.att_noise_std = 0.003, 0.0, 0.01, 0.005
    o.pos_bias_std = (0.02, 0.02, 0.04)
    o.pos_gm_std, o.pos_gm_tau, o.vel_gm_std, o.vel_gm_tau = 0.015, 2.5, 0.01, 0.5
    o.delay_steps_max = 4
    cfg.control.action_delay_steps = (0, 4)
    r = cfg.randomization
    r.rate_gain_scale, r.thrust_gain_scale, r.sensor_noise_scale = (0.6, 1.4), (0.8, 1.7), (0.5, 2.0)


@configclass
class HoverV1EnvCfg(HoverEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_sim2real_v1(self)


@configclass
class HoverV2EnvCfg(HoverEnvCfg):
    """v1 + v0-level white estimate noise (v1 policies lost robustness to white noise) + collective-rate penalty
    (v1 policies kept a lightly damped heave mode that became a limit cycle in PX4 SITL)."""

    def __post_init__(self):
        super().__post_init__()
        apply_sim2real_v1(self)
        o = self.sensors.odometry
        o.pos_noise_std, o.vel_noise_std, o.att_noise_std = 0.01, 0.03, 0.01
        self.w_collective_rate = 1.0


@configclass
class TrackingV1EnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_sim2real_v1(self)


@configclass
class WaypointV1EnvCfg(WaypointEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_sim2real_v1(self)


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
    "Quadsim-Hover-v1": (HoverEnv, HoverV1EnvCfg, "HoverPPORunnerCfg"),
    "Quadsim-Hover-v2": (HoverEnv, HoverV2EnvCfg, "HoverPPORunnerCfg"),
    "Quadsim-Track-v1": (TrackingEnv, TrackingV1EnvCfg, "TrackPPORunnerCfg"),
    "Quadsim-Waypoint-v1": (WaypointEnv, WaypointV1EnvCfg, "WaypointPPORunnerCfg"),
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
