"""Gymnasium registration of the quadsim Isaac Lab tasks.

Importing this module requires a running Isaac Sim ``SimulationApp``.
"""

import gymnasium as gym

from isaaclab.utils import configclass

from quadsim.isaac.patches import apply_isaaclab_patches

apply_isaaclab_patches()

from . import mdp
from .base import QuadrotorEnv, QuadrotorEnvCfg
from .base.terms import CostTermCfg, ObsTermCfg, RewardTermCfg
from .hover.hover_env import HoverEnv, HoverEnvCfg
from .navigation.nav_env import NavEnv, NavEnvCfg
from .langnav.langnav_env import LangNavEnv, LangNavEnvCfg, LangNavExpert
from .racing.race_env import RaceEnv, RaceEnvCfg
from .swarm.swarm_env import SwarmEnv, SwarmEnvCfg, SwarmPotentialBaseline
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


@configclass
class WaypointSpeedCostEnvCfg(WaypointEnvCfg):
    """Waypoint racing with a speed constraint |v| <= speed_limit, logged as a cost (no constraint enforcement):
    used to measure the constraint violation of unconstrained policies."""

    speed_limit: float = 2.0
    """[m/s], read at run time by the cost term (so ``env.speed_limit=...`` overrides work)"""

    def __post_init__(self):
        super().__post_init__()
        self.terms.costs["overspeed"] = CostTermCfg(func=mdp.speed_violation)


@configclass
class WaypointSafeEnvCfg(WaypointSpeedCostEnvCfg):
    """Constrained RL: PPO-Lagrangian (PID multiplier) keeps the expected over-speed time below ``cost_limit``."""

    def __post_init__(self):
        super().__post_init__()
        self.safety.lagrangian = True
        self.safety.cost_limit = 0.1  # s of over-speed flight per episode
        self.safety.ki = 0.05
        self.safety.kp = 0.5


@configclass
class WaypointSafeV1EnvCfg(WaypointSafeEnvCfg):
    """Waypoint-Safe with a constraint warm-up: the multiplier stays at 0 for the first 300 PPO iterations (32 steps
    each), so the policy learns to finish the course before the speed constraint is enforced (with v0 one of three
    seeds never learned to finish: the penalty rose first)."""

    def __post_init__(self):
        super().__post_init__()
        self.safety.warmup_steps = 300 * 32


@configclass
class HoverSensorsEnvCfg(HoverEnvCfg):
    """Configuration-only task extension: extra deployable sensors (downward rangefinder at 25 Hz with 20 ms latency,
    barometer) and an energy-efficiency reward term, without touching the hover task's code."""

    def __post_init__(self):
        super().__post_init__()
        self.terms.observations["range_down"] = ObsTermCfg(
            func=mdp.rangefinder_down, params={"max_range": 8.0}, noise_std=0.02, period_steps=2, delay_steps=1)
        self.terms.observations["baro"] = ObsTermCfg(func=mdp.baro_altitude, noise_std=0.0)
        self.terms.rewards["power"] = RewardTermCfg(func=mdp.rotor_power, weight=-0.05)


@configclass
class HoverHexaEnvCfg(HoverEnvCfg):
    """Hover with a 6-rotor vehicle (generic X-hexarotor preset)."""

    def __post_init__(self):
        super().__post_init__()
        self.dynamics.preset = "hexa_x"


@configclass
class HoverHexaMotorEnvCfg(HoverHexaEnvCfg):
    """Hexarotor with direct per-rotor thrust commands (6-D action)."""

    def __post_init__(self):
        super().__post_init__()
        self.control.mode = "motor"


@configclass
class RaceEllipseEnvCfg(RaceEnvCfg):
    """Fixed elliptic reference circuit (evaluation of policies trained on random circuits)."""

    def __post_init__(self):
        super().__post_init__()
        self.track = "ellipse"


@configclass
class RaceV1EnvCfg(RaceEnvCfg):
    """Random circuits in both turning directions (the v0 policy, trained on counter-clockwise circuits only, fails
    on the figure-8)."""

    def __post_init__(self):
        super().__post_init__()
        self.random_direction = True


@configclass
class RaceFigure8EnvCfg(RaceEnvCfg):
    """Fixed figure-8 circuit with a crossing (unseen layout family for random-circuit policies)."""

    def __post_init__(self):
        super().__post_init__()
        self.track = "figure8"
        self.num_gates = 8


@configclass
class LangNavHeldoutEnvCfg(LangNavEnvCfg):
    """Held-out instruction phrasings (generalization of the language grounding)."""

    def __post_init__(self):
        super().__post_init__()
        self.template_split = "heldout"


@configclass
class LangNavRgbEnvCfg(LangNavEnvCfg):
    """Vision-language variant: RGB camera + instruction, no landmark map (colours must be grounded in pixels)."""

    def __post_init__(self):
        super().__post_init__()
        self.symbolic = False
        self.vision.mode = "rgb"
        self.scene.num_envs = 128


@configclass
class LangNavRgbHeldoutEnvCfg(LangNavRgbEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.template_split = "heldout"


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
    "Quadsim-Waypoint-SpeedCost-v0": (WaypointEnv, WaypointSpeedCostEnvCfg, "WaypointPPORunnerCfg"),
    "Quadsim-Waypoint-Safe-v0": (WaypointEnv, WaypointSafeEnvCfg, "WaypointSafePPORunnerCfg"),
    "Quadsim-Waypoint-Safe-v1": (WaypointEnv, WaypointSafeV1EnvCfg, "WaypointSafePPORunnerCfg"),
    "Quadsim-Hover-Sensors-v0": (HoverEnv, HoverSensorsEnvCfg, "HoverPPORunnerCfg"),
    "Quadsim-Hover-Hexa-v0": (HoverEnv, HoverHexaEnvCfg, "HoverHexaPPORunnerCfg"),
    "Quadsim-Hover-Hexa-Motor-v0": (HoverEnv, HoverHexaMotorEnvCfg, "HoverHexaPPORunnerCfg"),
    "Quadsim-Race-v0": (RaceEnv, RaceEnvCfg, "RacePPORunnerCfg"),
    "Quadsim-Race-v1": (RaceEnv, RaceV1EnvCfg, "RacePPORunnerCfg"),
    "Quadsim-Race-Ellipse-v0": (RaceEnv, RaceEllipseEnvCfg, "RacePPORunnerCfg"),
    "Quadsim-Race-Figure8-v0": (RaceEnv, RaceFigure8EnvCfg, "RacePPORunnerCfg"),
    "Quadsim-Swarm-v0": (SwarmEnv, SwarmEnvCfg, "SwarmPPORunnerCfg"),
    "Quadsim-LangNav-v0": (LangNavEnv, LangNavEnvCfg, "LangNavPPORunnerCfg"),
    "Quadsim-LangNav-Heldout-v0": (LangNavEnv, LangNavHeldoutEnvCfg, "LangNavPPORunnerCfg"),
    "Quadsim-LangNav-RGB-v0": (LangNavEnv, LangNavRgbEnvCfg, "LangNavPPORunnerCfg"),
    "Quadsim-LangNav-RGB-Heldout-v0": (LangNavEnv, LangNavRgbHeldoutEnvCfg, "LangNavPPORunnerCfg"),
}

from quadsim.registry import BASELINES as _BASELINES  # noqa: E402

from . import baselines as _baselines  # noqa: E402,F401  (registers cascade / geometric / potential_field)

_BASELINES.register("swarm_potential", SwarmPotentialBaseline, overwrite=True)
_BASELINES.register("langnav_expert", LangNavExpert, overwrite=True)

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
