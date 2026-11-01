"""Example: add a custom sensor to an existing task through configuration terms (no task code changes).

A GNSS-like position fix (5 Hz, 100 ms latency, 0.3 m noise) and a downward rangefinder are appended to the
*deployable* policy observation of the hover task; the policy then learns to use them.

    python scripts/rl/train.py --headless --task Quadsim-Example-Hover-GNSS-v0 --plugins examples/custom_sensor.py

Sensor model knobs of :class:`~quadsim.tasks.base.terms.ObsTermCfg`: ``noise_std`` (white noise), ``period_steps``
(sample-and-hold rate), ``delay_steps`` (latency), ``scale``, ``clip``. The term function receives the env and may
read anything (ground truth, as a real sensor measures the true world) - add noise / rate / latency to make it
realistic, and mark privileged quantities ``privileged=True`` (critic only).
"""

import gymnasium as gym

from isaaclab.utils import configclass

from quadsim.tasks import mdp
from quadsim.tasks.base.terms import ObsTermCfg
from quadsim.tasks.hover.hover_env import HoverEnvCfg


def gnss_position(env, origin_offset=(0.0, 0.0, 0.0)):
    """True position in the local frame (the GNSS receiver measures it; noise/rate/latency come from the term)."""
    import torch

    return env.local_pos() + torch.tensor(origin_offset, device=env.device)


@configclass
class HoverGnssEnvCfg(HoverEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.terms.observations["gnss"] = ObsTermCfg(func=gnss_position, noise_std=0.3, period_steps=10,
                                                     delay_steps=5)  # 5 Hz, 100 ms, 0.3 m
        self.terms.observations["range_down"] = ObsTermCfg(func=mdp.rangefinder_down, params={"max_range": 6.0},
                                                           noise_std=0.02, period_steps=2)


gym.register(id="Quadsim-Example-Hover-GNSS-v0", entry_point="quadsim.tasks.hover.hover_env:HoverEnv",
             disable_env_checker=True,
             kwargs={"env_cfg_entry_point": HoverGnssEnvCfg,
                     "rsl_rl_cfg_entry_point": "quadsim.tasks.agents:HoverPPORunnerCfg"})


# a 3-D LiDAR (16 x 4 rays, 10 Hz, 20 ms latency, 2 cm noise) on the obstacle-navigation task
from quadsim.tasks.navigation.nav_env import NavEnv, NavEnvCfg  # noqa: E402


@configclass
class NavLidarEnvCfg(NavEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.terms.observations["lidar"] = ObsTermCfg(
            func=mdp.lidar_scan, params={"num_h": 16, "num_v": 4, "v_fov_deg": (-15.0, 15.0), "max_range": 10.0},
            noise_std=0.002, period_steps=5, delay_steps=1, clip=(0.0, 1.0))


gym.register(id="Quadsim-Example-Nav-Lidar-v0", entry_point=NavEnv, disable_env_checker=True,
             kwargs={"env_cfg_entry_point": NavLidarEnvCfg,
                     "rsl_rl_cfg_entry_point": "quadsim.tasks.agents:NavRangePPORunnerCfg"})
