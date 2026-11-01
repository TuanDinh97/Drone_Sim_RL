"""Example: reshape an existing task with configuration terms - extra reward, cost, termination and a curriculum.

    python scripts/rl/train.py --headless --task Quadsim-Example-Hover-Curriculum-v0 \
        --plugins examples/custom_reward_curriculum.py --max_iterations 600

* reward: energy efficiency (``mdp.rotor_power``) and an upright bonus;
* termination: leaving a 3 m box is a failure;
* curriculum: the hover task's initial tilt grows from 0.3 rad to 1.2 rad in 0.15 rad steps whenever the success
  rate of at least 256 episodes started at the current level exceeds 80 % (``mdp.attribute_curriculum`` changes
  ``cfg.init_max_tilt`` on the live config; logged as ``Curriculum/init_tilt`` in TensorBoard).
"""

import gymnasium as gym

from isaaclab.utils import configclass

from quadsim.tasks import mdp
from quadsim.tasks.base.terms import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg
from quadsim.tasks.hover.hover_env import HoverEnv, HoverEnvCfg


@configclass
class HoverCurriculumEnvCfg(HoverEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.terms.rewards["energy"] = RewardTermCfg(func=mdp.rotor_power, weight=-0.05)
        self.terms.rewards["upright"] = RewardTermCfg(func=mdp.upright, weight=0.05)
        self.terms.terminations["box"] = TerminationTermCfg(func=mdp.outside_box, params={"half_extent": (3, 3, 3.5)})
        self.terms.curricula["init_tilt"] = CurriculumTermCfg(
            func=mdp.attribute_curriculum,
            params={"attr": "init_max_tilt", "start": 0.3, "end": 1.2, "step": 0.15, "threshold": 0.8})


gym.register(id="Quadsim-Example-Hover-Curriculum-v0", entry_point=HoverEnv, disable_env_checker=True,
             kwargs={"env_cfg_entry_point": HoverCurriculumEnvCfg,
                     "rsl_rl_cfg_entry_point": "quadsim.tasks.agents:HoverPPORunnerCfg"})
