"""rsl_rl PPO configurations for the quadsim tasks (asymmetric actor-critic)."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 100
    experiment_name = "quadsim"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HoverPPORunnerCfg(QuadPPORunnerCfg):
    experiment_name = "quadsim_hover"
    max_iterations = 600


@configclass
class TrackPPORunnerCfg(QuadPPORunnerCfg):
    experiment_name = "quadsim_track"
    max_iterations = 1500


@configclass
class WaypointPPORunnerCfg(QuadPPORunnerCfg):
    experiment_name = "quadsim_waypoint"
    max_iterations = 1500


@configclass
class NavRangePPORunnerCfg(QuadPPORunnerCfg):
    experiment_name = "quadsim_nav_range"
    max_iterations = 2000


@configclass
class NavVisionPPORunnerCfg(QuadPPORunnerCfg):
    """Vision policies: CNN encoder on the ``image`` group fused with the state vector."""

    experiment_name = "quadsim_nav_depth"
    max_iterations = 2000
    num_steps_per_env = 32
    obs_groups = {"policy": ["policy", "image"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticCNN",
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # CNN back-prop on 2048 envs x 32 steps: 16 mini-batches of 4096 images keep activations < 0.5 GB (8 GB GPU)
        self.algorithm.num_mini_batches = 16


@configclass
class WaypointSafePPORunnerCfg(WaypointPPORunnerCfg):
    experiment_name = "quadsim_waypoint_safe"


@configclass
class HoverHexaPPORunnerCfg(HoverPPORunnerCfg):
    experiment_name = "quadsim_hover_hexa"


@configclass
class RacePPORunnerCfg(QuadPPORunnerCfg):
    experiment_name = "quadsim_race"
    max_iterations = 2000


@configclass
class SwarmPPORunnerCfg(QuadPPORunnerCfg):
    experiment_name = "quadsim_swarm"
    max_iterations = 1500


@configclass
class LangNavPPORunnerCfg(QuadPPORunnerCfg):
    experiment_name = "quadsim_langnav"
    max_iterations = 1000
