"""Policy networks compatible with rsl_rl 3.1.x (TensorDict observations, ``obs_groups``).

``ActorCriticCNN``: the actor encodes image groups (N, C, H, W) with a small CNN and fuses the
features with the normalized 1-D groups; the critic is an MLP on privileged 1-D groups only
(asymmetric actor-critic, so the expensive image path is not duplicated).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization


class DepthEncoder(nn.Module):
    """Conv encoder for low-resolution depth / RGB(-D) images (~64x48)."""

    def __init__(self, in_channels: int, height: int, width: int, feature_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2), nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, in_channels, height, width)).shape[-1]
        self.head = nn.Sequential(nn.Linear(n, feature_dim), nn.ELU())
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


class ActorCriticCNN(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        image_feature_dim: int = 128,
        **kwargs: dict[str, Any],
    ) -> None:
        nn.Module.__init__(self)
        self.obs_groups = obs_groups
        self.state_dependent_std = False
        self.noise_std_type = noise_std_type
        self.actor_groups_1d = [g for g in obs_groups["policy"] if obs[g].dim() == 2]
        self.actor_groups_img = [g for g in obs_groups["policy"] if obs[g].dim() == 4]
        assert len(self.actor_groups_img) == 1, "ActorCriticCNN expects exactly one image group"
        img = obs[self.actor_groups_img[0]]
        c, h, w = img.shape[1:]
        num_1d = sum(obs[g].shape[-1] for g in self.actor_groups_1d)
        num_critic = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        for g in obs_groups["critic"]:
            assert obs[g].dim() == 2, "critic groups must be 1-D"

        self.encoder = DepthEncoder(c, h, w, image_feature_dim)
        self.actor = MLP(num_1d + image_feature_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic, 1, critic_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_1d) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic) if critic_obs_normalization else nn.Identity()
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        else:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        self.distribution = None
        Normal.set_default_validate_args(False)
        print(f"ActorCriticCNN: image {tuple(img.shape[1:])}, 1d actor obs {num_1d}, critic obs {num_critic}")

    # ------------------------------------------------------------------ actor path
    def _actor_features(self, obs: TensorDict) -> torch.Tensor:
        x1d = torch.cat([obs[g] for g in self.actor_groups_1d], dim=-1)
        x1d = self.actor_obs_normalizer(x1d)
        feat = self.encoder(obs[self.actor_groups_img[0]])
        return torch.cat([x1d, feat], dim=-1)

    def _dist(self, mean: torch.Tensor):
        std = self.std.expand_as(mean) if self.noise_std_type == "scalar" else torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        self._dist(self.actor(self._actor_features(obs)))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.actor(self._actor_features(obs))

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:  # used only for 1-D bookkeeping
        return torch.cat([obs[g] for g in self.actor_groups_1d], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))


def register_rsl_rl_modules():
    """Make custom classes resolvable by rsl_rl's ``eval(class_name)`` in the runner module."""
    import rsl_rl.runners.on_policy_runner as opr

    opr.ActorCriticCNN = ActorCriticCNN
