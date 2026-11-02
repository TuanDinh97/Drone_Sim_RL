"""Plugging external vision-language-action (VLA) / VLM policies into quadsim evaluations.

A VLA policy is anything implementing :class:`VLAAdapter` (images + instruction strings + proprioception -> normalized
actions). ``scripts/rl/evaluate.py --algo vla --checkpoint my_pkg.my_module:make_adapter`` imports the factory, calls
it with the environment and evaluates the adapter in closed loop with the same episodes, metrics and confidence
intervals as RL policies, BC policies and classical controllers. The factory may load any model (e.g. a LeRobot
policy fine-tuned on a dataset exported with ``scripts/il/export_lerobot.py``); quadsim itself has no dependency on
VLA libraries. ``examples/vla_color_servo.py`` is a minimal, model-free adapter used to test the interface.
"""

from __future__ import annotations

import importlib
from typing import Protocol

import torch

from quadsim.registry import ALGORITHMS
from quadsim.rl.algorithms import Algorithm


class VLAAdapter(Protocol):
    def reset(self, env_ids: torch.Tensor) -> None: ...

    def __call__(self, images: torch.Tensor | None, instructions: list[str], proprio: torch.Tensor) -> torch.Tensor:
        """``images`` (N, C, H, W) float in [0, 1] (or None), ``proprio`` (N, P) deployable state observation without
        task tokens; returns (N, A) actions in [-1, 1] for the environment's action interface. An adapter with
        ``wants_policy_obs = True`` also receives ``policy_obs=`` (N, D), the full deployable policy observation."""
        ...


def proprio_of(env, obs_policy: torch.Tensor) -> torch.Tensor:
    """Deployable proprioceptive part of the policy observation (state estimate, gyro, last action)."""
    return obs_policy[:, : 12 + (env.action_dim if env.cfg.include_last_action else 0)]


@ALGORITHMS.register("vla")
class _VLAAlgorithm(Algorithm):
    """Evaluation of external models through :class:`VLAAdapter` (``--checkpoint module:factory``)."""

    name = "vla"

    def actor_module(self, act):
        return None  # external models are exported by their own toolchain

    def load_policy(self, checkpoint: str, env, task=None):
        mod, _, fn = checkpoint.partition(":")
        if mod.endswith(".py"):
            from quadsim.registry import import_plugins

            import_plugins([mod])
            import sys

            mod = "quadsim_plugin_" + mod.rsplit("/", 1)[-1][:-3]
            m = sys.modules[mod]
        else:
            m = importlib.import_module(mod)
        u = env.unwrapped
        adapter = getattr(m, fn or "make_adapter")(u)
        last_len = torch.zeros(u.num_envs, device=u.device)

        @torch.no_grad()
        def act(obs):
            new = u.episode_length_buf <= last_len  # episode restarted since the previous call
            last_len.copy_(u.episode_length_buf.float())
            if new.any() and hasattr(adapter, "reset"):
                adapter.reset(new.nonzero().squeeze(-1))
            ins = getattr(u, "instructions", [""] * u.num_envs)
            extra = {"policy_obs": obs["policy"]} if getattr(adapter, "wants_policy_obs", False) else {}
            return adapter(obs.get("image"), ins, proprio_of(u, obs["policy"]), **extra).clamp(-1, 1)

        act.policy = adapter
        return act
