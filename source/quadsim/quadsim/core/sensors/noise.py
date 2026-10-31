"""Stochastic building blocks for sensor and actuator models (batched, pure PyTorch)."""

from __future__ import annotations

import math

import torch


def sample_uniform(lo, hi, shape, device, generator=None) -> torch.Tensor:
    lo = torch.as_tensor(lo, dtype=torch.float32, device=device)
    hi = torch.as_tensor(hi, dtype=torch.float32, device=device)
    return lo + (hi - lo) * torch.rand(shape, device=device, generator=generator)


class GaussMarkov:
    """First-order Gauss-Markov (Ornstein-Uhlenbeck) process, exact discretization.

    ``dx = -x/tau dt + sigma * sqrt(2/tau) dW`` has stationary std ``sigma``. With ``tau = inf`` it
    degenerates to a random walk with intensity ``sigma_rw`` [unit/sqrt(s)] (bias instability model).
    """

    def __init__(self, shape, sigma: float, tau: float | None, device, random_walk_sigma: float = 0.0):
        self.x = torch.zeros(shape, device=device)
        self.sigma = sigma
        self.tau = tau
        self.rw = random_walk_sigma

    def reset(self, env_ids=None, initial_std: float | None = None):
        ids = slice(None) if env_ids is None else env_ids
        std = self.sigma if initial_std is None else initial_std
        self.x[ids] = torch.randn_like(self.x[ids]) * std

    def step(self, dt: float) -> torch.Tensor:
        if self.tau is not None and self.tau > 0 and self.sigma > 0:
            a = math.exp(-dt / self.tau)
            self.x = a * self.x + self.sigma * math.sqrt(1 - a * a) * torch.randn_like(self.x)
        if self.rw > 0:
            self.x = self.x + self.rw * math.sqrt(dt) * torch.randn_like(self.x)
        return self.x


class DelayBuffer:
    """Ring buffer returning, per environment, the value pushed ``delay[i]`` pushes ago.

    ``delay = 0`` returns the most recent value. Delays are integers in units of push calls.
    """

    def __init__(self, num_envs: int, max_delay: int, feature_shape, device):
        self.max_delay = max_delay
        self.buf = torch.zeros((max_delay + 1, num_envs) + tuple(feature_shape), device=device)
        self.head = 0
        self.delay = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._arange = torch.arange(num_envs, device=device)

    def set_delay(self, delay: torch.Tensor, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.delay[ids] = torch.clamp(delay.long(), 0, self.max_delay)

    def reset(self, env_ids, value: torch.Tensor | None = None):
        """Fill the history of ``env_ids`` with ``value`` (or zeros)."""
        if value is None:
            self.buf[:, env_ids] = 0.0
        else:
            self.buf[:, env_ids] = value.unsqueeze(0)

    def push(self, value: torch.Tensor) -> torch.Tensor:
        self.head = (self.head + 1) % (self.max_delay + 1)
        self.buf[self.head] = value
        idx = (self.head - self.delay) % (self.max_delay + 1)
        return self.buf[idx, self._arange]


class SampleAndHold:
    """Holds the last sample until the next sampling instant of a sensor with rate ``rate_hz``."""

    def __init__(self, num_envs: int, feature_shape, rate_hz: float, sim_dt: float, device):
        self.period = max(1, int(round(1.0 / (rate_hz * sim_dt))))
        self.value = torch.zeros((num_envs,) + tuple(feature_shape), device=device)
        self.counter = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids, phase: torch.Tensor | None = None):
        self.counter[env_ids] = 0 if phase is None else phase
        self.value[env_ids] = 0.0

    def step(self, new_sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fire = self.counter % self.period == 0
        self.value = torch.where(fire.view(-1, *([1] * (self.value.dim() - 1))), new_sample, self.value)
        self.counter += 1
        return self.value, fire
