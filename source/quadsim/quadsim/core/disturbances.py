"""Wind field and external disturbance models (batched, pure PyTorch)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .sensors.noise import GaussMarkov


@dataclass
class WindCfg:
    enabled: bool = False
    mean_speed_range: tuple[float, float] = (0.0, 3.0)  # m/s, horizontal, sampled per episode
    vertical_std: float = 0.2  # m/s mean vertical component
    turbulence_std: float = 0.5  # m/s, stationary std of each axis (Dryden-like Gauss-Markov)
    turbulence_tau: float = 2.0  # s, ~ L / V correlation time
    gust_prob_per_s: float = 0.05  # Poisson rate of discrete (1-cos) gusts
    gust_amp_range: tuple[float, float] = (1.0, 4.0)  # m/s
    gust_duration_range: tuple[float, float] = (0.5, 2.0)  # s


class WindModel:
    """Mean wind + Gauss-Markov turbulence + discrete 1-cosine gusts (MIL-F-8785C shape), world frame."""

    def __init__(self, cfg: WindCfg, num_envs: int, device):
        self.cfg = cfg
        self.n = num_envs
        self.device = device
        self.mean = torch.zeros(num_envs, 3, device=device)
        self.turb = GaussMarkov((num_envs, 3), cfg.turbulence_std, cfg.turbulence_tau, device)
        self.gust_dir = torch.zeros(num_envs, 3, device=device)
        self.gust_amp = torch.zeros(num_envs, device=device)
        self.gust_t = torch.zeros(num_envs, device=device)
        self.gust_T = torch.ones(num_envs, device=device)
        self.scale = torch.ones(num_envs, 1, device=device)  # curriculum multiplier

    def reset(self, env_ids):
        n = len(env_ids)
        c = self.cfg
        speed = c.mean_speed_range[0] + (c.mean_speed_range[1] - c.mean_speed_range[0]) * torch.rand(n, device=self.device)
        ang = torch.rand(n, device=self.device) * 2 * math.pi
        self.mean[env_ids] = torch.stack(
            [speed * torch.cos(ang), speed * torch.sin(ang), torch.randn(n, device=self.device) * c.vertical_std], -1
        )
        self.turb.reset(env_ids)
        self.gust_amp[env_ids] = 0.0
        self.gust_t[env_ids] = 0.0

    def step(self, dt: float) -> torch.Tensor:
        c = self.cfg
        if not c.enabled:
            return torch.zeros(self.n, 3, device=self.device)
        turb = self.turb.step(dt)
        # start new gusts
        start = (torch.rand(self.n, device=self.device) < c.gust_prob_per_s * dt) & (self.gust_amp == 0)
        if start.any():
            k = int(start.sum())
            d = torch.randn(k, 3, device=self.device)
            d[:, 2] *= 0.3
            self.gust_dir[start] = d / d.norm(dim=-1, keepdim=True)
            lo, hi = c.gust_amp_range
            self.gust_amp[start] = lo + (hi - lo) * torch.rand(k, device=self.device)
            lo, hi = c.gust_duration_range
            self.gust_T[start] = lo + (hi - lo) * torch.rand(k, device=self.device)
            self.gust_t[start] = 0.0
        active = self.gust_amp > 0
        self.gust_t = torch.where(active, self.gust_t + dt, self.gust_t)
        shape = 0.5 * (1 - torch.cos(2 * math.pi * self.gust_t / self.gust_T))
        gust = (self.gust_amp * shape)[:, None] * self.gust_dir
        done = active & (self.gust_t >= self.gust_T)
        self.gust_amp = torch.where(done, torch.zeros_like(self.gust_amp), self.gust_amp)
        return self.scale * (self.mean + turb + gust)


@dataclass
class PushCfg:
    enabled: bool = False
    force_range: tuple[float, float] = (0.0, 0.3)  # in units of vehicle weight
    torque_range: tuple[float, float] = (0.0, 0.0)  # N m
    duration_range: tuple[float, float] = (0.05, 0.3)  # s
    prob_per_s: float = 0.2


class PushDisturbance:
    """Random finite-duration external force/torque impulses (world-frame force, body torque)."""

    def __init__(self, cfg: PushCfg, num_envs: int, weight: torch.Tensor, device):
        self.cfg = cfg
        self.n = num_envs
        self.device = device
        self.weight = weight
        self.force = torch.zeros(num_envs, 3, device=device)
        self.torque = torch.zeros(num_envs, 3, device=device)
        self.remaining = torch.zeros(num_envs, device=device)

    def reset(self, env_ids):
        self.force[env_ids] = 0.0
        self.torque[env_ids] = 0.0
        self.remaining[env_ids] = 0.0

    def step(self, dt: float):
        c = self.cfg
        if not c.enabled:
            return self.force, self.torque
        self.remaining = (self.remaining - dt).clamp_min(0.0)
        idle = self.remaining <= 0
        self.force[idle] = 0.0
        self.torque[idle] = 0.0
        start = idle & (torch.rand(self.n, device=self.device) < c.prob_per_s * dt)
        if start.any():
            k = int(start.sum())
            d = torch.randn(k, 3, device=self.device)
            d = d / d.norm(dim=-1, keepdim=True)
            mag = (c.force_range[0] + (c.force_range[1] - c.force_range[0]) * torch.rand(k, device=self.device))
            self.force[start] = d * (mag * self.weight[start])[:, None]
            dt_ = torch.randn(k, 3, device=self.device)
            tmag = c.torque_range[0] + (c.torque_range[1] - c.torque_range[0]) * torch.rand(k, device=self.device)
            self.torque[start] = dt_ / dt_.norm(dim=-1, keepdim=True) * tmag[:, None]
            lo, hi = c.duration_range
            self.remaining[start] = lo + (hi - lo) * torch.rand(k, device=self.device)
        return self.force, self.torque
