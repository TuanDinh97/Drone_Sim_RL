"""Image-level sensor models for depth and RGB cameras (applied to rendered or ray-cast images)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DepthNoiseCfg:
    min_range: float = 0.2  # m, closer returns invalid
    max_range: float = 10.0  # m, farther clipped
    stereo_coeff: float = 0.005  # sigma_z = coeff * z^2 (structured-light / stereo error growth)
    base_std: float = 0.005  # m
    dropout_prob: float = 0.01  # per-pixel invalid probability
    quantization: float = 0.001  # m
    invalid_value: float = 0.0  # what invalid pixels read as (RealSense convention: 0)
    enabled: bool = True


def apply_depth_noise(depth: torch.Tensor, cfg: DepthNoiseCfg, scale: torch.Tensor | None = None) -> torch.Tensor:
    """depth (N, H, W) or (N, 1, H, W) metric -> noisy metric depth with invalid pixels set."""
    d = torch.nan_to_num(depth.clone(), nan=1e6, posinf=1e6)
    no_return = d >= cfg.max_range  # beyond range / sky: reads as max range (free space), never noised
    too_close = d < cfg.min_range
    if cfg.enabled:
        s = 1.0 if scale is None else scale.view(-1, *([1] * (d.dim() - 1)))
        std = cfg.base_std + cfg.stereo_coeff * d.clamp(max=cfg.max_range) ** 2
        d = d + s * torch.randn_like(d) * std
        if cfg.quantization > 0:
            d = torch.round(d / cfg.quantization) * cfg.quantization
        d = d.clamp(min=cfg.min_range)  # noise must not turn valid returns into "too close"
    d = torch.where(no_return, torch.full_like(d, cfg.max_range), d.clamp(max=cfg.max_range))
    invalid = too_close
    if cfg.enabled:
        invalid = invalid | (torch.rand_like(d) < cfg.dropout_prob)
    return torch.where(invalid, torch.full_like(d, cfg.invalid_value), d)


def normalize_depth(depth: torch.Tensor, max_range: float) -> torch.Tensor:
    """Metric depth -> [0, 1] (invalid=0 stays 0 so the network can learn 'unknown')."""
    return (depth / max_range).clamp(0.0, 1.0)


@dataclass
class RgbNoiseCfg:
    brightness: float = 0.2  # +- additive, per episode
    contrast: float = 0.2  # +- multiplicative around mean, per episode
    saturation: float = 0.2
    pixel_noise_std: float = 0.02
    enabled: bool = True


class RgbAugment:
    """Per-episode photometric randomization + per-frame pixel noise for RGB in [0, 1]."""

    def __init__(self, cfg: RgbNoiseCfg, num_envs: int, device):
        self.cfg = cfg
        self.b = torch.zeros(num_envs, 1, 1, 1, device=device)
        self.c = torch.ones(num_envs, 1, 1, 1, device=device)
        self.s = torch.ones(num_envs, 1, 1, 1, device=device)

    def reset(self, env_ids):
        n = len(env_ids)
        dev = self.b.device
        c = self.cfg
        self.b[env_ids] = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * c.brightness
        self.c[env_ids] = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * c.contrast
        self.s[env_ids] = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * c.saturation

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb (N, 3, H, W) in [0, 1]."""
        if not self.cfg.enabled:
            return rgb
        gray = rgb.mean(1, keepdim=True)
        x = gray + self.s * (rgb - gray)
        mean = x.mean((1, 2, 3), keepdim=True)
        x = mean + self.c * (x - mean) + self.b
        x = x + torch.randn_like(x) * self.cfg.pixel_noise_std
        return x.clamp(0.0, 1.0)
