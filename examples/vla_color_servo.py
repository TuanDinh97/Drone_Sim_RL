"""Example VLA adapter (model-free): keyword parsing + colour segmentation visual servoing for Quadsim-LangNav-RGB.

    python scripts/rl/evaluate.py --headless --enable_cameras --task Quadsim-LangNav-RGB-v0 --controllers policy \
        --algo vla --checkpoint examples/vla_color_servo.py:make_adapter --episodes 200 --num_envs 64

It implements the :class:`quadsim.il.vla.VLAAdapter` interface that a learned VLA (e.g. a LeRobot policy fine-tuned
on data exported with scripts/il/export_lerobot.py) would implement: (RGB images, instruction strings,
proprioception) -> normalized actions of the environment's action interface (velocity + yaw rate here). The
"language model" is a keyword lookup and the "vision model" a colour threshold - a transparent reference point for
learned vision-language policies, not a learned model.
"""

import torch

COLOR_RGB = {"red": (0.85, 0.08, 0.08), "green": (0.1, 0.7, 0.15), "blue": (0.1, 0.25, 0.9),
             "yellow": (0.9, 0.8, 0.1)}


class ColorServoVLA:
    def __init__(self, env, speed=1.2, stop_width=0.42, k_yaw=2.5):
        self.env, self.speed, self.stop_width, self.k_yaw = env, speed, stop_width, k_yaw

    def reset(self, env_ids):
        pass

    @torch.no_grad()
    def __call__(self, images, instructions, proprio):
        n, dev = images.shape[0], images.device
        target = torch.tensor([[COLOR_RGB.get(next((c for c in COLOR_RGB if c in s.lower()), "red"), (1, 0, 0))]
                               for s in instructions], device=dev).view(n, 3, 1, 1)
        rgb = images[:, :3]
        # pixels close to the named colour (normalized chromaticity distance)
        chrom = rgb / rgb.sum(1, keepdim=True).clamp_min(1e-3)
        tchrom = target / target.sum(1, keepdim=True)
        mask = ((chrom - tchrom).norm(dim=1) < 0.12) & (rgb.max(1).values > 0.15)
        w = images.shape[-1]
        cols = mask.float().sum(1)  # (N, W) column histogram
        seen = cols.sum(-1) > 3
        x = torch.arange(w, device=dev).float()
        cx = (cols * x).sum(-1) / cols.sum(-1).clamp_min(1) / (w - 1) * 2 - 1  # horizontal position in [-1, 1]
        width = (cols > 0).float().sum(-1) / w  # apparent width ~ 1 / distance
        a = torch.zeros(n, 4, device=dev)
        fwd = torch.clamp(1.0 - width / self.stop_width, -0.3, 1.0) * self.speed
        a[:, 0] = torch.where(seen, fwd, torch.zeros_like(fwd)) / self.env.max_vel[0]
        a[:, 3] = torch.where(seen, -self.k_yaw * cx, torch.full_like(cx, 0.5)) / self.env.cfg.control.max_yaw_rate
        return a.clamp(-1, 1)


def make_adapter(env):
    return ColorServoVLA(env)
