"""Example: plug a new learning algorithm into the framework - Augmented Random Search (Mania et al. 2018).

    python scripts/rl/train.py --headless --task Quadsim-Hover-v0 --algo ars --plugins examples/custom_algorithm.py \
        --max_iterations 150
    python scripts/rl/evaluate.py --headless --task Quadsim-Hover-v0 --plugins examples/custom_algorithm.py \
        --controllers policy --checkpoint logs/ars/quadsim_hover/<run>

An algorithm registers an :class:`quadsim.rl.algorithms.Algorithm` with ``default_cfg`` (a configclass, so
``agent.<field>=value`` command-line overrides work), ``train`` and ``load_policy``. ARS is a derivative-free method
whose rollouts map naturally onto thousands of parallel environments: every environment evaluates one perturbed
linear policy ``a = clip(W [norm(o), 1])``; the update moves ``W`` along the best-performing perturbation directions.
"""

import os
import time

import torch

from isaaclab.utils import configclass

from quadsim.registry import ALGORITHMS
from quadsim.rl.algorithms import Algorithm


@configclass
class ARSCfg:
    experiment_name: str = "quadsim"
    seed: int = 42
    device: str = "cuda:0"
    max_iterations: int = 150
    rollout_steps: int = 150
    """policy steps per evaluation of a perturbation (fixed horizon, episodes auto-reset inside)"""
    envs_per_direction: int = 16
    top_directions: int = 64
    step_size: float = 0.02
    noise: float = 0.03
    save_interval: int = 50


def _features(obs, mean, var):
    x = (obs - mean) / (var.sqrt() + 1e-6)
    return torch.cat([x.clamp(-5, 5), torch.ones_like(x[:, :1])], -1)


class LinearPolicy(torch.nn.Module):
    """The trained ARS policy as a module (``Algorithm.actor_module`` -> ONNX export)."""

    def __init__(self, w, mean, var):
        super().__init__()
        self.register_buffer("w", w)
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)

    def forward(self, obs):
        return (_features(obs, self.mean, self.var) @ self.w.T).clamp(-1, 1)


@ALGORITHMS.register("ars", overwrite=True)
class ARS(Algorithm):
    name = "ars"

    def default_cfg(self, task):
        from quadsim.rl.policy_io import agent_cfg_for

        cfg = ARSCfg()
        cfg.experiment_name = agent_cfg_for(task).experiment_name
        return cfg

    def train(self, env, cfg, log_dir, resume=None):
        from torch.utils.tensorboard import SummaryWriter

        torch.manual_seed(cfg.seed)
        u = env.unwrapped
        dev, n = u.device, u.num_envs
        n_dir = n // (2 * cfg.envs_per_direction)
        obs, _ = env.reset()
        o_dim, a_dim = obs["policy"].shape[1], u.action_dim
        w = torch.zeros(a_dim, o_dim + 1, device=dev)
        w[:, -1] = u.hover_action  # bias starts at the hover command
        mean, var, count = torch.zeros(o_dim, device=dev), torch.ones(o_dim, device=dev), 1e-4
        start = 0
        if resume:
            from quadsim.rl.policy_io import latest_checkpoint

            ck = torch.load(resume if resume.endswith(".pt") else latest_checkpoint(resume), map_location=dev)
            w, mean, var = ck["w"], ck["mean"], ck["var"]
            count, start = ck.get("count", count), ck.get("iteration", 0)
        e = torch.arange(n_dir * 2 * cfg.envs_per_direction, device=dev)
        d_of_env, sign = (e // cfg.envs_per_direction) % n_dir, 1.0 - 2.0 * (e // (cfg.envs_per_direction * n_dir))
        writer = SummaryWriter(log_dir)
        for it in range(start, start + cfg.max_iterations):
            t0 = time.time()
            delta = torch.randn(n_dir, a_dim, o_dim + 1, device=dev)
            w_env = w[None] + cfg.noise * sign[:, None, None] * delta[d_of_env]  # (E, A, O+1)
            ret = torch.zeros(n, device=dev)
            for _ in range(cfg.rollout_steps):
                o = obs["policy"]
                b_mean, b_var = o.mean(0), o.var(0, unbiased=False)  # running observation statistics (ARS V2)
                tot = count + n
                d = b_mean - mean
                mean, var = mean + d * n / tot, (var * count + b_var * n + d.square() * count * n / tot) / tot
                count = tot
                x = _features(o[: len(e)], mean, var)
                a = torch.zeros(n, a_dim, device=dev)
                a[: len(e)] = torch.einsum("eao,eo->ea", w_env, x).clamp(-1, 1)
                obs, r, *_ = env.step(a)
                ret += r
            r_dir = ret[: len(e)].view(2, n_dir, cfg.envs_per_direction).mean(-1)  # (+/-, directions)
            best = r_dir.max(0).values.topk(min(cfg.top_directions, n_dir)).indices
            diff = (r_dir[0, best] - r_dir[1, best])
            sigma = r_dir[:, best].std().clamp_min(1e-6)
            w = w + cfg.step_size / (len(best) * sigma) * torch.einsum("d,dao->ao", diff, delta[best])
            writer.add_scalar("ars/return_mean", r_dir.mean().item(), it)
            print(f"[ars] it {it + 1}/{start + cfg.max_iterations} return {r_dir.mean().item():.3f} "
                  f"(best {r_dir.max().item():.3f}) {time.time() - t0:.1f} s", flush=True)
            if (it + 1) % cfg.save_interval == 0 or it + 1 == start + cfg.max_iterations:
                torch.save({"algo": "ars", "w": w, "mean": mean, "var": var, "count": count, "iteration": it + 1},
                           os.path.join(log_dir, f"model_{it + 1}.pt"))

    def load_policy(self, checkpoint, env, task=None):
        ck = torch.load(checkpoint, map_location=str(env.unwrapped.device))
        policy = LinearPolicy(ck["w"], ck["mean"], ck["var"])

        @torch.no_grad()
        def act(obs):
            return policy(obs["policy"])

        act.policy = policy  # what Algorithm.actor_module returns (ONNX export)
        return act
