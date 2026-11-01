"""GPU-native off-policy RL (SAC, TD3) for massively parallel quadsim environments.

Design (follows the "many environments, large batches, several updates per step" recipe that makes off-policy RL
competitive on GPU simulators, e.g. FastTD3, Seo et al. 2025):

* asymmetric actor-critic like the PPO configuration: the actor sees the deployable ``policy`` observation, the twin
  Q-critics see the ``critic`` group (privileged state, DR parameters) when the task provides it;
* running observation normalization (statistics applied when sampling, so stored transitions stay raw);
* LayerNorm critics, clipped double-Q, Polyak-averaged targets; SAC with automatic temperature, TD3 with target
  policy smoothing, delayed actor updates and per-environment exploration noise levels;
* time-outs bootstrap from the current state (Isaac Lab returns the reset observation after a time-out; rsl_rl's PPO
  uses the same approximation); failures (``terminated``) do not bootstrap;
* rewards are multiplied by ``reward_scale`` (default ``1 / step_dt``): the tasks' rewards are dt-scaled, so per-step
  rewards would otherwise be ~50x smaller than SAC's entropy bonus.

Checkpoints ``model_<step>.pt`` hold networks, optimizers and normalizers (not the replay buffer: a resumed run
refills it during ``learning_starts`` steps).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OffPolicyCfg:
    algo: str = "sac"
    """sac | td3"""
    experiment_name: str = "quadsim"
    seed: int = 42
    device: str = "cuda:0"
    max_iterations: int = 20000
    """vectorized environment steps"""
    buffer_steps: int = 512
    """replay capacity in vectorized steps (capacity = buffer_steps * num_envs transitions)"""
    batch_size: int = 16384
    updates_per_step: int = 4
    learning_starts: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = 0.0
    actor_hidden_dims: list = field(default_factory=lambda: [256, 256, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    asymmetric: bool = True
    reward_scale: float | None = None
    """None = 1 / step_dt"""
    max_grad_norm: float = 10.0
    # SAC
    init_alpha: float = 0.05
    target_entropy_per_dim: float = -1.0
    # TD3
    explore_std_range: tuple = (0.05, 0.3)
    """per-environment Gaussian exploration std, sampled uniformly at start"""
    target_noise: float = 0.2
    target_noise_clip: float = 0.5
    policy_delay: int = 2
    save_interval: int = 2000
    log_interval: int = 50


def _mlp(i, hidden, o, layer_norm=False, act=nn.ELU):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h)] + ([nn.LayerNorm(h)] if layer_norm else []) + [act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-4, clip=10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta.square() * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / (self.var + 1e-8).sqrt()).clamp(-self.clip, self.clip)


class Actor(nn.Module):
    """Normalizer + MLP; ``forward`` returns the deterministic action in [-1, 1] (deployment / ONNX export)."""

    def __init__(self, obs_dim, act_dim, hidden, stochastic):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.net = _mlp(obs_dim, hidden, 2 * act_dim if stochastic else act_dim)
        self.stochastic, self.act_dim = stochastic, act_dim

    def forward(self, obs):
        out = self.net(self.norm(obs))
        return torch.tanh(out[..., : self.act_dim])

    def sample(self, obs):
        out = self.net(self.norm(obs))
        mean, log_std = out[..., : self.act_dim], out[..., self.act_dim:].clamp(-5.0, 1.0)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mean) / std).square() - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2.0) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.q1 = _mlp(obs_dim + act_dim, hidden, 1, layer_norm=True)
        self.q2 = _mlp(obs_dim + act_dim, hidden, 1, layer_norm=True)

    def forward(self, obs, act):
        x = torch.cat([self.norm(obs), act], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, steps, n, dims: dict, device):
        self.cap, self.n, self.ptr, self.full = steps, n, 0, False
        self.data = {k: torch.zeros(steps, n, d, device=device) for k, d in dims.items()}

    def add(self, **kw):
        for k, v in kw.items():
            self.data[k][self.ptr] = v.reshape(self.n, -1)
        self.ptr = (self.ptr + 1) % self.cap
        self.full |= self.ptr == 0

    def sample(self, batch):
        hi = self.cap if self.full else self.ptr
        t = torch.randint(0, hi, (batch,), device=self.data["r"].device)
        e = torch.randint(0, self.n, (batch,), device=t.device)
        return {k: v[t, e] for k, v in self.data.items()}


def _obs(obs: dict, asymmetric: bool):
    a = obs["policy"]
    c = obs["critic"] if asymmetric and "critic" in obs else a
    return a, c


class OffPolicyLearner:
    def __init__(self, env, cfg: OffPolicyCfg, log_dir: str | None):
        self.env, self.cfg, self.log_dir = env, cfg, log_dir
        u = env.unwrapped
        self.dev = torch.device(cfg.device)
        self.n = u.num_envs
        obs, _ = env.reset()
        oa, oc = _obs(obs, cfg.asymmetric)
        self.obs_dim, self.critic_dim, self.act_dim = oa.shape[-1], oc.shape[-1], u.action_dim
        sac = cfg.algo == "sac"
        self.actor = Actor(self.obs_dim, self.act_dim, cfg.actor_hidden_dims, stochastic=sac).to(self.dev)
        self.actor_t = None if sac else Actor(self.obs_dim, self.act_dim, cfg.actor_hidden_dims, False).to(self.dev)
        self.critic = TwinQ(self.critic_dim, self.act_dim, cfg.critic_hidden_dims).to(self.dev)
        self.critic_t = TwinQ(self.critic_dim, self.act_dim, cfg.critic_hidden_dims).to(self.dev)
        self.critic_t.load_state_dict(self.critic.state_dict())
        if self.actor_t is not None:
            self.actor_t.load_state_dict(self.actor.state_dict())
        self.opt_a = torch.optim.AdamW(self.actor.net.parameters(), cfg.actor_lr, weight_decay=cfg.weight_decay)
        self.opt_c = torch.optim.AdamW(list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()),
                                       cfg.critic_lr, weight_decay=cfg.weight_decay)
        self.log_alpha = torch.tensor(math.log(cfg.init_alpha), device=self.dev, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], cfg.actor_lr)
        self.target_entropy = cfg.target_entropy_per_dim * self.act_dim
        lo, hi = cfg.explore_std_range
        self.explore_std = lo + (hi - lo) * torch.rand(self.n, 1, device=self.dev)
        self.buffer = ReplayBuffer(cfg.buffer_steps, self.n, {"oa": self.obs_dim, "oc": self.critic_dim,
                                                                "a": self.act_dim, "r": 1, "oa2": self.obs_dim,
                                                                "oc2": self.critic_dim, "term": 1}, self.dev)
        self.reward_scale = cfg.reward_scale or 1.0 / u.step_dt
        self.step, self.updates = 0, 0
        self._obs = obs
        self.writer = None
        if log_dir:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, oa, explore=True):
        if not explore:
            return self.actor(oa)
        if self.step < self.cfg.learning_starts:
            return torch.rand(self.n, self.act_dim, device=self.dev) * 2 - 1
        if self.cfg.algo == "sac":
            return self.actor.sample(oa)[0]
        return (self.actor(oa) + self.explore_std * torch.randn(self.n, self.act_dim, device=self.dev)).clamp(-1, 1)

    # ------------------------------------------------------------------ learning
    def _update(self):
        c = self.cfg
        b = self.buffer.sample(c.batch_size)
        r, term = b["r"].squeeze(-1) * self.reward_scale, b["term"].squeeze(-1)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            if c.algo == "sac":
                a2, logp2 = self.actor.sample(b["oa2"])
                q1t, q2t = self.critic_t(b["oc2"], a2)
                q_next = torch.min(q1t, q2t) - alpha * logp2
            else:
                noise = (torch.randn_like(b["a"]) * c.target_noise).clamp(-c.target_noise_clip, c.target_noise_clip)
                a2 = (self.actor_t(b["oa2"]) + noise).clamp(-1, 1)
                q1t, q2t = self.critic_t(b["oc2"], a2)
                q_next = torch.min(q1t, q2t)
            y = r + c.gamma * (1.0 - term) * q_next
        q1, q2 = self.critic(b["oc"], b["a"])
        lc = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c.zero_grad(set_to_none=True)
        lc.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
        self.opt_c.step()
        out = {"loss/critic": lc.detach(), "q/mean": q1.detach().mean()}
        self.updates += 1
        if c.algo == "sac" or self.updates % c.policy_delay == 0:
            if c.algo == "sac":
                a, logp = self.actor.sample(b["oa"])
                q1p, q2p = self.critic(b["oc"], a)
                la = (alpha * logp - torch.min(q1p, q2p)).mean()
            else:
                la = -self.critic(b["oc"], self.actor(b["oa"]))[0].mean()
            self.opt_a.zero_grad(set_to_none=True)
            la.backward()
            nn.utils.clip_grad_norm_(self.actor.net.parameters(), c.max_grad_norm)
            self.opt_a.step()
            out["loss/actor"] = la.detach()
            if c.algo == "sac":
                l_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                self.opt_alpha.zero_grad(set_to_none=True)
                l_alpha.backward()
                self.opt_alpha.step()
                out["alpha"] = alpha
                out["entropy"] = -logp.detach().mean()
            with torch.no_grad():
                for net, tgt in ((self.critic, self.critic_t), (self.actor, self.actor_t)):
                    if tgt is None:
                        continue
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.lerp_(p, c.tau)
        if c.algo == "sac":  # SAC: target critic every step
            with torch.no_grad():
                for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                    pt.lerp_(p, c.tau)
        return out

    def _sync_norms(self):
        """Target networks share the online normalizer statistics."""
        self.critic_t.norm.load_state_dict(self.critic.norm.state_dict())
        if self.actor_t is not None:
            self.actor_t.norm.load_state_dict(self.actor.norm.state_dict())

    def learn(self, num_steps: int):
        c, env = self.cfg, self.env
        t0, ep_ret = time.time(), torch.zeros(self.n, device=self.dev)
        rets, logs = [], {}
        obs = self._obs
        end = self.step + num_steps
        while self.step < end:
            oa, oc = _obs(obs, c.asymmetric)
            self.actor.norm.update(oa)
            self.critic.norm.update(oc)
            self._sync_norms()
            a = self.act(oa)
            obs2, rew, term, trunc, extras = env.step(a)
            oa2, oc2 = _obs(obs2, c.asymmetric)
            done = (term | trunc)[:, None]
            # time-out / reset: the returned observation is the reset state -> bootstrap from the current state
            oa_next = torch.where(done, oa, oa2)
            oc_next = torch.where(done, oc, oc2)
            self.buffer.add(oa=oa, oc=oc, a=a, r=rew, oa2=oa_next, oc2=oc_next, term=term.float())
            ep_ret += rew
            if done.any():
                d = done.squeeze(-1)
                rets.append(ep_ret[d].clone())
                ep_ret[d] = 0.0
            for k, v in extras.get("log", {}).items():
                logs[k] = v
            obs = obs2
            self.step += 1
            if self.step >= c.learning_starts:
                for _ in range(c.updates_per_step):
                    out = self._update()
            if self.writer and self.step % c.log_interval == 0:
                fps = c.log_interval * self.n / (time.time() - t0)
                t0 = time.time()
                self.writer.add_scalar("perf/env_steps_per_s", fps, self.step)
                if rets:
                    r = torch.cat(rets).mean().item()
                    self.writer.add_scalar("train/episode_return", r, self.step)
                    rets.clear()
                if self.step >= c.learning_starts:
                    for k, v in out.items():
                        self.writer.add_scalar(k, float(v), self.step)
                for k, v in logs.items():
                    self.writer.add_scalar(k, float(v), self.step)
                print(f"[{c.algo}] step {self.step}/{end}  {fps:,.0f} env-steps/s  "
                      + (f"return {r:.3f}  " if 'r' in locals() else "")
                      + " ".join(f"{k} {float(v):.3g}" for k, v in out.items()) if self.step >= c.learning_starts
                      else f"[{c.algo}] step {self.step} (random warm-up)", flush=True)
            if self.log_dir and (self.step % c.save_interval == 0 or self.step == end):
                self.save(os.path.join(self.log_dir, f"model_{self.step}.pt"))
        self._obs = obs

    # ------------------------------------------------------------------ checkpoints
    def save(self, path):
        torch.save({"algo": self.cfg.algo, "step": self.step, "cfg": asdict(self.cfg) if hasattr(self.cfg, "__dataclass_fields__") else {},
                    "dims": {"obs": self.obs_dim, "critic": self.critic_dim, "act": self.act_dim},
                    "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "critic_t": self.critic_t.state_dict(),
                    "actor_t": None if self.actor_t is None else self.actor_t.state_dict(),
                    "opt_a": self.opt_a.state_dict(), "opt_c": self.opt_c.state_dict(),
                    "log_alpha": self.log_alpha.detach(), "opt_alpha": self.opt_alpha.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_t.load_state_dict(ck["critic_t"])
        if self.actor_t is not None and ck.get("actor_t"):
            self.actor_t.load_state_dict(ck["actor_t"])
        self.opt_a.load_state_dict(ck["opt_a"])
        self.opt_c.load_state_dict(ck["opt_c"])
        self.opt_alpha.load_state_dict(ck["opt_alpha"])
        with torch.no_grad():
            self.log_alpha.copy_(ck["log_alpha"])
        self.step = ck["step"]
        self.cfg.learning_starts = self.step + self.cfg.learning_starts  # refill the (unsaved) replay buffer


def load_actor(checkpoint: str, device="cpu") -> Actor:
    """Deterministic actor (normalizer included) from an off-policy checkpoint: ``actor(obs) -> action``."""
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    d, cfg = ck["dims"], ck.get("cfg", {})
    actor = Actor(d["obs"], d["act"], cfg.get("actor_hidden_dims", [256, 256, 128]), stochastic=ck["algo"] == "sac")
    actor.load_state_dict(ck["actor"])
    return actor.to(device).eval()
