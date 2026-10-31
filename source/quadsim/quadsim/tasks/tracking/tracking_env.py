"""Trajectory tracking task with analytic random reference trajectories.

Each environment draws a reference ``p_ref(t) = c + sum_k A_k sin(w_k t + phi_k)`` per axis, which
covers circles, figure-eights (Lissajous 1:2) and generic smooth curves, with analytic velocity and
acceleration (used by the classical baselines as feed-forward). The policy observes the reference
at several future instants expressed in its estimated body frame.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from quadsim.core import math as qm
from quadsim.tasks.base import QuadrotorEnv, QuadrotorEnvCfg


@configclass
class TrackingEnvCfg(QuadrotorEnvCfg):
    episode_length_s: float = 10.0
    horizon_s: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.6, 1.0)
    num_harmonics: int = 3
    amp_range: tuple[float, float] = (0.2, 1.5)  # m (scaled by curriculum)
    freq_range: tuple[float, float] = (0.15, 0.6)  # Hz
    z_amp_scale: float = 0.3
    center_height: float = 2.0
    max_speed: float = 4.0  # references with higher peak speed are rescaled in time
    shape_probs: tuple[float, float, float] = (0.3, 0.3, 0.4)  # circle, figure-8, random
    init_pos_noise: float = 0.3
    init_vel_noise: float = 0.3
    init_max_tilt: float = 0.3
    max_track_error: float = 3.0
    # reward weights
    w_pos: float = 1.0
    w_pos_fine: float = 0.5
    w_vel: float = 0.1
    w_ang_vel: float = 0.02
    w_action_rate: float = 0.05
    w_crash: float = 10.0


class TrackingEnv(QuadrotorEnv):
    cfg: TrackingEnvCfg

    @classmethod
    def task_obs_dim(cls, cfg) -> int:
        return 3 * len(cfg.horizon_s) + 3

    def __init__(self, cfg: TrackingEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, h, dev = self.num_envs, cfg.num_harmonics, self.device
        self.center = torch.zeros(n, 3, device=dev)
        self.amp = torch.zeros(n, h, 3, device=dev)
        self.freq = torch.zeros(n, h, 3, device=dev)  # rad/s
        self.phase = torch.zeros(n, h, 3, device=dev)
        self.t_start = torch.zeros(n, dtype=torch.long, device=dev)
        self.horizon = torch.tensor(cfg.horizon_s, device=dev)
        self.track_err_sum = torch.zeros(n, device=dev)
        self.track_err_cnt = torch.zeros(n, device=dev)
        self.track_err_max = torch.zeros(n, device=dev)

    # -------------------------------------------------------------- reference
    def reference(self, t: torch.Tensor, env_ids=None):
        """t: (K,) or (K, H) seconds since episode start -> pos, vel, acc of shape t.shape + (3,)."""
        ids = slice(None) if env_ids is None else env_ids
        a, w, ph, c = self.amp[ids], self.freq[ids], self.phase[ids], self.center[ids]
        extra = t.dim() - 1
        tt = t.reshape(t.shape + (1, 1))
        a = a.reshape(a.shape[:1] + (1,) * extra + a.shape[1:])
        w = w.reshape(w.shape[:1] + (1,) * extra + w.shape[1:])
        ph = ph.reshape(ph.shape[:1] + (1,) * extra + ph.shape[1:])
        arg = w * tt + ph
        pos = (a * torch.sin(arg)).sum(-2) + c.reshape(c.shape[:1] + (1,) * extra + (3,))
        vel = (a * w * torch.cos(arg)).sum(-2)
        acc = (-a * w * w * torch.sin(arg)).sum(-2)
        return pos, vel, acc

    def episode_time(self):
        return (self.episode_length_buf - self.t_start).float() * self.step_dt

    def _sample_reference(self, env_ids):
        k, h, dev, c = len(env_ids), self.cfg.num_harmonics, self.device, self.cfg
        amp = torch.zeros(k, h, 3, device=dev)
        freq = torch.zeros(k, h, 3, device=dev)
        phase = torch.zeros(k, h, 3, device=dev)
        shape = torch.multinomial(torch.tensor(c.shape_probs, device=dev), k, replacement=True)
        f0 = (c.freq_range[0] + (c.freq_range[1] - c.freq_range[0]) * torch.rand(k, device=dev)) * 2 * math.pi
        r = c.amp_range[0] + (c.amp_range[1] - c.amp_range[0]) * torch.rand(k, device=dev)
        # circle: x = r sin(wt + pi/2), y = r sin(wt)
        circ = shape == 0
        amp[circ, 0, 0] = r[circ]
        amp[circ, 0, 1] = r[circ]
        freq[circ, 0, :2] = f0[circ, None]
        phase[circ, 0, 0] = math.pi / 2
        # figure-8 (Lissajous 1:2): x = r sin(wt), y = r/2 sin(2wt)
        f8 = shape == 1
        amp[f8, 0, 0] = r[f8]
        amp[f8, 0, 1] = 0.5 * r[f8]
        freq[f8, 0, 0] = f0[f8]
        freq[f8, 0, 1] = 2 * f0[f8]
        # random smooth curve: h harmonics per axis
        rnd = shape == 2
        nr = int(rnd.sum())
        if nr > 0:
            amp[rnd] = (torch.rand(nr, h, 3, device=dev) * r[rnd, None, None] / h)
            freq[rnd] = (c.freq_range[0] + (c.freq_range[1] - c.freq_range[0]) * torch.rand(nr, h, 3, device=dev)) \
                * 2 * math.pi
            phase[rnd] = torch.rand(nr, h, 3, device=dev) * 2 * math.pi
        # vertical motion for all
        amp[:, -1, 2] += c.z_amp_scale * r * torch.rand(k, device=dev)
        freq[:, -1, 2] = torch.where(freq[:, -1, 2] > 0, freq[:, -1, 2], f0)
        # random rotation about z of the horizontal pattern via phase offsets is implicit; rescale peak speed
        peak = (amp * freq).sum(1).norm(dim=-1)
        scale = torch.clamp(c.max_speed / peak.clamp_min(1e-6), max=1.0)
        freq = freq * scale[:, None, None]
        self.amp[env_ids], self.freq[env_ids], self.phase[env_ids] = amp, freq, phase
        self.center[env_ids] = torch.tensor([0.0, 0.0, c.center_height], device=dev)

    # -------------------------------------------------------------- obs
    def _task_obs(self):
        t = self.episode_time()[:, None] + self.horizon[None]  # (N, H)
        pos, vel, _ = self.reference(t)
        rel = pos - self.est_pos()[:, None, :]
        q = self.est_quat()[:, None, :].expand(-1, rel.shape[1], -1)
        rel_b = qm.quat_rotate_inverse(q, rel).reshape(self.num_envs, -1)
        vel_b = self.world_to_est_body(vel[:, 0])
        return torch.cat([rel_b, vel_b], -1)

    def _task_privileged_obs(self):
        pos, vel, acc = self.reference(self.episode_time()[:, None])
        q = self.robot.data.root_quat_w
        return torch.cat([qm.quat_rotate_inverse(q, pos[:, 0] - self.local_pos()),
                          qm.quat_rotate_inverse(q, vel[:, 0] - self.robot.data.root_lin_vel_w),
                          qm.quat_rotate_inverse(q, acc[:, 0])], -1)

    def debug_points(self):
        t = self.episode_time()[:, None] + torch.linspace(0.0, 2.0, 12, device=self.device)[None]
        pos, _, _ = self.reference(t)
        ids = torch.ones(12, dtype=torch.long, device=self.device)
        ids[0] = 0
        return pos, ids

    # -------------------------------------------------------------- reward / dones
    def _track_errors(self):
        pos, vel, _ = self.reference(self.episode_time()[:, None])
        e_p = (pos[:, 0] - self.local_pos()).norm(dim=-1)
        e_v = (vel[:, 0] - self.robot.data.root_lin_vel_w).norm(dim=-1)
        return e_p, e_v

    def _compute_task_reward(self):
        c, dt = self.cfg, self.step_dt
        e_p, e_v = self._track_errors()
        self.track_err_sum += e_p.square()
        self.track_err_cnt += 1
        self.track_err_max = torch.maximum(self.track_err_max, e_p)
        return {
            "pos": c.w_pos * (1 - torch.tanh(e_p / 0.5)) * dt,
            "pos_fine": c.w_pos_fine * torch.exp(-(e_p / 0.1) ** 2) * dt,
            "vel": -c.w_vel * e_v.square() * dt,
            "ang_vel": -c.w_ang_vel * self.robot.data.root_ang_vel_b.square().sum(-1) * dt,
            "action_rate": -c.w_action_rate * (self.actions - self.prev_actions).square().sum(-1) * dt,
            "crash": -c.w_crash * self.crashed.float(),
        }

    def _task_crash(self):
        e_p, _ = self._track_errors()
        return e_p > self.cfg.max_track_error

    def _task_episode_metrics(self, env_ids):
        rmse = torch.sqrt(self.track_err_sum[env_ids] / self.track_err_cnt[env_ids].clamp_min(1))
        out = {"track_rmse": rmse, "track_max_err": self.track_err_max[env_ids].clone()}
        self.track_err_sum[env_ids] = 0
        self.track_err_cnt[env_ids] = 0
        self.track_err_max[env_ids] = 0
        return out

    # -------------------------------------------------------------- reset
    def _sample_initial_state(self, env_ids):
        k, dev, c = len(env_ids), self.device, self.cfg
        self._sample_reference(env_ids)
        pos0, vel0, _ = self.reference(torch.zeros(k, device=dev), env_ids)
        pos = pos0 + (torch.rand(k, 3, device=dev) * 2 - 1) * c.init_pos_noise
        vel = vel0 + (torch.rand(k, 3, device=dev) * 2 - 1) * c.init_vel_noise
        quat = qm.random_quat_tilt(k, c.init_max_tilt, dev)
        return pos, quat, vel, torch.zeros(k, 3, device=dev)

    def _reset_task(self, env_ids):
        # reference time starts at 0 for every reset env, also after the initial reset de-synchronization
        self.t_start[env_ids] = self.episode_length_buf[env_ids]
