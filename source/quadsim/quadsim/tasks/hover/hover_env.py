"""Hover / stabilization task: recover from a perturbed (possibly tumbling) state and hold a pose."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from quadsim.core import math as qm
from quadsim.tasks.base import QuadrotorEnv, QuadrotorEnvCfg


@configclass
class HoverEnvCfg(QuadrotorEnvCfg):
    episode_length_s: float = 8.0
    # initial-state distribution (relative to the target)
    init_pos_radius: float = 1.5
    init_max_tilt: float = 1.0  # rad
    init_lin_vel: float = 1.5  # m/s per axis (uniform +-)
    init_ang_vel: float = 3.0  # rad/s per axis (uniform +-)
    target_height_range: tuple[float, float] = (1.0, 2.0)
    target_yaw_random: bool = True
    max_target_distance: float = 4.0  # terminate if farther from the target
    # reward weights
    w_pos: float = 1.0
    w_pos_fine: float = 0.5
    w_yaw: float = 0.2
    w_lin_vel: float = 0.05
    w_ang_vel: float = 0.02
    w_action_rate: float = 0.05
    w_crash: float = 10.0
    success_radius: float = 0.1


class HoverEnv(QuadrotorEnv):
    cfg: HoverEnvCfg

    @classmethod
    def task_obs_dim(cls, cfg) -> int:
        return 5

    def __init__(self, cfg: HoverEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_yaw = torch.zeros(self.num_envs, device=self.device)

    # -------------------------------------------------------------- obs
    def _task_obs(self):
        rel_b = self.world_to_est_body(self.target_pos - self.est_pos())
        e_yaw = qm.wrap_to_pi(self.target_yaw - qm.yaw_from_quat(self.est_quat()))
        return torch.cat([rel_b, torch.sin(e_yaw)[:, None], torch.cos(e_yaw)[:, None]], -1)

    def _task_privileged_obs(self):
        d = self.robot.data
        rel_b = qm.quat_rotate_inverse(d.root_quat_w, self.target_pos - self.local_pos())
        return rel_b

    # -------------------------------------------------------------- reward
    def _errors(self):
        e_p = (self.target_pos - self.local_pos()).norm(dim=-1)
        e_yaw = qm.wrap_to_pi(self.target_yaw - qm.yaw_from_quat(self.robot.data.root_quat_w)).abs()
        return e_p, e_yaw

    def _compute_task_reward(self):
        c = self.cfg
        d = self.robot.data
        e_p, e_yaw = self._errors()
        dt = self.step_dt
        return {
            "pos": c.w_pos * (1 - torch.tanh(e_p / 0.8)) * dt,
            "pos_fine": c.w_pos_fine * torch.exp(-(e_p / 0.1) ** 2) * dt,
            "yaw": c.w_yaw * 0.5 * (1 + torch.cos(e_yaw)) * dt,
            "lin_vel": -c.w_lin_vel * d.root_lin_vel_w.square().sum(-1) * dt,
            "ang_vel": -c.w_ang_vel * d.root_ang_vel_b.square().sum(-1) * dt,
            "action_rate": -c.w_action_rate * (self.actions - self.prev_actions).square().sum(-1) * dt,
            "crash": -c.w_crash * self.crashed.float(),
        }

    def _task_crash(self):
        return (self.target_pos - self.local_pos()).norm(dim=-1) > self.cfg.max_target_distance

    # -------------------------------------------------------------- reset
    def _sample_target(self, env_ids):
        k = len(env_ids)
        dev = self.device
        lo, hi = self.cfg.target_height_range
        tgt = torch.zeros(k, 3, device=dev)
        tgt[:, :2] = (torch.rand(k, 2, device=dev) * 2 - 1) * 1.0
        tgt[:, 2] = lo + (hi - lo) * torch.rand(k, device=dev)
        return tgt

    def _sample_initial_state(self, env_ids):
        k = len(env_ids)
        dev = self.device
        c = self.cfg
        self.target_pos[env_ids] = self._sample_target(env_ids)
        self.target_yaw[env_ids] = (torch.rand(k, device=dev) * 2 - 1) * torch.pi if c.target_yaw_random else 0.0
        d = torch.randn(k, 3, device=dev)
        d = d / d.norm(dim=-1, keepdim=True)
        pos = self.target_pos[env_ids] + d * c.init_pos_radius * torch.rand(k, 1, device=dev) ** (1 / 3)
        pos[:, 2] = pos[:, 2].clamp(min=0.5)
        quat = qm.random_quat_tilt(k, c.init_max_tilt, dev)
        lin_vel = (torch.rand(k, 3, device=dev) * 2 - 1) * c.init_lin_vel
        ang_vel = (torch.rand(k, 3, device=dev) * 2 - 1) * c.init_ang_vel
        return pos, quat, lin_vel, ang_vel

    def debug_points(self):
        return self.target_pos[:, None, :], torch.zeros(1, dtype=torch.long, device=self.device)

    def _task_episode_metrics(self, env_ids):
        e_p, e_yaw = self._errors()
        speed = self.robot.data.root_lin_vel_w.norm(dim=-1)
        ok = (e_p < self.cfg.success_radius) & (speed < 0.2) & self.reset_time_outs
        return {"final_pos_err": e_p[env_ids], "final_yaw_err": e_yaw[env_ids], "success": ok[env_ids].float()}


@configclass
class HoverEnvCfg_PLAY(HoverEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.debug_vis = True
