"""Waypoint navigation: fly through a random sequence of waypoints as fast as possible."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from quadsim.core import math as qm
from quadsim.tasks.base import QuadrotorEnv, QuadrotorEnvCfg


@configclass
class WaypointEnvCfg(QuadrotorEnvCfg):
    episode_length_s: float = 15.0
    num_waypoints: int = 5
    area_xy: float = 4.0  # waypoints in [-a, a]^2
    height_range: tuple[float, float] = (1.0, 3.0)
    min_separation: float = 1.5
    reach_radius: float = 0.3
    init_max_tilt: float = 0.3
    # reward weights
    w_progress: float = 1.0
    w_reach: float = 5.0
    w_time: float = 0.1
    w_ang_vel: float = 0.01
    w_action_rate: float = 0.05
    w_crash: float = 10.0
    max_obs_distance: float = 5.0


class WaypointEnv(QuadrotorEnv):
    cfg: WaypointEnvCfg

    @classmethod
    def task_obs_dim(cls, cfg) -> int:
        return 7

    def __init__(self, cfg: WaypointEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, w, dev = self.num_envs, cfg.num_waypoints, self.device
        self.waypoints = torch.zeros(n, w, 3, device=dev)
        self.wp_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev_dist = torch.zeros(n, device=dev)
        self.reached_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self.done_all = torch.zeros(n, dtype=torch.bool, device=dev)
        self.time_to_finish = torch.zeros(n, device=dev)
        self._ar = torch.arange(n, device=dev)
        self.t_start = torch.zeros(n, dtype=torch.long, device=dev)

    def _wp(self, offset=0):
        idx = (self.wp_idx + offset).clamp(max=self.cfg.num_waypoints - 1)
        return self.waypoints[self._ar, idx]

    def _clip_rel(self, rel):
        d = rel.norm(dim=-1, keepdim=True)
        return rel * torch.clamp(self.cfg.max_obs_distance / d.clamp_min(1e-6), max=1.0)

    def debug_points(self):
        w = self.cfg.num_waypoints
        cur = self._wp(0)[:, None, :]
        ids = torch.ones(w + 1, dtype=torch.long, device=self.device)
        ids[0] = 0
        return torch.cat([cur, self.waypoints], 1), ids

    def _task_obs(self):
        p = self.est_pos()
        rel0 = self._clip_rel(self.world_to_est_body(self._wp(0) - p))
        rel1 = self._clip_rel(self.world_to_est_body(self._wp(1) - self._wp(0)))
        remaining = (self.cfg.num_waypoints - self.wp_idx).float()[:, None] / self.cfg.num_waypoints
        return torch.cat([rel0, rel1, remaining], -1)

    def _task_privileged_obs(self):
        return qm.quat_rotate_inverse(self.robot.data.root_quat_w, self._wp(0) - self.local_pos())

    def _compute_task_reward(self):
        c, dt = self.cfg, self.step_dt
        dist = (self._wp(0) - self.local_pos()).norm(dim=-1)
        progress = self.prev_dist - dist
        reached = (dist < c.reach_radius) & ~self.done_all
        self.reached_now = reached
        # advance waypoint index
        last = self.wp_idx == c.num_waypoints - 1
        self.done_all |= reached & last
        t_ep = (self.episode_length_buf - self.t_start).float() * self.step_dt
        self.time_to_finish = torch.where(reached & last, t_ep,
                                          self.time_to_finish)
        self.wp_idx = torch.where(reached & ~last, self.wp_idx + 1, self.wp_idx)
        new_dist = (self._wp(0) - self.local_pos()).norm(dim=-1)
        self.prev_dist = new_dist
        return {
            "progress": c.w_progress * torch.where(reached, torch.zeros_like(progress), progress),
            "reach": c.w_reach * reached.float(),
            "time": -c.w_time * dt * (~self.done_all).float(),
            "ang_vel": -c.w_ang_vel * self.robot.data.root_ang_vel_b.square().sum(-1) * dt,
            "action_rate": -c.w_action_rate * (self.actions - self.prev_actions).square().sum(-1) * dt,
            "crash": -c.w_crash * self.crashed.float(),
        }

    def _task_timeout(self):
        return self.done_all

    def _sample_initial_state(self, env_ids):
        k, dev, c = len(env_ids), self.device, self.cfg
        w = c.num_waypoints
        lo, hi = c.height_range
        wps = torch.zeros(k, w, 3, device=dev)
        prev = torch.zeros(k, 3, device=dev)
        prev[:, 2] = 1.0
        for i in range(w):
            cand = torch.zeros(k, 8, 3, device=dev)
            cand[..., :2] = (torch.rand(k, 8, 2, device=dev) * 2 - 1) * c.area_xy
            cand[..., 2] = lo + (hi - lo) * torch.rand(k, 8, device=dev)
            ok = (cand - prev[:, None]).norm(dim=-1) >= c.min_separation
            first_ok = torch.where(ok.any(-1), ok.float().argmax(-1), torch.zeros(k, dtype=torch.long, device=dev))
            wps[:, i] = cand[torch.arange(k, device=dev), first_ok]
            prev = wps[:, i]
        self.waypoints[env_ids] = wps
        self.wp_idx[env_ids] = 0
        self.done_all[env_ids] = False
        self.time_to_finish[env_ids] = 0.0
        pos = torch.zeros(k, 3, device=dev)
        pos[:, :2] = (torch.rand(k, 2, device=dev) * 2 - 1) * 0.5
        pos[:, 2] = 1.0
        self.prev_dist[env_ids] = (wps[:, 0] - pos).norm(dim=-1)
        quat = qm.random_quat_tilt(k, c.init_max_tilt, dev)
        return pos, quat, torch.zeros(k, 3, device=dev), torch.zeros(k, 3, device=dev)

    def _reset_task(self, env_ids):
        self.t_start[env_ids] = self.episode_length_buf[env_ids]

    def _task_episode_metrics(self, env_ids):
        done = self.done_all[env_ids].float()
        frac = (self.wp_idx[env_ids].float() + done) / self.cfg.num_waypoints
        t = torch.where(done > 0, self.time_to_finish[env_ids], torch.full_like(done, float("nan")))
        return {"success": done, "waypoints_reached_frac": frac, "time_to_finish": t}
