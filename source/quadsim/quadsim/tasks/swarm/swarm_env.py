"""Cooperative multi-agent task: groups of drones swap positions without collisions (decentralized execution).

Multi-agent support without a separate MARL environment class: ``agents_per_group`` consecutive environments form one
group that shares a local frame (``env_origins``), episode clock and resets. Each agent is one row of the vectorized
environment, so any single-agent algorithm becomes *parameter-sharing* multi-agent RL (one policy, decentralized
execution); the critic observation is centralized (true states and goals of the whole group, MAPPO-style).

Scenario (antipodal swap, a standard multi-robot coordination benchmark): the agents of a group start equally spaced
on a horizontal circle (random radius, rotation and per-agent height) and must reach the antipodal point, so all
straight-line paths meet in the centre. Inter-agent collisions (centre distance < 2 ``agent_radius``) terminate
both agents; any termination resets the whole group (non-crashed members are truncated, not failed).

Policy observation (deployable with inter-vehicle communication of state estimates): own state, goal relative to
the vehicle and, for the other group members sorted by distance, their relative position and velocity in the
estimated body frame (from their *estimates*, as broadcast over a radio link).
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from quadsim.core import math as qm
from quadsim.tasks.base import QuadrotorEnv, QuadrotorEnvCfg


@configclass
class SwarmEnvCfg(QuadrotorEnvCfg):
    episode_length_s: float = 10.0
    agents_per_group: int = 4
    circle_radius: tuple[float, float] = (2.0, 3.0)
    height_range: tuple[float, float] = (1.2, 2.2)
    agent_radius: float = 0.4
    """collision radius: x500 rotor-tip extent 0.37 m, so the analytic check fires before frames touch"""
    safe_distance: float = 1.2
    goal_radius: float = 0.3
    max_obs_distance: float = 6.0
    # reward
    w_progress: float = 1.0
    w_goal: float = 0.5
    w_separation: float = 1.0
    w_crash: float = 10.0
    w_action_rate: float = 0.02

    def __post_init__(self):
        super().__post_init__()
        self.control.mode = "velocity"
        self.scene.env_spacing = 10.0
        self.max_distance = 8.0


class SwarmEnv(QuadrotorEnv):
    cfg: SwarmEnvCfg

    @classmethod
    def task_obs_dim(cls, cfg) -> int:
        return 3 + 6 * (cfg.agents_per_group - 1)

    def __init__(self, cfg: SwarmEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        m, n, dev = cfg.agents_per_group, self.num_envs, self.device
        if n % m:
            raise ValueError(f"num_envs ({n}) must be a multiple of agents_per_group ({m})")
        self.m, self.g = m, n // m
        self.member = torch.arange(n, device=dev) % m
        leader = torch.arange(n, device=dev) - self.member
        self.env_origins = self.scene.env_origins[leader].clone()  # the whole group shares the leader's frame
        self.goal = torch.zeros(n, 3, device=dev)
        self.prev_dist = torch.zeros(n, device=dev)
        self.agent_collision = torch.zeros(n, dtype=torch.bool, device=dev)
        self.min_sep = torch.full((n,), 1e3, device=dev)
        self.reached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.t_reach = torch.full((n,), float("nan"), device=dev)
        others = torch.stack([torch.cat([torch.arange(0, i), torch.arange(i + 1, m)]) for i in range(m)])
        self._others = others.to(dev)  # (M, M-1) member index of the other agents

    # ------------------------------------------------------------------ group helpers
    def _group(self, x):
        return x.view(self.g, self.m, *x.shape[1:])

    def _others_of(self, x):
        """(N, M-1, ...) values of the other members of each agent's group."""
        xg = self._group(x)
        idx = self._others[None].expand(self.g, -1, -1)  # (G, M, M-1)
        out = torch.gather(xg[:, None].expand(-1, self.m, -1, *x.shape[1:]), 2,
                           idx.reshape(self.g, self.m, self.m - 1, *([1] * (x.dim() - 1))).expand(
                               -1, -1, -1, *x.shape[1:]))
        return out.reshape(self.num_envs, self.m - 1, *x.shape[1:])

    def _pair_distances(self, pos):
        pg = self._group(pos)
        return (pg[:, :, None] - pg[:, None]).norm(dim=-1)  # (G, M, M)

    # ------------------------------------------------------------------ observations
    def _task_obs(self):
        p, v = self.est_pos(), self.odom_meas[:, 7:10]
        goal = self.world_to_est_body(self.goal - p)
        rel_p = self._others_of(p) - p[:, None]
        rel_v = self._others_of(v) - v[:, None]
        order = rel_p.norm(dim=-1).argsort(-1)
        rel_p = torch.gather(rel_p, 1, order[..., None].expand(-1, -1, 3))
        rel_v = torch.gather(rel_v, 1, order[..., None].expand(-1, -1, 3))
        q = self.est_quat()[:, None].expand(-1, self.m - 1, -1)
        rp_b, rv_b = qm.quat_rotate_inverse(q, rel_p), qm.quat_rotate_inverse(q, rel_v)
        d = rp_b.norm(dim=-1, keepdim=True)
        rp_b = rp_b * torch.clamp(self.cfg.max_obs_distance / d.clamp_min(1e-6), max=1.0)
        gd = goal.norm(dim=-1, keepdim=True)
        goal = goal * torch.clamp(self.cfg.max_obs_distance / gd.clamp_min(1e-6), max=1.0)
        return torch.cat([goal, rp_b.flatten(1), rv_b.flatten(1)], -1)

    def _task_privileged_obs(self):
        """Centralized critic: true relative states and goals of the whole group."""
        p, v = self.local_pos(), self.robot.data.root_lin_vel_w
        return torch.cat([self.goal - p, (self._others_of(p) - p[:, None]).flatten(1),
                          (self._others_of(v) - v[:, None]).flatten(1),
                          (self._others_of(self.goal) - p[:, None]).flatten(1)], -1)

    # ------------------------------------------------------------------ dynamics of the task
    def _task_crash(self):
        d = self._pair_distances(self.local_pos())
        d = d + torch.eye(self.m, device=self.device) * 1e3
        nearest = d.min(-1).values.reshape(-1)
        self.min_sep = torch.minimum(self.min_sep, nearest)
        self.agent_collision = nearest < 2 * self.cfg.agent_radius
        return self.agent_collision

    def _get_dones(self):
        # one clock per group (rsl_rl randomizes episode_length_buf at the start of training)
        self.episode_length_buf[:] = self._group(self.episode_length_buf)[:, :1].expand(-1, self.m).reshape(-1)
        crashed, time_out = super()._get_dones()
        group_done = self._group(crashed | time_out).any(-1, keepdim=True).expand(-1, self.m).reshape(-1)
        return crashed, (time_out | group_done) & ~crashed

    def _compute_task_reward(self):
        c, dt = self.cfg, self.step_dt
        p = self.local_pos()
        dist = (self.goal - p).norm(dim=-1)
        progress = self.prev_dist - dist
        self.prev_dist = dist
        at_goal = dist < c.goal_radius
        t = self.episode_length_buf.float() * dt
        self.t_reach = torch.where(at_goal & ~self.reached, t, self.t_reach)
        self.reached |= at_goal
        d = self._pair_distances(p) + torch.eye(self.m, device=self.device) * 1e3
        near = (c.safe_distance - d).clamp(min=0.0) / (c.safe_distance - 2 * c.agent_radius)
        return {
            "progress": c.w_progress * progress,
            "goal": c.w_goal * at_goal.float() * dt,
            "separation": -c.w_separation * near.square().sum(-1).reshape(-1) * dt,
            "action_rate": -c.w_action_rate * (self.actions - self.prev_actions).square().sum(-1) * dt,
            "crash": -c.w_crash * self.crashed.float(),
        }

    # ------------------------------------------------------------------ reset (whole groups)
    def _sample_initial_state(self, env_ids):
        c, dev = self.cfg, self.device
        k = len(env_ids)
        groups = env_ids // self.m
        ug, inv = torch.unique(groups, return_inverse=True)
        ng = len(ug)
        radius = c.circle_radius[0] + (c.circle_radius[1] - c.circle_radius[0]) * torch.rand(ng, device=dev)
        rot = torch.rand(ng, device=dev) * 2 * math.pi
        mem = self.member[env_ids]
        th = rot[inv] + 2 * math.pi * mem.float() / self.m
        r = radius[inv]
        lo, hi = c.height_range
        z0 = lo + (hi - lo) * torch.rand(k, device=dev)
        z1 = lo + (hi - lo) * torch.rand(k, device=dev)
        pos = torch.stack([r * torch.cos(th), r * torch.sin(th), z0], -1)
        self.goal[env_ids] = torch.stack([-r * torch.cos(th), -r * torch.sin(th), z1], -1)
        self.prev_dist[env_ids] = (self.goal[env_ids] - pos).norm(dim=-1)
        self.reached[env_ids] = False
        self.t_reach[env_ids] = float("nan")
        self.min_sep[env_ids] = 1e3
        self.agent_collision[env_ids] = False
        yaw = th + math.pi  # facing the goal
        quat = qm.quat_from_euler_xyz(torch.zeros(k, device=dev), torch.zeros(k, device=dev), yaw)
        return pos, quat, torch.zeros(k, 3, device=dev), torch.zeros(k, 3, device=dev)

    def _task_episode_metrics(self, env_ids):
        dist = (self.goal[env_ids] - self.local_pos()[env_ids]).norm(dim=-1)
        ok = (dist < self.cfg.goal_radius).float()
        coll = self.agent_collision[env_ids].float()
        # group-level: every member at its goal and no member crashed (computed on complete groups)
        grp_ok = torch.full_like(ok, float("nan"))
        if len(env_ids) % self.m == 0:
            crashed = self.reset_terminated[env_ids].float()
            g_ok = (ok.view(-1, self.m).min(-1).values * (1 - crashed.view(-1, self.m).max(-1).values))
            grp_ok = g_ok[:, None].expand(-1, self.m).reshape(-1)
        return {"success": grp_ok, "agent_at_goal": ok, "agent_collision": coll,
                "min_separation": self.min_sep[env_ids], "time_to_goal": self.t_reach[env_ids]}

    # ------------------------------------------------------------------ baselines
    def baseline_targets(self):
        """Straight-line baseline (no coordination): every agent flies directly to its own goal."""
        z = torch.zeros_like(self.goal)
        d = self.goal - self.est_pos()
        return self.goal, z, z, torch.atan2(d[:, 1], d[:, 0])

    def debug_points(self):
        return self.goal[:, None], torch.zeros(1, dtype=torch.long, device=self.device)


class SwarmPotentialBaseline:
    """Classical decentralized baseline (velocity mode): goal attraction + inverse-distance repulsion from the
    communicated positions of the other group members (artificial potential field). Gains from a 3-point sweep on the
    4-agent swap (influence 1.5 m: 0 % success / 53 % collisions; 2.5 m: 100 % / 0 %; docs/PROGRESS.md)."""

    def __init__(self, env, speed: float = 1.5, influence: float = 2.5, k_rep: float = 4.0):
        self.env, self.speed, self.influence, self.k_rep = env, speed, influence, k_rep

    def reset(self, env_ids=None):
        pass

    def __call__(self):
        env = self.env
        p = env.est_pos()
        q = env.est_quat()
        goal = env.goal - p
        dist = goal.norm(dim=-1, keepdim=True)
        v = goal / dist.clamp_min(1e-6) * dist.clamp(max=self.speed)
        rel = env._others_of(p) - p[:, None]  # (N, M-1, 3)
        d = rel.norm(dim=-1, keepdim=True).clamp_min(0.05)
        w = (1.0 / d - 1.0 / self.influence).clamp(min=0.0)
        v = v - self.k_rep * (w * rel / d).sum(1)
        n = v.norm(dim=-1, keepdim=True)
        v = v * torch.clamp(self.speed / n.clamp_min(1e-6), max=1.0)
        yaw = qm.yaw_from_quat(q)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        v_h = torch.stack([cy * v[:, 0] + sy * v[:, 1], -sy * v[:, 0] + cy * v[:, 1], v[:, 2]], -1)
        return torch.cat([v_h / env.max_vel, torch.zeros(env.num_envs, 1, device=env.device)], -1).clamp(-1, 1)
