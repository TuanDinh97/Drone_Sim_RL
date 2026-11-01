"""Example: a new task from scratch - precision landing on a pad (subclass of the generic quadrotor env).

    python scripts/rl/train.py --headless --task Quadsim-Example-Land-v0 --plugins examples/custom_task.py \
        --max_iterations 400
    python scripts/rl/evaluate.py --headless --task Quadsim-Example-Land-v0 --plugins examples/custom_task.py \
        --controllers policy --checkpoint logs/rsl_rl/quadsim_example_land/<run>

A task implements five hooks; everything else (vehicle, actuators, onboard controller, sensors with noise and
latency, domain randomization, disturbances, logging, evaluation metrics, export) comes from QuadrotorEnv:
  task_obs_dim / _task_obs        deployable task observation (here: pad position in the estimated body frame)
  _task_privileged_obs            critic-only information
  _compute_task_reward            dict of reward terms (logged individually)
  _task_crash / _task_timeout     failure / success terminations
  _sample_initial_state           per-episode scenario
  _task_episode_metrics           per-episode metrics for evaluation (success, ...)
"""

import gymnasium as gym
import torch

from isaaclab.utils import configclass

from quadsim.core import math as qm
from quadsim.tasks.agents import QuadPPORunnerCfg
from quadsim.tasks.base import QuadrotorEnv, QuadrotorEnvCfg


@configclass
class LandEnvCfg(QuadrotorEnvCfg):
    episode_length_s: float = 8.0
    pad_radius: float = 0.4
    pad_range: float = 1.0
    start_height: tuple[float, float] = (2.0, 3.0)
    touchdown_height: float = 0.12  # CoM height at which the landing gear touches the ground
    max_touchdown_speed: float = 0.6
    w_progress: float = 1.0
    w_success: float = 10.0
    w_crash: float = 10.0
    w_descent: float = 0.5
    w_action_rate: float = 0.05
    w_time: float = 1.0
    """per-second penalty until touchdown (reward v2: without it the first policy learned to hover above the pad)"""
    w_soft: float = 2.0
    """per-second reward for being low *and* slow over the pad (reward v2)"""

    def __post_init__(self):
        super().__post_init__()
        self.min_height = -1.0  # touchdown is handled by the task


class LandEnv(QuadrotorEnv):
    cfg: LandEnvCfg

    @classmethod
    def task_obs_dim(cls, cfg):
        return 3

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.pad = torch.zeros(n, 3, device=dev)
        self.prev_dist = torch.zeros(n, device=dev)
        self.landed_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self.touchdown = torch.zeros(n, dtype=torch.bool, device=dev)
        self.td_speed = torch.full((n,), float("nan"), device=dev)

    def _task_obs(self):
        return self.world_to_est_body(self.pad - self.est_pos())

    def _task_privileged_obs(self):
        return qm.quat_rotate_inverse(self.robot.data.root_quat_w, self.pad - self.local_pos())

    def _task_crash(self):
        p, v = self.local_pos(), self.robot.data.root_lin_vel_w
        self.touchdown = p[:, 2] < self.cfg.touchdown_height
        on_pad = (p[:, :2] - self.pad[:, :2]).norm(dim=-1) < self.cfg.pad_radius
        q = self.robot.data.root_quat_w
        upright = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2) > 0.94  # < 20 deg tilt
        soft = v.norm(dim=-1) < self.cfg.max_touchdown_speed
        self.td_speed = torch.where(self.touchdown, v.norm(dim=-1), self.td_speed)
        self.landed_ok = self.touchdown & on_pad & upright & soft
        return self.touchdown & ~self.landed_ok

    def _task_timeout(self):
        return self.landed_ok

    def _compute_task_reward(self):
        c, dt = self.cfg, self.step_dt
        dist = (self.pad - self.local_pos()).norm(dim=-1)
        progress = self.prev_dist - dist
        self.prev_dist = dist
        vz = self.robot.data.root_lin_vel_w[:, 2]
        z = self.local_pos()[:, 2]
        near = torch.exp(-z / 0.5)  # descent-rate penalty grows near the ground
        over_pad = ((self.local_pos()[:, :2] - self.pad[:, :2]).norm(dim=-1) < c.pad_radius).float()
        soft = over_pad * torch.exp(-z / 0.3) * torch.exp(-(vz / 0.5) ** 2)
        return {"progress": c.w_progress * progress,
                "time": -c.w_time * dt * torch.ones_like(z),
                "low_and_slow": c.w_soft * soft * dt,
                "success": c.w_success * self.landed_ok.float(),
                "descent_rate": -c.w_descent * near * vz.clamp(max=0.0).square() * dt,
                "action_rate": -c.w_action_rate * (self.actions - self.prev_actions).square().sum(-1) * dt,
                "crash": -c.w_crash * self.crashed.float()}

    def _sample_initial_state(self, env_ids):
        k, dev, c = len(env_ids), self.device, self.cfg
        self.pad[env_ids, :2] = (torch.rand(k, 2, device=dev) * 2 - 1) * c.pad_range
        self.pad[env_ids, 2] = 0.0
        pos = torch.zeros(k, 3, device=dev)
        pos[:, :2] = (torch.rand(k, 2, device=dev) * 2 - 1) * 1.5
        pos[:, 2] = c.start_height[0] + (c.start_height[1] - c.start_height[0]) * torch.rand(k, device=dev)
        self.prev_dist[env_ids] = (self.pad[env_ids] - pos).norm(dim=-1)
        self.landed_ok[env_ids] = False
        self.touchdown[env_ids] = False
        self.td_speed[env_ids] = float("nan")
        quat = qm.random_quat_tilt(k, 0.2, dev)
        vel = (torch.rand(k, 3, device=dev) * 2 - 1) * 0.5
        return pos, quat, vel, torch.zeros(k, 3, device=dev)

    def _task_episode_metrics(self, env_ids):
        return {"success": self.landed_ok[env_ids].float(), "touchdown_speed": self.td_speed[env_ids]}

    def baseline_targets(self):
        """Classical baseline: align over the pad at 1 m, then descend slowly (set-point 0.25 m below the estimate,
        i.e. ~0.4 m/s with the cascade's position gain)."""
        z = torch.zeros_like(self.pad)
        tgt = self.pad.clone()
        p = self.est_pos()
        off = (p[:, :2] - self.pad[:, :2]).norm(dim=-1) > 0.15
        tgt[:, 2] = torch.where(off, torch.full_like(tgt[:, 2], 1.0), p[:, 2] - 0.25)
        return tgt, z, z, torch.zeros(self.num_envs, device=self.device)


@configclass
class LandPPORunnerCfg(QuadPPORunnerCfg):
    experiment_name = "quadsim_example_land"
    max_iterations = 400


gym.register(id="Quadsim-Example-Land-v0", entry_point=LandEnv, disable_env_checker=True,
             kwargs={"env_cfg_entry_point": LandEnvCfg, "rsl_rl_cfg_entry_point": LandPPORunnerCfg})
