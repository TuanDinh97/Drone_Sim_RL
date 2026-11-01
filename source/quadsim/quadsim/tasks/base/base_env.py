"""Generic quadrotor environment on Isaac Lab (DirectRLEnv).

Per physics step (``_apply_action``, 200 Hz by default):
    IMU/odometry sampling -> delayed policy command -> onboard controller(s) -> mixer
    -> rotor-speed command -> motor dynamics -> aerodynamic wrench (+ wind, pushes, payload)
    -> body-frame wrench written to PhysX -> PhysX integrates the rigid body + contacts.

Tasks subclass :class:`QuadrotorEnv` and implement ``_task_obs``, ``_task_privileged_obs``,
``_compute_task_reward``, ``_task_dones`` and ``_reset_task``. Without subclassing, any task can be extended through
its configuration: vehicle (``cfg.dynamics.preset``: registered name or vehicle file), action interface
(``cfg.control.mode``: registered :class:`~quadsim.core.controllers.interfaces.ActionInterface`), extra
observation / reward / termination / cost / event / curriculum terms (``cfg.terms``) and constrained RL
(``cfg.safety``), see :mod:`quadsim.tasks.base.terms`.

Observation groups returned by ``_get_observations``:
    ``policy``  deployable observations only (noisy/delayed estimates, gyro, last action, task info)
    ``critic``  policy obs + privileged simulator state (true state, rotor speeds, wind, DR params)
    ``image``   (optional) (N, C, H, W) camera tensor for vision policies
"""

from __future__ import annotations

import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv

from quadsim.core import QuadrotorModel
from quadsim.core import math as qm
from quadsim.core.controllers.interfaces import FlightController
from quadsim.core.disturbances import PushDisturbance, WindModel
from quadsim.core.sensors import (
    BarometerModel,
    DelayBuffer,
    ImuModel,
    OdometryModel,
    RgbAugment,
    apply_depth_noise,
    cast,
    normalize_depth,
    pinhole_ray_dirs,
)
from quadsim.core.sensors.noise import sample_uniform
from quadsim.core.vehicles import get_vehicle
from quadsim.registry import ACTION_INTERFACES

from .base_env_cfg import QuadrotorEnvCfg
from .terms import TermManager


class QuadrotorEnv(DirectRLEnv):
    cfg: QuadrotorEnvCfg

    def __init__(self, cfg: QuadrotorEnvCfg, render_mode: str | None = None, **kwargs):
        # resolve vehicle preset (allows CLI/Hydra overrides of cfg.dynamics.preset)
        self.params = get_vehicle(cfg.dynamics.preset)
        cfg.robot.spawn.params = self.params
        self._iface_cls = ACTION_INTERFACES.get(cfg.control.mode)
        self._configure_spaces(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.env_origins = self.scene.env_origins.clone()
        """Origin of each env's local frame (tasks may let several envs share one, e.g. multi-agent groups)."""

        # ---------------- physics model (per-env tensors) ----------------
        p = self.params
        self.model = QuadrotorModel(p, n, dev)
        if not cfg.dynamics.rotor_drag:
            self.model.rotor_drag_coeff[:] = 0.0
            self.model.rolling_moment_coeff[:] = 0.0
        if not cfg.dynamics.body_drag:
            self.model.body_drag_lin[:] = 0.0
            self.model.body_drag_quad[:] = 0.0
        if not cfg.dynamics.rotor_gyroscopics:
            self.model.rotor_inertia[:] = 0.0
        self.model.ground_effect = cfg.dynamics.ground_effect
        self._nominal = {k: getattr(self.model, k).clone() for k in
                         ("k_f", "k_m", "tau_up", "tau_down", "omega_max", "rotor_drag_coeff", "rolling_moment_coeff",
                          "body_drag_lin", "body_drag_quad")}
        self.payload_force_w = torch.zeros(n, 3, device=dev)
        r = cfg.randomization
        self._dr_extra = [k for k in ("rate_gain_scale", "thrust_gain_scale") if getattr(r, k) is not None]
        self.dr_obs = torch.zeros(n, 8 + len(self._dr_extra), device=dev)  # privileged: normalized DR parameters
        self._randomize_inertial_properties()

        # ---------------- onboard flight controller (nominal model) + action interface ----------------
        c = cfg.control
        self.fc = FlightController(p, c, n, self.physics_dt, dev, thrust_gain_enabled=r.thrust_gain_scale is not None)
        self.action_iface = self._iface_cls(self.fc)
        # aliases (baselines, validation scripts and tasks use them)
        self.mixer, self.rate_ctrl, self.att_ctrl, self.pos_ctrl = fc_parts = (
            self.fc.mixer, self.fc.rate_ctrl, self.fc.att_ctrl, self.fc.pos_ctrl)
        del fc_parts
        self.max_rate, self.max_vel, self.t_max_nominal = self.fc.max_rate, self.fc.max_vel, self.fc.t_max_nominal
        self.thrust_gain = self.fc.thrust_gain
        self.action_dim = gym.spaces.flatdim(self.single_action_space)
        self.actions = torch.zeros(n, self.action_dim, device=dev)
        self.prev_actions = torch.zeros_like(self.actions)
        self.action_delay = DelayBuffer(n, max(c.action_delay_steps[1], 0), (self.action_dim,), dev)
        self.hover_action = self._hover_action()

        # ---------------- sensors ----------------
        s = cfg.sensors
        if not s.noise_enabled:
            s.imu.enabled = False
            s.odometry.enabled = False
            s.baro.enabled = False
            s.imu.delay_steps_max = 0
            s.odometry.delay_steps_max = 0
        self.imu = ImuModel(s.imu, n, self.physics_dt, dev)
        self.odom = OdometryModel(s.odometry, n, self.physics_dt, dev)
        self.baro = BarometerModel(s.baro, n, self.physics_dt, dev)
        self.imu_meas = torch.zeros(n, 6, device=dev)
        self.odom_meas = torch.zeros(n, 13, device=dev)
        self.baro_meas = torch.zeros(n, 1, device=dev)

        # ---------------- disturbances ----------------
        self.wind = WindModel(cfg.wind, n, dev)
        self.wind_w = torch.zeros(n, 3, device=dev)
        self.push = PushDisturbance(cfg.push, n, self.model.mass * 9.81, dev)

        # ---------------- vision ----------------
        v = cfg.vision
        self.vision_enabled = v.mode != "none"
        if self.vision_enabled:
            self._cam_dirs_c = pinhole_ray_dirs(v.width, v.height, v.hfov_deg, dev)
            pitch = math.radians(v.pitch_deg)
            self._cam_quat_b = torch.tensor([math.cos(-pitch / 2), 0.0, math.sin(-pitch / 2), 0.0], device=dev)
            self._cam_pos_b = torch.tensor(v.offset_pos, device=dev)
            self.rgb_aug = RgbAugment(v.rgb_noise, n, dev)
            ch = {"depth": 1, "rgb": 3, "rgbd": 4}[v.mode]
            self.image = torch.zeros(n, ch, v.height, v.width, device=dev)

        # ---------------- bookkeeping ----------------
        self.episode_sums: dict[str, torch.Tensor] = {}
        self.collect_episode_metrics = False  # evaluation scripts switch this on
        self.episode_metric_buffer: list[dict] = []
        self.crashed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_episode_metrics: dict[str, torch.Tensor] = {}
        """Per-env metrics of the most recently finished episodes (NaN where none finished yet); curricula use it."""

        # ---------------- configurable extra terms ----------------
        self.terms = TermManager(self, cfg.terms, cfg.safety)
        if self.terms.policy_dim:
            self._refresh_observation_space()
        self.set_debug_vis(self.cfg.debug_vis)

    # ======================================================================== setup
    @classmethod
    def task_obs_dim(cls, cfg: QuadrotorEnvCfg) -> int:
        """Size of the task-specific part of the policy observation (override in tasks)."""
        return 0

    def _configure_spaces(self, cfg: QuadrotorEnvCfg):
        """Derive gym spaces from the configuration (before DirectRLEnv builds them)."""
        cfg.action_space = self._iface_cls.action_dim(cfg.control, self.params)
        cfg.observation_space = 12 + (cfg.action_space if cfg.include_last_action else 0) + self.task_obs_dim(cfg)

    def _refresh_observation_space(self):
        """Grow the policy observation space by the configured extra (deployable) observation terms."""
        from isaaclab.envs.utils.spaces import spec_to_gym_space

        self.cfg.observation_space = int(self.cfg.observation_space) + self.terms.policy_dim
        self.single_observation_space["policy"] = spec_to_gym_space(self.cfg.observation_space)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)

    def obs_layout(self) -> list[list]:
        """Named blocks of the policy observation ``[[name, dim], ...]`` (datasets, BC policies, export)."""
        out = [["state", 12]] + ([["last_action", self.action_dim]] if self.cfg.include_last_action else [])
        if self.task_obs_dim(self.cfg):
            out.append(["task", self.task_obs_dim(self.cfg)])
        out += [[o.name, o.dim] for o in self.terms.policy_obs]
        return out

    @property
    def yaw_sp(self) -> torch.Tensor:
        return self.fc.yaw_sp

    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot)
        self.scene.rigid_objects["robot"] = self.robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._setup_task_scene()
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)
        sun = sim_utils.DistantLightCfg(intensity=1500.0, angle=1.0)
        sun.func("/World/Sun", sun, orientation=(0.9, 0.2, 0.3, 0.0))

    def _setup_task_scene(self):
        """Hook for tasks to add obstacles, cameras, markers (before cloning)."""

    # ======================================================================== DR
    def _randomize_inertial_properties(self):
        """Startup randomization of mass / inertia (written to PhysX) and CoM offset (model side)."""
        dr = self.cfg.randomization
        n, dev = self.num_envs, self.device
        view = self.robot.root_physx_view
        masses = view.get_masses().clone()  # (N, 1) cpu
        inertias = view.get_inertias().clone()  # (N, 9) cpu
        base_m = torch.full((n,), self.params.mass)
        base_i = torch.tensor(self.params.inertia).repeat(n, 1)
        if dr.enabled:
            ms = sample_uniform(*dr.mass_scale, (n,), "cpu")
            is_ = sample_uniform(*dr.inertia_scale, (n, 3), "cpu")
            base_m = base_m * ms
            base_i = base_i * is_
            self.model.com_offset[:, :2] = sample_uniform(-dr.com_offset_xy, dr.com_offset_xy, (n, 2), dev)
        masses[:, 0] = base_m
        inertias[:] = 0.0
        inertias[:, 0], inertias[:, 4], inertias[:, 8] = base_i[:, 0], base_i[:, 1], base_i[:, 2]
        ids = torch.arange(n, dtype=torch.int32)
        view.set_masses(masses, ids)
        view.set_inertias(inertias, ids)
        self.model.mass[:] = base_m.to(dev)
        self.model.inertia[:] = base_i.to(dev)
        self.dr_obs[:, 0] = self.model.mass / self.params.mass - 1.0

    def _randomize_episode(self, env_ids):
        dr = self.cfg.randomization
        k = len(env_ids)
        r = self.model.num_rotors
        dev = self.device
        nom = self._nominal
        if not dr.enabled:
            for key, val in nom.items():
                getattr(self.model, key)[env_ids] = val[env_ids]
            self.payload_force_w[env_ids] = 0.0
            return
        kf = sample_uniform(*dr.k_f_scale, (k, r), dev)
        km = sample_uniform(*dr.k_m_scale, (k, r), dev)
        tau = sample_uniform(*dr.motor_tau_scale, (k, 1), dev)
        wmax = sample_uniform(*dr.omega_max_scale, (k, 1), dev)
        drag = sample_uniform(*dr.drag_scale, (k, 1), dev)
        self.model.k_f[env_ids] = nom["k_f"][env_ids] * kf
        self.model.k_m[env_ids] = nom["k_m"][env_ids] * km
        self.model.tau_up[env_ids] = nom["tau_up"][env_ids] * tau
        self.model.tau_down[env_ids] = nom["tau_down"][env_ids] * tau
        self.model.omega_max[env_ids] = nom["omega_max"][env_ids] * wmax
        self.model.rotor_drag_coeff[env_ids] = nom["rotor_drag_coeff"][env_ids] * drag[:, 0]
        self.model.rolling_moment_coeff[env_ids] = nom["rolling_moment_coeff"][env_ids] * drag[:, 0]
        self.model.body_drag_lin[env_ids] = nom["body_drag_lin"][env_ids] * drag
        self.model.body_drag_quad[env_ids] = nom["body_drag_quad"][env_ids] * drag
        payload = sample_uniform(*dr.payload_weight_frac, (k,), dev)
        self.payload_force_w[env_ids] = 0.0
        self.payload_force_w[env_ids, 2] = -payload * self.model.mass[env_ids] * 9.81
        for j, key in enumerate(self._dr_extra):
            g = sample_uniform(*getattr(dr, key), (k,), dev)
            if key == "rate_gain_scale":
                self.rate_ctrl.gain_scale[env_ids] = g[:, None]
            else:
                self.thrust_gain[env_ids] = g
            self.dr_obs[env_ids, 8 + j] = g - 1.0
        ns = sample_uniform(*dr.sensor_noise_scale, (k, 1), dev)
        self.imu.noise_scale[env_ids] = ns
        self.odom.noise_scale[env_ids] = ns
        # privileged DR summary for the critic
        self.dr_obs[env_ids, 0] = self.model.mass[env_ids] / self.params.mass - 1.0
        self.dr_obs[env_ids, 1] = kf.mean(-1) - 1.0
        self.dr_obs[env_ids, 2] = km.mean(-1) - 1.0
        self.dr_obs[env_ids, 3] = tau[:, 0] - 1.0
        self.dr_obs[env_ids, 4] = wmax[:, 0] - 1.0
        self.dr_obs[env_ids, 5] = drag[:, 0] - 1.0
        self.dr_obs[env_ids, 6] = payload
        self.dr_obs[env_ids, 7] = ns[:, 0] - 1.0

    # ======================================================================== actions
    def _hover_action(self) -> torch.Tensor:
        return self.action_iface.hover_action()

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions[:] = torch.nan_to_num(actions, nan=0.0).clamp(-1.0, 1.0)
        if self.terms.cfg.events:
            self.terms.interval_events()

    def _collective_from_action(self, a0: torch.Tensor) -> torch.Tensor:
        return self.fc.collective_from_action(a0)

    def _controller(self, a: torch.Tensor, gyro_b: torch.Tensor) -> torch.Tensor:
        """Delayed normalized action -> per-rotor thrust commands (N, R) using onboard estimates."""
        est = self.odom_meas
        quat = qm.quat_normalize(est[:, 3:7]) if self.cfg.sensors.odometry.enabled else self.robot.data.root_quat_w
        return self.action_iface(a, quat, est, gyro_b)

    def _apply_action(self):
        d = self.robot.data
        quat, pos, vel_w, angvel_b = d.root_quat_w, d.root_pos_w, d.root_lin_vel_w, d.root_ang_vel_b
        # --- sensors (sampled at physics rate, internally held at their own rates) ---
        self.imu_meas = self.imu.step(quat, vel_w, angvel_b)
        self.odom_meas = self.odom.step(pos - self.env_origins, quat, vel_w, self.imu_meas[:, :3]
                                        if self.cfg.sensors.odometry.ang_vel_from_imu else angvel_b)
        self.baro_meas = self.baro.step(pos[:, 2] - self.env_origins[:, 2])
        # --- delayed command & onboard control ---
        a = self.action_delay.push(self.actions)
        gyro = self.imu_meas[:, :3] if self.cfg.control.use_imu_gyro_for_rate_loop else angvel_b
        thrust_cmd = self._controller(a, gyro)
        omega_cmd = self.mixer.thrust_to_omega(thrust_cmd)
        self.model.step_motors(omega_cmd, self.physics_dt)
        # --- aerodynamics & disturbances ---
        self.wind_w = self.wind.step(self.physics_dt)
        lin_vel_b = qm.quat_rotate_inverse(quat, vel_w)
        wind_b = qm.quat_rotate_inverse(quat, self.wind_w)
        heights = None
        if self.model.ground_effect:
            arms_w = qm.quat_rotate(quat[:, None, :].expand(-1, self.model.num_rotors, -1), self.model.lever_arms())
            heights = pos[:, None, 2] + arms_w[..., 2] - self.env_origins[:, None, 2]
        w = self.model.compute_wrench(lin_vel_b, angvel_b, wind_b, heights)
        push_f_w, push_t_b = self.push.step(self.physics_dt)
        force_b = w.force + qm.quat_rotate_inverse(quat, push_f_w + self.payload_force_w)
        torque_b = w.torque + push_t_b
        self.last_wrench = w
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=force_b[:, None, :].contiguous(), torques=torque_b[:, None, :].contiguous()
        )

    # ======================================================================== observations
    def _state_obs(self) -> torch.Tensor:
        """Deployable proprioceptive observation (body frame, from onboard estimates)."""
        est = self.odom_meas
        quat = qm.quat_normalize(est[:, 3:7])
        rot = qm.quat_to_rotmat(quat)
        vel_b = qm.quat_rotate_inverse(quat, est[:, 7:10])
        parts = [vel_b, rot[..., :2].reshape(-1, 6), self.imu_meas[:, :3]]
        if self.cfg.include_last_action:
            parts.append(self.actions)
        return torch.cat(parts, -1)

    def _privileged_obs(self) -> torch.Tensor:
        d = self.robot.data
        rot = qm.quat_to_rotmat(d.root_quat_w)
        omega_n = self.model.omega / self.params.omega_max
        wind_b = qm.quat_rotate_inverse(d.root_quat_w, self.wind_w)
        push_b = qm.quat_rotate_inverse(d.root_quat_w, self.push.force) / (self.params.weight + 1e-6)
        return torch.cat([
            d.root_lin_vel_b, d.root_ang_vel_b, rot[..., :2].reshape(-1, 6),
            (d.root_pos_w[:, 2:3] - self.env_origins[:, 2:3]),
            omega_n, wind_b, push_b, self.dr_obs,
        ], -1)

    def _get_observations(self) -> dict:
        task_obs = self._task_obs()
        policy = torch.cat([self._state_obs(), task_obs, *self.terms.policy_terms()], -1)
        critic = torch.cat([policy, self._privileged_obs(), self._task_privileged_obs(), *self.terms.critic_terms()],
                           -1)
        obs = {"policy": torch.nan_to_num(policy), "critic": torch.nan_to_num(critic)}
        if self.vision_enabled:
            if self.common_step_counter % self.cfg.vision.update_period_steps == 0:
                self.image = self._render_image()
            obs["image"] = self.image
        return obs

    # ---------------------------------------------------------------- vision
    def camera_pose_w(self):
        d = self.robot.data
        q = qm.quat_mul(d.root_quat_w, self._cam_quat_b.expand(self.num_envs, 4))
        p = d.root_pos_w + qm.quat_rotate(d.root_quat_w, self._cam_pos_b.expand(self.num_envs, 3))
        return p, q

    def obstacle_primitives(self):
        """(cylinders (N,K,4), boxes (N,M,7)) in env-local coordinates; tasks with obstacles override."""
        return None, None

    def raycast_depth(self) -> torch.Tensor:
        """Analytic distance-to-image-plane depth (N, H, W) of the onboard camera."""
        v = self.cfg.vision
        p, q = self.camera_pose_w()
        p = p - self.env_origins
        n, r = self.num_envs, self._cam_dirs_c.shape[0]
        dirs_w = qm.quat_rotate(q[:, None, :].expand(n, r, 4), self._cam_dirs_c[None].expand(n, r, 3))
        origins = p[:, None, :].expand(n, r, 3)
        cyl, box = self.obstacle_primitives()
        if box is None and "cuda" in str(self.device):
            from quadsim.core.sensors.raycast_warp import cast_warp

            t = cast_warp(origins, dirs_w, cyl, ground=True, max_range=1e3)  # O(N*R) memory
        else:
            t = cast(origins, dirs_w, cyl, box, ground=True, max_range=1e3)
        depth = t * self._cam_dirs_c[None, :, 0]  # range -> distance to image plane
        return depth.view(n, v.height, v.width)

    def _render_image(self) -> torch.Tensor:
        v = self.cfg.vision
        if v.source == "raycast":
            if v.mode != "depth":
                raise ValueError("raycast vision source only provides depth; use source='render' for RGB")
            depth = apply_depth_noise(self.raycast_depth(), v.depth_noise)
            return normalize_depth(depth, v.depth_noise.max_range)[:, None]
        return self._render_camera()

    def _render_camera(self) -> torch.Tensor:
        raise NotImplementedError("rendered cameras are provided by tasks that add a TiledCamera")

    # ======================================================================== rewards/dones
    def _get_rewards(self) -> torch.Tensor:
        terms = self._compute_task_reward()
        if self.terms.cfg.rewards:
            terms.update(self.terms.rewards())
        reward = torch.zeros(self.num_envs, device=self.device)
        for k, v in terms.items():
            reward += v
            self.episode_sums.setdefault(k, torch.zeros(self.num_envs, device=self.device))
            self.episode_sums[k] += v
        if self.terms.has_costs:
            self.extras["cost"] = self.terms.compute_cost()
            self.extras["raw_reward"] = reward
            if self.cfg.safety.lagrangian:
                reward = self.terms.shape_reward(reward)
        return reward

    def _generic_crash(self) -> torch.Tensor:
        d = self.robot.data
        local = d.root_pos_w - self.env_origins
        tilt = torch.acos((1 - 2 * (d.root_quat_w[:, 1] ** 2 + d.root_quat_w[:, 2] ** 2)).clamp(-1, 1))
        crash = (local[:, 2] < self.cfg.min_height) | (local.norm(dim=-1) > self.cfg.max_distance)
        crash |= tilt > self.cfg.max_tilt_terminate
        crash |= d.root_lin_vel_w.norm(dim=-1) > self.cfg.max_lin_vel
        crash |= d.root_ang_vel_b.norm(dim=-1) > self.cfg.max_ang_vel
        crash |= ~torch.isfinite(d.root_state_w).all(-1)
        return crash

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.crashed = self._generic_crash() | self._task_crash()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out |= self._task_timeout()
        if self.terms.cfg.terminations:
            fail, trunc = self.terms.terminations()
            self.crashed |= fail
            time_out |= trunc
        return self.crashed, time_out & ~self.crashed

    # ======================================================================== reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        # episode logging (per-env metrics -> means for TensorBoard, raw values for evaluation)
        log = {}
        for k, v in self.episode_sums.items():
            log[f"Episode_Reward/{k}"] = (v[env_ids].mean() / self.max_episode_length_s).item()
            v[env_ids] = 0.0
        played = self.episode_length_buf[env_ids] > 0
        metrics = {
            "crash": self.reset_terminated[env_ids].float(),
            "time_out": self.reset_time_outs[env_ids].float(),
            "episode_length_s": self.episode_length_buf[env_ids].float() * self.step_dt,
        }
        metrics.update(self._task_episode_metrics(env_ids))
        if self.terms.has_costs:
            metrics["cost"] = self.terms.episode_cost[env_ids].clone()
        for k, v in metrics.items():
            vals = v[played & torch.isfinite(v)]
            if len(vals) > 0:
                log[f"Metrics/{k}"] = vals.mean().item()
            buf = self.last_episode_metrics.setdefault(k, torch.full((self.num_envs,), float("nan"), device=self.device))
            buf[env_ids[played]] = v[played].float()
        log.update(self.terms.before_reset(env_ids, played))
        if self.collect_episode_metrics and played.any():
            ids = env_ids[played]
            self.episode_metric_buffer.append({"env_id": ids.cpu(), **{k: v[played].cpu() for k, v in metrics.items()}})
        self.extras["log"] = log

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._randomize_episode(env_ids)
        pos, quat, lin_vel, ang_vel = self._sample_initial_state(env_ids)
        root = torch.cat([pos + self.env_origins[env_ids], quat, lin_vel, ang_vel], -1)
        self.robot.write_root_state_to_sim(root, env_ids)
        # rotor speeds start at (randomized-model) hover
        self.model.reset(env_ids)
        k = len(env_ids)
        dev = self.device
        c = self.cfg.control
        self.actions[env_ids] = self.hover_action
        self.prev_actions[env_ids] = self.hover_action
        self.action_delay.reset(env_ids, self.hover_action.expand(k, -1))
        self.action_delay.set_delay(torch.randint(c.action_delay_steps[0], c.action_delay_steps[1] + 1, (k,),
                                                  device=dev), env_ids)
        self.fc.reset(env_ids, quat, ang_vel)
        self.action_iface.reset(env_ids)
        state13 = torch.cat([pos, quat, lin_vel, ang_vel], -1)
        self.imu.reset(env_ids, lin_vel)
        self.odom.reset(env_ids, state13)
        self.odom_meas[env_ids] = state13
        self.imu_meas[env_ids] = torch.cat([ang_vel, qm.quat_rotate_inverse(quat, torch.tensor(
            [0.0, 0.0, 9.81], device=dev).expand(k, 3))], -1)
        self.baro.reset(env_ids)
        self.wind.reset(env_ids)
        self.push.reset(env_ids)
        if self.vision_enabled:
            self.rgb_aug.reset(env_ids)
        self._reset_task(env_ids)
        self.terms.after_reset(env_ids)

    # ======================================================================== task hooks
    def _sample_initial_state(self, env_ids):
        k = len(env_ids)
        dev = self.device
        pos = torch.zeros(k, 3, device=dev)
        pos[:, 2] = 1.0
        quat = torch.zeros(k, 4, device=dev)
        quat[:, 0] = 1.0
        return pos, quat, torch.zeros(k, 3, device=dev), torch.zeros(k, 3, device=dev)

    def _task_obs(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)

    def _task_privileged_obs(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)

    def _compute_task_reward(self) -> dict[str, torch.Tensor]:
        return {}

    def _task_crash(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _task_timeout(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_task(self, env_ids):
        pass

    def _task_episode_metrics(self, env_ids) -> dict[str, torch.Tensor]:
        """Per-env task metrics of the finishing episodes (NaN = undefined for that env)."""
        return {}

    # ---------------------------------------------------------------- debug visualization
    def debug_points(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Task hook: (points (N, M, 3) env-local, marker ids (M,)): 0 = primary target, 1 = secondary."""
        return None

    def _set_debug_vis_impl(self, debug_vis: bool):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        if debug_vis:
            if not hasattr(self, "_markers"):
                cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/quadsim_targets",
                    markers={
                        "target": sim_utils.SphereCfg(
                            radius=0.08, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2))),
                        "secondary": sim_utils.SphereCfg(
                            radius=0.035, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.0))),
                    },
                )
                self._markers = VisualizationMarkers(cfg)
            self._markers.set_visibility(True)
        elif hasattr(self, "_markers"):
            self._markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "scene") or not hasattr(self, "_markers") or not self.sim.is_playing():
            return  # called during start-up / shutdown
        out = self.debug_points()
        if out is None:
            return
        pts, ids = out
        pts = pts + self.env_origins[:, None, :]
        n, m = pts.shape[:2]
        self._markers.visualize(translations=pts.reshape(-1, 3), marker_indices=ids.repeat(n))

    # ---------------------------------------------------------------- helpers for tasks
    def local_pos(self) -> torch.Tensor:
        return self.robot.data.root_pos_w - self.env_origins

    def est_pos(self) -> torch.Tensor:
        return self.odom_meas[:, :3]

    def est_quat(self) -> torch.Tensor:
        return qm.quat_normalize(self.odom_meas[:, 3:7])

    def world_to_est_body(self, vec_w: torch.Tensor) -> torch.Tensor:
        return qm.quat_rotate_inverse(self.est_quat(), vec_w)

