"""Obstacle-avoidance navigation through a randomized cylinder forest toward a goal.

Obstacles are kinematic PhysX cylinders (``RigidObjectCollection``) re-positioned every episode, so
they are physically collidable *and* rendered by the onboard camera. Exactly the same primitive list
is used for (i) analytic collision checking, (ii) analytic ray-cast range/depth sensing and
(iii) privileged critic observations.

Observation modes (``cfg.vision.mode``): ``none`` (state + range ring, i.e. LiDAR-like),
``depth`` (``raycast`` or ``render`` source), ``rgb`` / ``rgbd`` (``render`` source only).
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

from quadsim.core import math as qm
from quadsim.core.sensors import apply_depth_noise, cast, lidar_ray_dirs, normalize_depth
from quadsim.tasks.base import QuadrotorEnv, QuadrotorEnvCfg


@configclass
class NavEnvCfg(QuadrotorEnvCfg):
    episode_length_s: float = 20.0
    num_obstacles: int = 24
    obstacle_radius_range: tuple[float, float] = (0.15, 0.45)
    obstacle_height: float = 4.0
    field_x: tuple[float, float] = (2.0, 10.0)
    field_y: tuple[float, float] = (-4.0, 4.0)
    active_obstacles_range: tuple[int, int] = (4, 24)  # per-episode count ~ U[lo, current curriculum max]
    curriculum: bool = True
    curriculum_start: int = 8  # initial max number of active obstacles
    curriculum_step: int = 2
    curriculum_success_threshold: float = 0.7  # EMA success rate required to densify the forest
    min_obstacle_gap: float = 0.9  # m between cylinder surfaces (vehicle must fit)
    start_y: float = 2.0
    goal_x: tuple[float, float] = (11.0, 12.0)
    goal_y: float = 3.0
    height_range: tuple[float, float] = (1.0, 2.0)
    goal_radius: float = 0.5
    max_height: float = 3.5
    corridor_margin: float = 1.0
    vehicle_radius: float = 0.35  # collision radius used for the analytic check (X500 tip-to-tip ~0.6 m)
    # LiDAR-like horizontal range ring (always computed; in the policy obs when vision.mode == "none")
    range_rays: int = 32
    range_max: float = 8.0
    # reward weights
    w_progress: float = 1.0
    w_goal: float = 10.0
    w_crash: float = 10.0
    w_heading: float = 0.05
    w_height: float = 0.1
    w_proximity: float = 0.2
    w_ang_vel: float = 0.01
    w_action_rate: float = 0.02
    max_speed_obs: float = 5.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 30.0  # foreign obstacles stay beyond the 10 m depth range
        self.control.mode = "velocity"
        self.max_distance = 25.0


class NavEnv(QuadrotorEnv):
    cfg: NavEnvCfg

    @classmethod
    def task_obs_dim(cls, cfg) -> int:
        return 4 + (cfg.range_rays if cfg.vision.mode == "none" else 0)

    def __init__(self, cfg: NavEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, k, dev = self.num_envs, cfg.num_obstacles, self.device
        self.goal = torch.zeros(n, 3, device=dev)
        self.cylinders = torch.zeros(n, k, 4, device=dev)  # (cx, cy, r, h), r=0 -> inactive
        self.cyl_radius_fixed = self._radii().to(dev)
        self.prev_goal_dist = torch.zeros(n, device=dev)
        self.reached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.collided = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ring_dirs = lidar_ray_dirs(cfg.range_rays, 1, (0.0, 0.0), dev)
        self.range_ring = torch.full((n, cfg.range_rays), cfg.range_max, device=dev)
        self.range_ring_true = self.range_ring.clone()
        self.min_clearance = torch.full((n,), 1e3, device=dev)
        self.path_len = torch.zeros(n, device=dev)
        self.t_start = torch.zeros(n, dtype=torch.long, device=dev)
        self.active_max = cfg.curriculum_start if cfg.curriculum else cfg.active_obstacles_range[1]
        self.success_ema = 0.0

    def _radii(self):
        lo, hi = self.cfg.obstacle_radius_range
        k = self.cfg.num_obstacles
        return torch.tensor([lo + (hi - lo) * ((i * 0.618034) % 1.0) for i in range(k)])

    # ------------------------------------------------------------------ scene
    def _setup_task_scene(self):
        c = self.cfg
        objs = {}
        for i, r in enumerate(self._radii().tolist()):
            objs[f"cyl{i:02d}"] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacle{i:02d}",
                spawn=sim_utils.CylinderCfg(
                    radius=r, height=c.obstacle_height, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.35 + 0.4 * ((i * 0.37) % 1), 0.3 + 0.3 * ((i * 0.71) % 1), 0.2)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
            )
        self.obstacles = RigidObjectCollection(RigidObjectCollectionCfg(rigid_objects=objs))
        self.scene.rigid_object_collections["obstacles"] = self.obstacles
        v = c.vision
        if v.mode != "none" and v.source == "render":
            types = {"depth": ["distance_to_image_plane"], "rgb": ["rgb"],
                     "rgbd": ["rgb", "distance_to_image_plane"]}[v.mode]
            f = 1.0
            hap = 2 * f * math.tan(math.radians(v.hfov_deg) / 2)
            pitch = math.radians(v.pitch_deg)
            self.camera = TiledCamera(TiledCameraCfg(
                prim_path="/World/envs/env_.*/Robot/camera",
                offset=TiledCameraCfg.OffsetCfg(pos=v.offset_pos, rot=(math.cos(-pitch / 2), 0.0, math.sin(-pitch / 2), 0.0),
                                                convention="world"),
                data_types=types,
                spawn=sim_utils.PinholeCameraCfg(focal_length=f, horizontal_aperture=hap,
                                                 clipping_range=(0.05, 50.0)),
                width=v.width, height=v.height,
            ))
            self.scene.sensors["camera"] = self.camera

    def obstacle_primitives(self):
        return self.cylinders, None

    # ------------------------------------------------------------------ sensing
    def _update_range_ring(self):
        """Horizontal 360 deg range ring in the yaw-aligned frame of the estimated pose (LiDAR-like)."""
        n, r = self.num_envs, self.cfg.range_rays
        yaw = qm.yaw_from_quat(self.robot.data.root_quat_w)
        q_yaw = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        dirs = qm.quat_rotate(q_yaw[:, None, :].expand(n, r, 4), self.ring_dirs[None].expand(n, r, 3))
        origins = self.local_pos()[:, None, :].expand(n, r, 3)
        ring = cast(origins, dirs, self.cylinders, None, ground=False, max_range=self.cfg.range_max)
        self.range_ring_true = ring
        if self.cfg.sensors.noise_enabled:  # 2 cm range noise + 1 % dropouts (read as max range)
            ring = ring + 0.02 * torch.randn_like(ring)
            ring = torch.where(torch.rand_like(ring) < 0.01, torch.full_like(ring, self.cfg.range_max), ring)
        self.range_ring = ring.clamp(0.0, self.cfg.range_max)

    def _render_camera(self):
        v = self.cfg.vision
        out = self.camera.data.output
        imgs = []
        if v.mode in ("rgb", "rgbd"):
            rgb = out["rgb"][..., :3].permute(0, 3, 1, 2).float() / 255.0
            imgs.append(self.rgb_aug(rgb))
        if v.mode in ("depth", "rgbd"):
            depth = out["distance_to_image_plane"][..., 0]
            depth = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, 1e3))
            depth = apply_depth_noise(depth, v.depth_noise)
            imgs.append(normalize_depth(depth, v.depth_noise.max_range)[:, None])
        return torch.cat(imgs, 1)

    def debug_points(self):
        return self.goal[:, None, :], torch.zeros(1, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ obs
    def _task_obs(self):
        self._update_range_ring()
        rel_b = self.world_to_est_body(self.goal - self.est_pos())
        dist = rel_b.norm(dim=-1, keepdim=True)
        parts = [rel_b / dist.clamp_min(1e-6), torch.clamp(dist, max=10.0) / 10.0]
        if self.cfg.vision.mode == "none":
            parts.append(self.range_ring / self.cfg.range_max)
        return torch.cat(parts, -1)

    def _task_privileged_obs(self):
        q = self.robot.data.root_quat_w
        rel = qm.quat_rotate_inverse(q, self.goal - self.local_pos())
        ring = self.range_ring_true / self.cfg.range_max
        return torch.cat([rel, ring], -1)

    # ------------------------------------------------------------------ reward / dones
    def _obstacle_clearance(self):
        p = self.local_pos()
        c = self.cylinders
        d_xy = (p[:, None, :2] - c[..., :2]).norm(dim=-1) - c[..., 2]
        above = p[:, None, 2] > c[..., 3]
        d = torch.where((c[..., 2] > 0) & ~above, d_xy, torch.full_like(d_xy, 1e3))
        return d.min(-1).values

    def _get_dones(self):
        clearance = self._obstacle_clearance()
        self.min_clearance = torch.minimum(self.min_clearance, clearance)
        self.collided = clearance < self.cfg.vehicle_radius
        self.reached = (self.goal - self.local_pos()).norm(dim=-1) < self.cfg.goal_radius
        return super()._get_dones()

    def _task_crash(self):
        p = self.local_pos()
        c = self.cfg
        out = (p[:, 2] > c.max_height) | (p[:, 1].abs() > c.field_y[1] + c.corridor_margin) | (p[:, 0] < -2.0)
        return self.collided | out

    def _task_timeout(self):
        return self.reached

    def _compute_task_reward(self):
        c, dt = self.cfg, self.step_dt
        d = self.robot.data
        dist = (self.goal - self.local_pos()).norm(dim=-1)
        progress = self.prev_goal_dist - dist
        self.prev_goal_dist = dist
        self.path_len += d.root_lin_vel_w.norm(dim=-1) * dt
        # keep the camera (body x axis) aligned with the direction of travel
        vel_h = d.root_lin_vel_w[:, :2]
        fwd = qm.quat_rotate(d.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3))
        heading = (fwd[:, :2] * vel_h).sum(-1) / vel_h.norm(dim=-1).clamp_min(0.3)
        h_lo, h_hi = c.height_range
        z = self.local_pos()[:, 2]
        height_err = torch.relu(h_lo - z) + torch.relu(z - h_hi)
        clearance = self._obstacle_clearance()
        proximity = torch.exp(-torch.clamp(clearance - c.vehicle_radius, min=0.0) / 0.3)
        return {
            "progress": c.w_progress * progress,
            "goal": c.w_goal * self.reached.float(),
            "heading": c.w_heading * heading * dt,
            "height": -c.w_height * height_err * dt,
            "proximity": -c.w_proximity * proximity * dt,
            "ang_vel": -c.w_ang_vel * d.root_ang_vel_b.square().sum(-1) * dt,
            "action_rate": -c.w_action_rate * (self.actions - self.prev_actions).square().sum(-1) * dt,
            "crash": -c.w_crash * self.crashed.float(),
        }

    # ------------------------------------------------------------------ reset
    def _place_obstacles(self, env_ids, start, goal):
        """Rejection-sample non-overlapping cylinders that keep the start/goal free."""
        k, m, dev, c = len(env_ids), self.cfg.num_obstacles, self.device, self.cfg
        radii = self.cyl_radius_fixed
        cand_n = 6
        xs = c.field_x[0] + (c.field_x[1] - c.field_x[0]) * torch.rand(k, m, cand_n, device=dev)
        ys = c.field_y[0] + (c.field_y[1] - c.field_y[0]) * torch.rand(k, m, cand_n, device=dev)
        placed = torch.zeros(k, m, 4, device=dev)
        lo, hi = c.active_obstacles_range
        n_active = torch.randint(lo, max(lo, min(hi, self.active_max)) + 1, (k,), device=dev)
        for i in range(m):
            cand = torch.stack([xs[:, i], ys[:, i]], -1)  # (k, cand_n, 2)
            ok = torch.ones(k, cand_n, dtype=torch.bool, device=dev)
            for pt in (start, goal):
                ok &= (cand - pt[:, None, :2]).norm(dim=-1) > radii[i] + 1.0
            if i > 0:
                prev = placed[:, :i]
                gap = (cand[:, :, None, :] - prev[:, None, :, :2]).norm(dim=-1) - prev[:, None, :, 2] - radii[i]
                ok &= ((gap > c.min_obstacle_gap) | (prev[:, None, :, 2] == 0)).all(-1)
            any_ok = ok.any(-1)
            j = ok.float().argmax(-1)
            sel = cand[torch.arange(k, device=dev), j]
            active = any_ok & (i < n_active)
            placed[:, i, :2] = sel
            placed[:, i, 2] = torch.where(active, radii[i].expand(k), torch.zeros(k, device=dev))
            placed[:, i, 3] = c.obstacle_height
        return placed

    def _sample_initial_state(self, env_ids):
        k, dev, c = len(env_ids), self.device, self.cfg
        h_lo, h_hi = c.height_range
        start = torch.zeros(k, 3, device=dev)
        start[:, 1] = (torch.rand(k, device=dev) * 2 - 1) * c.start_y
        start[:, 2] = h_lo + (h_hi - h_lo) * torch.rand(k, device=dev)
        goal = torch.zeros(k, 3, device=dev)
        goal[:, 0] = c.goal_x[0] + (c.goal_x[1] - c.goal_x[0]) * torch.rand(k, device=dev)
        goal[:, 1] = (torch.rand(k, device=dev) * 2 - 1) * c.goal_y
        goal[:, 2] = h_lo + (h_hi - h_lo) * torch.rand(k, device=dev)
        self.goal[env_ids] = goal
        cyl = self._place_obstacles(env_ids, start, goal)
        self.cylinders[env_ids] = cyl
        # write obstacle poses (inactive ones parked below ground)
        m = c.num_obstacles
        pose = torch.zeros(k, m, 7, device=dev)
        pose[..., 0:2] = cyl[..., :2]
        pose[..., 2] = torch.where(cyl[..., 2] > 0, torch.full_like(cyl[..., 2], c.obstacle_height / 2),
                                   torch.full_like(cyl[..., 2], -10.0))
        pose[..., :3] += self.scene.env_origins[env_ids, None, :]
        pose[..., 3] = 1.0
        self.obstacles.write_object_pose_to_sim(pose, env_ids=env_ids)
        self.prev_goal_dist[env_ids] = (goal - start).norm(dim=-1)
        self.min_clearance[env_ids] = 1e3
        self.path_len[env_ids] = 0.0
        yaw = torch.atan2(goal[:, 1] - start[:, 1], goal[:, 0] - start[:, 0]) + (torch.rand(k, device=dev) - 0.5) * 0.5
        z = torch.zeros_like(yaw)
        quat = torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], -1)
        return start, quat, torch.zeros(k, 3, device=dev), torch.zeros(k, 3, device=dev)

    def _reset_task(self, env_ids):
        self.t_start[env_ids] = self.episode_length_buf[env_ids]
        self.reached[env_ids] = False
        self.collided[env_ids] = False

    def _update_curriculum(self, succ: torch.Tensor):
        c = self.cfg
        if not c.curriculum or len(succ) == 0:
            return
        self.success_ema = 0.98 * self.success_ema + 0.02 * succ.mean().item()
        if self.success_ema > c.curriculum_success_threshold and self.active_max < c.active_obstacles_range[1]:
            self.active_max = min(self.active_max + c.curriculum_step, c.active_obstacles_range[1])
            self.success_ema = 0.5 * c.curriculum_success_threshold  # require renewed success at the new level

    def _task_episode_metrics(self, env_ids):
        succ = self.reached[env_ids].float()
        played = self.episode_length_buf[env_ids] > 0
        self._update_curriculum(succ[played])
        t = (self.episode_length_buf[env_ids] - self.t_start[env_ids]).float() * self.step_dt
        return {
            "success": succ,
            "collision": self.collided[env_ids].float(),
            "min_clearance": self.min_clearance[env_ids].clamp(max=10.0),
            "time_to_goal": torch.where(succ > 0, t, torch.full_like(t, float("nan"))),
            "path_length": torch.where(succ > 0, self.path_len[env_ids], torch.full_like(t, float("nan"))),
            "active_obstacles": (self.cylinders[env_ids, :, 2] > 0).float().sum(-1),
            "curriculum_max_obstacles": torch.full_like(succ, float(self.active_max)),
        }
