"""Language-conditioned landmark navigation ("fly to the red pillar"): a lightweight VLA / imitation-learning task.

``num_landmarks`` coloured pillars (red, green, blue, yellow, ...) stand in front of the vehicle (inside the onboard
camera's field of view at the start). A natural-language instruction names one of them; the episode succeeds when the
vehicle stops in front of that pillar (within ``reach_distance`` of its surface, slower than ``stop_speed``).
Stopping in front of another pillar ends the episode as a *wrong choice* (grounding error); touching any pillar is
a crash.

Observation groups:
  ``policy``  deployable proprioception + instruction-token ids (``max_tokens``, see :mod:`quadsim.il.language`) +
              (``symbolic=True`` only) landmark positions relative to the vehicle in the estimated body frame,
              listed in a fixed *colour* order (a map with semantic labels);
  ``image``   onboard RGB camera (``vision.mode = "rgb"``, rendered; needs ``--enable_cameras``);
  ``critic``  privileged: true goal-relative state and target index.

The instruction text of each environment is available as ``env.instructions[i]`` (the recorder stores it with every
episode) and its token ids are part of the observation, so RL, BC and VLA-style policies all see the same task.
Instruction templates are split into training and held-out sets (``template_split``) to test generalization to new
phrasings.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

from quadsim.core import math as qm
from quadsim.tasks.base import QuadrotorEnv, QuadrotorEnvCfg

COLORS = {"red": (0.85, 0.08, 0.08), "green": (0.1, 0.7, 0.15), "blue": (0.1, 0.25, 0.9), "yellow": (0.9, 0.8, 0.1),
          "white": (0.92, 0.92, 0.92), "purple": (0.55, 0.15, 0.7)}

TRAIN_TEMPLATES = (
    "fly to the {c} pillar", "go to the {c} one", "navigate to the {c} pillar", "head for the {c} post",
    "approach the {c} marker", "move toward the {c} pillar", "reach the {c} column", "the {c} pillar please",
)
HELDOUT_TEMPLATES = (
    "could you take me over to the {c} post", "your target is the {c} column", "stop in front of the {c} marker",
)


@configclass
class LangNavEnvCfg(QuadrotorEnvCfg):
    episode_length_s: float = 10.0
    num_landmarks: int = 4
    colors: tuple = ("red", "green", "blue", "yellow")
    landmark_radius: float = 0.25
    landmark_height: float = 3.0
    distance_range: tuple[float, float] = (3.0, 6.5)
    bearing_range: float = 0.7
    """max |bearing| of a landmark seen from the start [rad] (camera HFOV 87 deg = +-0.76 rad)"""
    min_bearing_separation: float = 0.35
    reach_distance: float = 1.0
    """success zone: horizontal distance to the pillar surface below this ..."""
    wrong_distance: float = 0.6
    """grounding error: stopped this close to a pillar that was not named"""
    approach_distance: float = 0.75
    """distance of the expert's stopping point from the target surface"""
    stop_speed: float = 0.5
    """... while slower than this [m/s] ..."""
    dwell_steps: int = 10
    """... for this many consecutive policy steps (0.2 s); the same dwell near another pillar is a wrong choice"""
    start_height: float = 1.5
    vehicle_radius: float = 0.35
    symbolic: bool = True
    template_split: str = "train"
    """train | heldout | all"""
    max_tokens: int = 12
    # reward
    w_progress: float = 1.0
    w_success: float = 5.0
    w_wrong: float = 5.0
    w_crash: float = 10.0
    w_speed_near: float = 0.5
    w_action_rate: float = 0.02

    def __post_init__(self):
        super().__post_init__()
        self.control.mode = "velocity"
        self.control.max_velocity = (2.0, 2.0, 1.0)
        self.scene.env_spacing = 20.0
        self.vision.width, self.vision.height = 64, 48
        self.vision.source = "render"


class LangNavEnv(QuadrotorEnv):
    cfg: LangNavEnvCfg

    @classmethod
    def task_obs_dim(cls, cfg) -> int:
        return cfg.max_tokens + (3 * cfg.num_landmarks if cfg.symbolic else 0)

    def __init__(self, cfg: LangNavEnvCfg, render_mode=None, **kwargs):
        from quadsim.il.language import Tokenizer

        if cfg.num_landmarks > len(COLORS):
            raise ValueError(f"num_landmarks={cfg.num_landmarks} but only {len(COLORS)} colours are defined")
        if 2 * cfg.bearing_range < (cfg.num_landmarks - 1) * cfg.min_bearing_separation:
            raise ValueError(f"{cfg.num_landmarks} landmarks need 2 * bearing_range >= (num_landmarks - 1) * "
                             f"min_bearing_separation (= {(cfg.num_landmarks - 1) * cfg.min_bearing_separation:.2f} rad)")
        super().__init__(cfg, render_mode, **kwargs)
        n, k, dev = self.num_envs, cfg.num_landmarks, self.device
        self.landmarks = torch.zeros(n, k, 3, device=dev)  # (x, y, z_top) env-local
        self.target = torch.zeros(n, dtype=torch.long, device=dev)
        self.template = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev_dist = torch.zeros(n, device=dev)
        self.success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.wrong = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hit = torch.zeros(n, dtype=torch.bool, device=dev)
        self.t_success = torch.full((n,), float("nan"), device=dev)
        self.start_xy = torch.zeros(n, 2, device=dev)
        self.layout_min_sep = torch.zeros(n, device=dev)
        self.hit_target = torch.zeros(n, dtype=torch.bool, device=dev)
        self.dwell_target = torch.zeros(n, dtype=torch.long, device=dev)
        self.dwell_other = torch.zeros(n, dtype=torch.long, device=dev)
        self.templates = {"train": TRAIN_TEMPLATES, "heldout": HELDOUT_TEMPLATES,
                          "all": TRAIN_TEMPLATES + HELDOUT_TEMPLATES}[cfg.template_split]
        self.tokenizer = Tokenizer.default(colors=list(COLORS), max_tokens=cfg.max_tokens)
        self.instructions = [""] * n
        self.tokens = torch.zeros(n, cfg.max_tokens, device=dev)

    # ------------------------------------------------------------------ scene
    def _setup_task_scene(self):
        c = self.cfg
        objs = {}
        for i, name in enumerate(c.colors[: c.num_landmarks]):
            objs[f"lm_{name}"] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Landmark_{name}",
                spawn=sim_utils.CylinderCfg(
                    radius=c.landmark_radius, height=c.landmark_height, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[name], roughness=0.6)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)))
        self.pillars = RigidObjectCollection(RigidObjectCollectionCfg(rigid_objects=objs))
        self.scene.rigid_object_collections["landmarks"] = self.pillars
        v = c.vision
        if v.mode != "none" and v.source == "render":
            types = {"depth": ["distance_to_image_plane"], "rgb": ["rgb"],
                     "rgbd": ["rgb", "distance_to_image_plane"]}[v.mode]
            hap = 2 * math.tan(math.radians(v.hfov_deg) / 2)
            pitch = math.radians(v.pitch_deg)
            self.camera = TiledCamera(TiledCameraCfg(
                prim_path="/World/envs/env_.*/Robot/camera",
                offset=TiledCameraCfg.OffsetCfg(pos=v.offset_pos, rot=(math.cos(-pitch / 2), 0.0,
                                                                       math.sin(-pitch / 2), 0.0), convention="world"),
                data_types=types,
                spawn=sim_utils.PinholeCameraCfg(focal_length=1.0, horizontal_aperture=hap, clipping_range=(0.05, 50.0)),
                width=v.width, height=v.height))
            self.scene.sensors["camera"] = self.camera

    def _render_camera(self):
        out = self.camera.data.output
        rgb = out["rgb"][..., :3].permute(0, 3, 1, 2).float() / 255.0
        return self.rgb_aug(rgb)

    def obstacle_primitives(self):
        k = self.cfg.num_landmarks
        cyl = torch.cat([self.landmarks[..., :2], torch.full_like(self.landmarks[..., :1], self.cfg.landmark_radius),
                         torch.full_like(self.landmarks[..., :1], self.cfg.landmark_height)], -1)
        return cyl.view(self.num_envs, k, 4), None

    # ------------------------------------------------------------------ geometry
    def _surface_dist(self, p):
        """(N, K) horizontal distance from ``p`` to every pillar surface."""
        return (p[:, None, :2] - self.landmarks[..., :2]).norm(dim=-1) - self.cfg.landmark_radius

    def goal_point(self):
        """Point in front of the target pillar on the line from the start (used by the scripted expert)."""
        lm = self.landmarks[torch.arange(self.num_envs, device=self.device), self.target]
        d = lm[:, :2] - self.start_xy
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        g = lm.clone()
        g[:, :2] = lm[:, :2] - d * (self.cfg.landmark_radius + self.cfg.approach_distance)
        g[:, 2] = self.cfg.start_height
        return g

    # ------------------------------------------------------------------ observations
    def obs_layout(self):
        out = [["state", 12]] + ([["last_action", self.action_dim]] if self.cfg.include_last_action else [])
        out.append(["tokens", self.cfg.max_tokens])
        if self.cfg.symbolic:
            out.append(["landmarks", 3 * self.cfg.num_landmarks])
        return out + [[o.name, o.dim] for o in self.terms.policy_obs]

    def _task_obs(self):
        parts = [self.tokens]
        if self.cfg.symbolic:
            rel = self.landmarks.clone()
            rel[..., 2] = self.cfg.start_height
            rel = rel - self.est_pos()[:, None]
            q = self.est_quat()[:, None].expand(-1, self.cfg.num_landmarks, -1)
            parts.append(qm.quat_rotate_inverse(q, rel).flatten(1))
        return torch.cat(parts, -1)

    def _task_privileged_obs(self):
        g = self.goal_point() - self.local_pos()
        onehot = torch.nn.functional.one_hot(self.target, self.cfg.num_landmarks).float()
        return torch.cat([qm.quat_rotate_inverse(self.robot.data.root_quat_w, g), onehot], -1)

    # ------------------------------------------------------------------ dynamics of the task
    def _task_crash(self):
        p = self.local_pos()
        d = self._surface_dist(p)
        self.hit = (d < self.cfg.vehicle_radius).any(-1)
        self.hit_target = d[torch.arange(self.num_envs, device=self.device), self.target] < self.cfg.vehicle_radius
        speed = self.robot.data.root_lin_vel_w.norm(dim=-1)
        ar = torch.arange(self.num_envs, device=self.device)
        slow = speed < self.cfg.stop_speed
        near_target = d[ar, self.target] < self.cfg.reach_distance
        near_other = d < self.cfg.wrong_distance
        near_other[ar, self.target] = False
        self.dwell_target = torch.where(near_target & slow, self.dwell_target + 1, torch.zeros_like(self.dwell_target))
        self.dwell_other = torch.where(near_other.any(-1) & slow, self.dwell_other + 1,
                                       torch.zeros_like(self.dwell_other))
        new_success = (self.dwell_target >= self.cfg.dwell_steps) & ~self.success & ~self.wrong & ~self.hit
        self.wrong |= (self.dwell_other >= self.cfg.dwell_steps) & ~self.success & ~new_success
        self.t_success = torch.where(new_success, self.episode_length_buf.float() * self.step_dt, self.t_success)
        self.success |= new_success
        return self.hit | (p[:, 2] > 4.0)

    def _task_timeout(self):
        return self.success | self.wrong

    def _compute_task_reward(self):
        c, dt = self.cfg, self.step_dt
        p = self.local_pos()
        dist = (self.goal_point() - p).norm(dim=-1)
        progress = self.prev_dist - dist
        self.prev_dist = dist
        speed = self.robot.data.root_lin_vel_w.norm(dim=-1)
        near = (dist < 1.5).float()
        return {
            "progress": c.w_progress * progress,
            "success": c.w_success * self.success.float(),
            "wrong": -c.w_wrong * self.wrong.float(),
            "speed_near_goal": -c.w_speed_near * near * speed * dt,
            "action_rate": -c.w_action_rate * (self.actions - self.prev_actions).square().sum(-1) * dt,
            "crash": -c.w_crash * self.crashed.float(),
        }

    # ------------------------------------------------------------------ reset
    def _sample_initial_state(self, env_ids):
        c, dev = self.cfg, self.device
        k, m = len(env_ids), c.num_landmarks
        yaw0 = (torch.rand(k, device=dev) * 2 - 1) * math.pi
        # bearings with guaranteed spacing (uniform "spacing trick"), randomly assigned to the colours, so every
        # straight start->pillar path clears the other pillars by >= d_min sin(min_bearing_separation)
        span = 2 * c.bearing_range - (m - 1) * c.min_bearing_separation
        u = torch.rand(k, m, device=dev).sort(-1).values * span
        bear = -c.bearing_range + u + c.min_bearing_separation * torch.arange(m, device=dev)
        bear = torch.gather(bear, 1, torch.rand(k, m, device=dev).argsort(-1))
        dist = c.distance_range[0] + (c.distance_range[1] - c.distance_range[0]) * torch.rand(k, m, device=dev)
        placed = torch.stack([dist * torch.cos(bear), dist * torch.sin(bear)], -1)
        sep = (placed[:, :, None] - placed[:, None]).norm(dim=-1) + torch.eye(m, device=dev) * 1e3
        self.layout_min_sep[env_ids] = sep.min(-1).values.min(-1).values
        cy, sy = torch.cos(yaw0), torch.sin(yaw0)
        w = torch.stack([cy[:, None] * placed[..., 0] - sy[:, None] * placed[..., 1],
                         sy[:, None] * placed[..., 0] + cy[:, None] * placed[..., 1]], -1)
        lm = torch.zeros(k, m, 3, device=dev)
        lm[..., :2] = w
        lm[..., 2] = c.landmark_height
        self.landmarks[env_ids] = lm
        pose = torch.zeros(k, m, 7, device=dev)
        pose[..., :2] = w
        pose[..., 2] = c.landmark_height / 2
        pose[..., :3] += self.scene.env_origins[env_ids, None]
        pose[..., 3] = 1.0
        # the collection's object order is its own (by name): map colour slot -> collection index explicitly
        if not hasattr(self, "_slot_of_object"):
            names = list(self.pillars.object_names)
            self._slot_of_object = torch.tensor([list(c.colors[:m]).index(nm.split("_", 1)[1]) for nm in names],
                                                device=dev)
        self.pillars.write_object_pose_to_sim(pose[:, self._slot_of_object], env_ids=env_ids)
        self.target[env_ids] = torch.randint(0, m, (k,), device=dev)
        self.template[env_ids] = torch.randint(0, len(self.templates), (k,), device=dev)
        for e, t, tpl in zip(env_ids.tolist(), self.target[env_ids].tolist(), self.template[env_ids].tolist()):
            self.instructions[e] = self.templates[tpl].format(c=c.colors[t])
        self.tokens[env_ids] = self.tokenizer.encode_batch([self.instructions[e] for e in env_ids.tolist()],
                                                            device=dev).float()
        self.success[env_ids] = False
        self.wrong[env_ids] = False
        self.dwell_target[env_ids] = 0
        self.dwell_other[env_ids] = 0
        self.hit[env_ids] = False
        self.t_success[env_ids] = float("nan")
        pos = torch.zeros(k, 3, device=dev)
        pos[:, 2] = c.start_height
        self.start_xy[env_ids] = 0.0
        quat = qm.quat_from_euler_xyz(torch.zeros(k, device=dev), torch.zeros(k, device=dev), yaw0)
        self.prev_dist[env_ids] = (self.goal_point()[env_ids] - pos).norm(dim=-1)
        return pos, quat, torch.zeros(k, 3, device=dev), torch.zeros(k, 3, device=dev)

    def _task_episode_metrics(self, env_ids):
        return {"success": self.success[env_ids].float(), "wrong_landmark": self.wrong[env_ids].float(),
                "pillar_collision": self.hit[env_ids].float(), "time_to_success": self.t_success[env_ids],
                "layout_min_separation": self.layout_min_sep[env_ids],
                "target_collision": self.hit_target[env_ids].float()}

    # ------------------------------------------------------------------ expert / baselines
    def baseline_targets(self):
        """Privileged scripted expert: fly to the approach point of the *instructed* pillar and stop there."""
        g = self.goal_point()
        z = torch.zeros_like(g)
        lm = self.landmarks[torch.arange(self.num_envs, device=self.device), self.target]
        d = lm - self.est_pos()
        return g, z, z, torch.atan2(d[:, 1], d[:, 0])

    def debug_points(self):
        return self.goal_point()[:, None], torch.zeros(1, dtype=torch.long, device=self.device)


class LangNavExpert:
    """Privileged scripted expert (velocity interface): straight-line approach to the instructed pillar's stopping
    point with a norm-limited, distance-proportional speed profile and repulsion from the other pillars."""

    def __init__(self, env, speed: float = 1.8, k_pos: float = 1.0, influence: float = 1.3, k_rep: float = 1.5):
        self.env, self.speed, self.k_pos, self.influence, self.k_rep = env, speed, k_pos, influence, k_rep

    def reset(self, env_ids=None):
        pass

    def __call__(self):
        env = self.env
        p, q = env.est_pos(), env.est_quat()
        g = env.goal_point()
        e = g - p
        dist = e.norm(dim=-1, keepdim=True)
        v = e / dist.clamp_min(1e-6) * torch.clamp(self.k_pos * dist, max=self.speed)
        # repulsion (horizontal) from pillars other than the target, active within ``influence`` of their surface
        rel = p[:, None, :2] - env.landmarks[..., :2]
        d = rel.norm(dim=-1, keepdim=True)
        s = (d - env.cfg.landmark_radius).clamp_min(0.05)
        w = (1.0 / s - 1.0 / self.influence).clamp(min=0.0)
        mask = torch.ones_like(w)
        mask[torch.arange(env.num_envs, device=env.device), env.target] = 0.0
        mask = mask * dist[:, None].clamp(max=1.0)  # fade out at the stopping point (no attraction/repulsion trap)
        v[:, :2] = v[:, :2] + self.k_rep * (w * mask * rel / d.clamp_min(1e-6)).sum(1)
        n = v[:, :2].norm(dim=-1, keepdim=True)
        v[:, :2] = v[:, :2] * torch.clamp(self.speed / n.clamp_min(1e-6), max=1.0)
        lm = env.landmarks[torch.arange(env.num_envs, device=env.device), env.target]
        yaw = qm.yaw_from_quat(q)
        yaw_d = torch.atan2(lm[:, 1] - p[:, 1], lm[:, 0] - p[:, 0])
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        v_h = torch.stack([cy * v[:, 0] + sy * v[:, 1], -sy * v[:, 0] + cy * v[:, 1], v[:, 2]], -1)
        yaw_rate = 1.5 * qm.wrap_to_pi(yaw_d - yaw)
        return torch.cat([v_h / env.max_vel, (yaw_rate / env.cfg.control.max_yaw_rate)[:, None]], -1).clamp(-1, 1)
