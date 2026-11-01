"""Drone racing through a sequence of square gates (multi-lap), state-based.

Track: ``num_gates`` gates on a closed circuit. ``track = "random"`` samples a new jittered elliptic circuit every
episode (random size, rotation, gate heights and yaw); ``ellipse`` and ``figure8`` are fixed reference circuits used
to test generalization to unseen layouts (the figure-8 path crosses itself, so the policy must ignore gates that are
not its target).

Gates are analytic (not PhysX bodies): a gate is passed when the vehicle centre crosses the gate plane in the flight
direction inside the opening shrunk by the vehicle radius, and it is a crash when the vehicle touches the frame
(near the plane inside the frame band, or crossing the plane through the frame). Passing the wrong gate is allowed.

Policy observation (deployable: state estimate + known track map, as in championship-level racing with known gate
poses): for the target gate and the one after it, the gate centre relative to the vehicle and the gate normal, both in
the estimated body frame. Reward: progress toward the target gate centre (Song et al. 2021), gate bonus, crash
penalty, body-rate and action-rate penalties.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from quadsim.core import math as qm
from quadsim.tasks.base import QuadrotorEnv, QuadrotorEnvCfg


@configclass
class RaceEnvCfg(QuadrotorEnvCfg):
    episode_length_s: float = 20.0
    num_gates: int = 7
    laps: int = 2
    track: str = "random"
    """random | ellipse | figure8"""
    random_direction: bool = False
    """random circuits: also clockwise (mirrored) circuits, i.e. both turning directions (v1)"""
    radius_x: tuple[float, float] = (4.0, 7.0)
    radius_y: tuple[float, float] = (3.0, 5.0)
    gate_height_range: tuple[float, float] = (1.3, 2.5)
    gate_pos_jitter: float = 0.5
    gate_yaw_jitter: float = 0.3
    gate_inner: float = 1.5
    """side of the square opening [m]"""
    gate_outer: float = 2.1
    """outer side of the frame [m]"""
    vehicle_radius: float = 0.15
    start_distance: float = 2.0
    obs_gates: int = 2
    max_obs_distance: float = 10.0
    init_max_tilt: float = 0.2
    # reward
    w_progress: float = 1.0
    w_gate: float = 1.0
    w_crash: float = 5.0
    w_body_rate: float = 0.002
    w_action_rate: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        self.dynamics.preset = "racer5"
        self.control.max_body_rate = (10.0, 10.0, 5.0)
        self.scene.env_spacing = 25.0
        self.max_distance = 20.0
        self.max_lin_vel = 30.0
        self.max_ang_vel = 40.0


class RaceEnv(QuadrotorEnv):
    cfg: RaceEnvCfg

    @classmethod
    def task_obs_dim(cls, cfg) -> int:
        return 6 * cfg.obs_gates

    def __init__(self, cfg: RaceEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, g, dev = self.num_envs, cfg.num_gates, self.device
        self.gate_pos = torch.zeros(n, g, 3, device=dev)
        self.gate_yaw = torch.zeros(n, g, device=dev)
        self.target = torch.zeros(n, dtype=torch.long, device=dev)
        self.gates_passed = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev_pos = torch.zeros(n, 3, device=dev)
        self.prev_dist = torch.zeros(n, device=dev)
        self.finished = torch.zeros(n, dtype=torch.bool, device=dev)
        self.gate_crash = torch.zeros(n, dtype=torch.bool, device=dev)
        self.lap_time = torch.full((n,), float("nan"), device=dev)
        self.passed_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self.path_len = torch.zeros(n, device=dev)
        self._ar = torch.arange(n, device=dev)

    # ------------------------------------------------------------------ geometry helpers
    def gate_frames(self, idx: torch.Tensor | None = None):
        """(centre, normal, lateral) of the gates ``idx`` (N,) or all gates (N, G, 3)."""
        if idx is None:
            c, yaw = self.gate_pos, self.gate_yaw
        else:
            c, yaw = self.gate_pos[self._ar, idx], self.gate_yaw[self._ar, idx]
        z = torch.zeros_like(yaw)
        n = torch.stack([torch.cos(yaw), torch.sin(yaw), z], -1)
        lat = torch.stack([-torch.sin(yaw), torch.cos(yaw), z], -1)
        return c, n, lat

    def _target_offset(self, k: int):
        return (self.target + k) % self.cfg.num_gates

    # ------------------------------------------------------------------ observations
    def _task_obs(self):
        p = self.est_pos()
        parts = []
        for k in range(self.cfg.obs_gates):
            c, n, _ = self.gate_frames(self._target_offset(k))
            rel = self.world_to_est_body(c - p)
            d = rel.norm(dim=-1, keepdim=True)
            parts += [rel * torch.clamp(self.cfg.max_obs_distance / d.clamp_min(1e-6), max=1.0),
                      self.world_to_est_body(n)]
        return torch.cat(parts, -1)

    def _task_privileged_obs(self):
        c, n, _ = self.gate_frames(self.target)
        q = self.robot.data.root_quat_w
        return torch.cat([qm.quat_rotate_inverse(q, c - self.local_pos()), qm.quat_rotate_inverse(q, n)], -1)

    # ------------------------------------------------------------------ gate logic
    def _gate_events(self):
        """Per-step gate crossing / collision (true state). Returns (passed_target (N,), crash (N,))."""
        cfg = self.cfg
        p0, p1 = self.prev_pos, self.local_pos()
        c, n, lat = self.gate_frames()  # (N, G, 3)
        x0 = ((p0[:, None] - c) * n).sum(-1)
        x1 = ((p1[:, None] - c) * n).sum(-1)
        crossed = (x0 < 0) != (x1 < 0)
        alpha = (x0 / (x0 - x1).where(crossed, torch.ones_like(x0))).clamp(0, 1)
        q = p0[:, None] + alpha[..., None] * (p1 - p0)[:, None]
        yz = lambda pt: torch.maximum(((pt - c) * lat).sum(-1).abs(), (pt - c)[..., 2].abs())  # noqa: E731
        m_cross, m_now = yz(q), yz(p1[:, None].expand_as(c))
        r, half_in, half_out = cfg.vehicle_radius, cfg.gate_inner / 2, cfg.gate_outer / 2
        in_band = lambda m: (m > half_in - r) & (m < half_out + r)  # noqa: E731
        crash = (crossed & in_band(m_cross)) | ((x1.abs() < r) & in_band(m_now))
        passed = crossed & (x0 < 0) & (m_cross <= half_in - r)
        passed_target = passed[self._ar, self.target]
        return passed_target, crash.any(-1)

    def _update_gates(self):
        """Gate progress of this step (called first in ``_get_dones``, so crash / finish terminate immediately)."""
        cfg = self.cfg
        passed, crash = self._gate_events()
        passed &= ~self.finished
        self.passed_now = passed
        self.gates_passed += passed.long()
        self.target = torch.where(passed, self._target_offset(1), self.target)
        first_lap = passed & (self.gates_passed == cfg.num_gates)
        self.lap_time = torch.where(first_lap, self.episode_length_buf.float() * self.step_dt, self.lap_time)
        self.finished |= self.gates_passed >= cfg.num_gates * cfg.laps
        p = self.local_pos()
        self.path_len += (p - self.prev_pos).norm(dim=-1)
        self.prev_pos = p.clone()
        return crash

    def _compute_task_reward(self):
        cfg, dt = self.cfg, self.step_dt
        passed = self.passed_now
        c, _, _ = self.gate_frames(self.target)
        dist = (c - self.local_pos()).norm(dim=-1)
        progress = torch.where(passed, torch.zeros_like(dist), self.prev_dist - dist)
        self.prev_dist = dist
        return {
            "progress": cfg.w_progress * progress * (~self.finished | passed).float(),
            "gate": cfg.w_gate * passed.float(),
            "body_rate": -cfg.w_body_rate * self.robot.data.root_ang_vel_b.square().sum(-1) * dt,
            "action_rate": -cfg.w_action_rate * (self.actions - self.prev_actions).square().sum(-1) * dt,
            "crash": -cfg.w_crash * self.crashed.float(),
        }

    def _task_crash(self):
        self.gate_crash = self._update_gates()
        return self.gate_crash

    def _task_timeout(self):
        return self.finished

    # ------------------------------------------------------------------ tracks / reset
    def _track(self, k):
        cfg, dev, g = self.cfg, self.device, self.cfg.num_gates
        i = torch.arange(g, device=dev, dtype=torch.float32)
        rnd = lambda *s: torch.rand(*s, device=dev)  # noqa: E731
        if cfg.track == "figure8":
            a, b = 7.0, 8.0
            th = 2 * math.pi * (i + 0.5) / g
            x, y = a * torch.sin(th), b * torch.sin(th) * torch.cos(th)
            dx, dy = a * torch.cos(th), b * torch.cos(2 * th)
            pos = torch.stack([x, y, torch.zeros_like(x)], -1).expand(k, g, 3).clone()
            yaw = torch.atan2(dy, dx).expand(k, g).clone()
            pos[..., 2] = 1.6 + 0.6 * torch.sin(2 * th)
            return pos, yaw
        if cfg.track == "ellipse":
            a, b = torch.full((k, 1), 6.0, device=dev), torch.full((k, 1), 4.0, device=dev)
            th = (2 * math.pi * i / g).expand(k, g)
            rot = torch.zeros(k, 1, device=dev)
            jit_p, jit_y = torch.zeros(k, g, 2, device=dev), torch.zeros(k, g, device=dev)
            z = 1.6 + 0.6 * torch.sin(2 * th)
        else:
            a = cfg.radius_x[0] + (cfg.radius_x[1] - cfg.radius_x[0]) * rnd(k, 1)
            b = cfg.radius_y[0] + (cfg.radius_y[1] - cfg.radius_y[0]) * rnd(k, 1)
            th = 2 * math.pi * (i + (rnd(k, g) - 0.5) * 0.4) / g
            rot = rnd(k, 1) * 2 * math.pi
            jit_p = (rnd(k, g, 2) * 2 - 1) * cfg.gate_pos_jitter
            jit_y = (rnd(k, g) * 2 - 1) * cfg.gate_yaw_jitter
            lo, hi = cfg.gate_height_range
            z = lo + (hi - lo) * rnd(k, g)
        x, y = a * torch.cos(th), b * torch.sin(th)
        dx, dy = -a * torch.sin(th), b * torch.cos(th)
        if cfg.track == "random" and cfg.random_direction:  # mirror half of the circuits: clockwise laps
            sgn = torch.where(rnd(k, 1) < 0.5, -1.0, 1.0)
            y, dy = y * sgn, dy * sgn
        cr, sr = torch.cos(rot), torch.sin(rot)
        pos = torch.stack([cr * x - sr * y, sr * x + cr * y, z], -1)
        pos[..., :2] += jit_p
        yaw = torch.atan2(dy, dx) + rot + jit_y
        return pos, yaw

    def _sample_initial_state(self, env_ids):
        k, dev, cfg = len(env_ids), self.device, self.cfg
        pos_g, yaw_g = self._track(k)
        self.gate_pos[env_ids], self.gate_yaw[env_ids] = pos_g, yaw_g
        n0 = torch.stack([torch.cos(yaw_g[:, 0]), torch.sin(yaw_g[:, 0])], -1)
        pos = pos_g[:, 0].clone()
        pos[:, :2] -= n0 * cfg.start_distance
        quat = qm.quat_mul(qm.quat_from_euler_xyz(torch.zeros(k, device=dev), torch.zeros(k, device=dev),
                                                  yaw_g[:, 0]), qm.random_quat_tilt(k, cfg.init_max_tilt, dev))
        self.target[env_ids] = 0
        self.gates_passed[env_ids] = 0
        self.finished[env_ids] = False
        self.gate_crash[env_ids] = False
        self.passed_now[env_ids] = False
        self.lap_time[env_ids] = float("nan")
        self.path_len[env_ids] = 0.0
        self.prev_pos[env_ids] = pos
        self.prev_dist[env_ids] = (pos_g[:, 0] - pos).norm(dim=-1)
        return pos, quat, torch.zeros(k, 3, device=dev), torch.zeros(k, 3, device=dev)

    def _task_episode_metrics(self, env_ids):
        t = self.episode_length_buf[env_ids].float() * self.step_dt
        return {"success": self.finished[env_ids].float(),
                "gates_passed": self.gates_passed[env_ids].float(),
                "laps": self.gates_passed[env_ids].float() / self.cfg.num_gates,
                "lap_time": self.lap_time[env_ids],
                "mean_speed": self.path_len[env_ids] / t.clamp_min(1e-3)}

    # ------------------------------------------------------------------ baselines / visualization
    def baseline_targets(self):
        """Classical baseline (gate-aligned waypoints): first an approach point ``approach`` m in front of the target
        gate on its axis, then, once on the axis, through the centre along the normal at ``baseline_speed``."""
        c, n, lat = self.gate_frames(self.target)
        rel = self.est_pos() - c
        x = (rel * n).sum(-1)
        off = torch.maximum((rel * lat).sum(-1).abs(), rel[:, 2].abs())
        approach = getattr(self, "baseline_approach", 1.5)
        aligned = (x > -approach - 0.3) & (off < 0.25)
        pos = torch.where(aligned[:, None], c + 1.0 * n, c - approach * n)
        vel = torch.where(aligned[:, None], n * getattr(self, "baseline_speed", 3.0), torch.zeros_like(n))
        return pos, vel, torch.zeros_like(c), self.gate_yaw[self._ar, self.target]

    def debug_points(self):
        c, _, _ = self.gate_frames(self.target)
        return torch.cat([c[:, None], self.gate_pos], 1), torch.tensor(
            [0] + [1] * self.cfg.num_gates, device=self.device)

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        if debug_vis and not hasattr(self, "_gate_markers"):
            w, o = (self.cfg.gate_outer - self.cfg.gate_inner) / 2, self.cfg.gate_outer
            mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.1))
            self._gate_markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/race_gates",
                markers={"h": sim_utils.CuboidCfg(size=(0.05, o, w), visual_material=mat),
                         "v": sim_utils.CuboidCfg(size=(0.05, w, o), visual_material=mat)}))
        if hasattr(self, "_gate_markers"):
            self._gate_markers.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if not hasattr(self, "_gate_markers") or not self.sim.is_playing():
            return
        c, n, lat = self.gate_frames()
        off = (self.cfg.gate_inner + self.cfg.gate_outer) / 4
        up = torch.zeros_like(c)
        up[..., 2] = 1.0
        bars = torch.stack([c + off * up, c - off * up, c + off * lat, c - off * lat], 2)  # (N, G, 4, 3)
        bars = bars + self.env_origins[:, None, None]
        yaw = self.gate_yaw[..., None].expand(-1, -1, 4)
        q = qm.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        ids = torch.tensor([0, 0, 1, 1], device=self.device).repeat(self.num_envs * self.cfg.num_gates)
        self._gate_markers.visualize(translations=bars.reshape(-1, 3), orientations=q.reshape(-1, 4),
                                     marker_indices=ids)
