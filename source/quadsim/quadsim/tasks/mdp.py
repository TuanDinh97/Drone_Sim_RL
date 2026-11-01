"""Reusable MDP term functions for ``cfg.terms`` (:mod:`quadsim.tasks.base.terms`).

All functions take the environment first (Isaac Lab convention ``func(env, **params)``; events and curricula
``func(env, env_ids, **params)``) and work on any :class:`~quadsim.tasks.base.QuadrotorEnv`.

Deployability: functions marked *deployable* only read onboard information (state estimate, IMU, barometer) or model
a physical onboard sensor (they read the true state like any sensor measures the true world; add noise, rate and
latency through :class:`~quadsim.tasks.base.terms.ObsTermCfg`). Functions marked *privileged* must only be used with
``privileged=True`` (critic) or for rewards / costs / terminations.
"""

from __future__ import annotations

import torch

from quadsim.core import math as qm


# ============================================================================ observations
def rangefinder_down(env, max_range: float = 8.0) -> torch.Tensor:
    """*Deployable sensor*: downward rangefinder (body -z) distance to the ground plane, saturated at ``max_range``."""
    d = env.robot.data
    h = d.root_pos_w[:, 2] - env.env_origins[:, 2]
    cos_tilt = (1 - 2 * (d.root_quat_w[:, 1] ** 2 + d.root_quat_w[:, 2] ** 2)).clamp(min=1e-3)
    return (h / cos_tilt).clamp(0.0, max_range)[:, None]


def baro_altitude(env) -> torch.Tensor:
    """*Deployable*: barometric altitude (noisy, drifting; :class:`~quadsim.core.sensors.BarometerModel`)."""
    return env.baro_meas


def imu_accel(env) -> torch.Tensor:
    """*Deployable*: accelerometer specific force, body frame [m/s^2]."""
    return env.imu_meas[:, 3:6]


def est_position(env) -> torch.Tensor:
    """*Deployable*: estimated position in the local frame [m]."""
    return env.est_pos()


def est_lin_vel_world(env) -> torch.Tensor:
    """*Deployable*: estimated velocity in the local (world-aligned) frame [m/s]."""
    return env.odom_meas[:, 7:10]


def rotor_speeds(env) -> torch.Tensor:
    """*Deployable with ESC telemetry* (e.g. DShot RPM): rotor speeds normalized by the nominal maximum."""
    return env.model.omega / env.params.omega_max


def episode_progress(env) -> torch.Tensor:
    """*Deployable*: elapsed fraction of the episode."""
    return (env.episode_length_buf.float() / env.max_episode_length)[:, None]


def last_action(env) -> torch.Tensor:
    """*Deployable*: the action applied at the previous policy step (combine with ``ObsTermCfg.history``)."""
    return env.actions


def gyro(env) -> torch.Tensor:
    """*Deployable*: IMU angular rate, body frame [rad/s]."""
    return env.imu_meas[:, :3]


def lidar_scan(env, num_h: int = 32, num_v: int = 4, v_fov_deg: tuple = (-15.0, 15.0), max_range: float = 10.0,
               normalize: bool = True) -> torch.Tensor:
    """*Deployable sensor*: body-mounted 3-D LiDAR (``num_v`` channels x ``num_h`` azimuths) ranging the task's
    obstacle primitives (``env.obstacle_primitives()``) and the ground, from the true pose (a real LiDAR measures the
    true world; add noise / rate / latency through the term). Returns (N, num_v * num_h) ranges (/ max_range)."""
    from quadsim.core.sensors import cast
    from quadsim.core.sensors.raycast import lidar_ray_dirs

    key = (num_h, num_v, tuple(v_fov_deg))
    cache = env.__dict__.setdefault("_lidar_dirs", {})
    if key not in cache:
        cache[key] = lidar_ray_dirs(num_h, num_v, v_fov_deg, env.device)
    dirs_b = cache[key]
    d = env.robot.data
    n, r = env.num_envs, dirs_b.shape[0]
    dirs = qm.quat_rotate(d.root_quat_w[:, None].expand(n, r, 4), dirs_b[None].expand(n, r, 3))
    origins = env.local_pos()[:, None].expand(n, r, 3)
    cyl, box = env.obstacle_primitives()
    rng = cast(origins, dirs, cyl, box, ground=True, max_range=max_range)
    return rng / max_range if normalize else rng


def true_lin_vel_body(env) -> torch.Tensor:
    """*Privileged*: true body-frame velocity."""
    return env.robot.data.root_lin_vel_b


def wind_world(env) -> torch.Tensor:
    """*Privileged*: wind velocity, world frame."""
    return env.wind_w


# ============================================================================ rewards (multiplied by weight * dt)
def alive(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def upright(env) -> torch.Tensor:
    """cos(tilt) in [-1, 1]."""
    q = env.robot.data.root_quat_w
    return 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)


def action_l2(env) -> torch.Tensor:
    return (env.actions - env.hover_action).square().sum(-1)


def action_rate_l2(env) -> torch.Tensor:
    return (env.actions - env.prev_actions).square().sum(-1)


def rotor_power(env) -> torch.Tensor:
    """Ideal induced power proxy ``sum_i T_i^1.5`` normalized by its hover value (momentum theory)."""
    t = env.model.k_f * env.model.omega.square()
    t_h = env.params.weight / env.params.num_rotors
    return (t.clamp(min=0.0) ** 1.5).sum(-1) / (env.params.num_rotors * t_h ** 1.5)


def ang_vel_l2(env) -> torch.Tensor:
    return env.robot.data.root_ang_vel_b.square().sum(-1)


# ============================================================================ terminations
def above_height(env, height: float) -> torch.Tensor:
    return env.local_pos()[:, 2] > height


def outside_box(env, half_extent: tuple[float, float, float]) -> torch.Tensor:
    return (env.local_pos().abs() > torch.tensor(half_extent, device=env.device)).any(-1)


def tilt_exceeds(env, max_tilt: float) -> torch.Tensor:
    return torch.acos(upright(env).clamp(-1, 1)) > max_tilt


# ============================================================================ costs (rates; multiplied by weight * dt)
def speed_violation(env, limit: float | None = None, indicator: bool = True) -> torch.Tensor:
    """Speed above ``limit`` [m/s] (default: the task's ``cfg.speed_limit``): indicator (1/0) or excess speed."""
    if limit is None:
        limit = env.cfg.speed_limit
    v = env.robot.data.root_lin_vel_w.norm(dim=-1)
    return (v > limit).float() if indicator else (v - limit).clamp(min=0.0)


def tilt_violation(env, max_tilt: float) -> torch.Tensor:
    return tilt_exceeds(env, max_tilt).float()


def altitude_violation(env, min_height: float = 0.0, max_height: float = 1e9) -> torch.Tensor:
    z = env.local_pos()[:, 2]
    return ((z < min_height) | (z > max_height)).float()


def keep_out_cylinder(env, center_xy: tuple[float, float], radius: float) -> torch.Tensor:
    """Inside a vertical keep-out cylinder (no-fly zone) in the local frame."""
    c = torch.tensor(center_xy, device=env.device)
    return ((env.local_pos()[:, :2] - c).norm(dim=-1) < radius).float()


# ============================================================================ events
def randomize_wind_mean(env, env_ids, speed_range: tuple[float, float] = (0.0, 3.0)) -> None:
    """Horizontal mean wind of random direction and speed for each new episode. Replaces the episode's mean wind
    and acts only when the wind model is enabled (``env.wind.enabled=True``)."""
    k = len(env_ids)
    th = torch.rand(k, device=env.device) * 2 * torch.pi
    s = speed_range[0] + (speed_range[1] - speed_range[0]) * torch.rand(k, device=env.device)
    env.wind.mean[env_ids] = torch.stack([s * torch.cos(th), s * torch.sin(th), torch.zeros_like(s)], -1)


def scale_motor_time_constant(env, env_ids, scale_range: tuple[float, float] = (1.0, 2.0)) -> None:
    """Slower motors (on top of the built-in randomization): multiplies the episode's rotor time constants."""
    g = scale_range[0] + (scale_range[1] - scale_range[0]) * torch.rand(len(env_ids), 1, device=env.device)
    env.model.tau_up[env_ids] *= g
    env.model.tau_down[env_ids] *= g


def random_velocity_kick(env, env_ids, max_speed: float = 1.0) -> None:
    """Instantaneous random velocity change (e.g. a bump or gust) of up to ``max_speed`` per axis."""
    d = env.robot.data
    v = d.root_lin_vel_w[env_ids] + (torch.rand(len(env_ids), 3, device=env.device) * 2 - 1) * max_speed
    state = torch.cat([d.root_pos_w[env_ids], d.root_quat_w[env_ids], v, d.root_ang_vel_w[env_ids]], -1)
    env.robot.write_root_state_to_sim(state, env_ids)


# ============================================================================ curricula
def attribute_curriculum(env, env_ids, attr: str, start: float, end: float, step: float, metric: str = "success",
                         threshold: float = 0.8, min_episodes: int = 256) -> float:
    """Moves the task config attribute ``attr`` from ``start`` toward ``end`` by ``step`` whenever the mean of
    ``metric`` over at least ``min_episodes`` finished episodes *that started at the current level* exceeds
    ``threshold`` (episodes still running when the level changes are not counted toward the new level).
    State lives on the env (``env.curriculum_state``), so it is checkpoint-agnostic and restarts at ``start``."""
    st = env.__dict__.setdefault("curriculum_state", {})
    if attr not in st:
        st[attr] = {"value": start, "sum": 0.0, "count": 0,
                    "episode_level": torch.full((env.num_envs,), float(start), device=env.device)}
        setattr(env.cfg, attr, start)
    s = st[attr]
    m = env.last_episode_metrics.get(metric)
    if m is not None and len(env_ids):
        v = m[env_ids]  # the episodes finishing now (``env_ids`` = played episodes)
        v = v[torch.isfinite(v) & (s["episode_level"][env_ids] == s["value"])]
        s["sum"] += float(v.sum())
        s["count"] += len(v)
    if s["count"] >= min_episodes and s["value"] != end:
        if s["sum"] / s["count"] > threshold:
            s["value"] = min(s["value"] + step, end) if end > start else max(s["value"] - step, end)
            setattr(env.cfg, attr, s["value"])
        s["sum"], s["count"] = 0.0, 0  # judge the next window on fresh episodes
    s["episode_level"][env_ids] = s["value"]  # these envs start their next episode at the current level
    return s["value"]


def env_wrench_l2(env) -> torch.Tensor:
    """*Privileged*: magnitude of the external disturbance force (wind drag excluded) normalized by weight."""
    return (env.push.force.norm(dim=-1) + env.payload_force_w.norm(dim=-1)) / env.params.weight


def heading_error(env, target_yaw: torch.Tensor) -> torch.Tensor:
    return qm.wrap_to_pi(target_yaw - qm.yaw_from_quat(env.robot.data.root_quat_w)).abs()
