"""Proprioceptive / navigation sensor models: IMU, barometer, magnetometer, GPS, odometry.

Noise conventions follow the Kalibr/rotors_simulator IMU model:
  * white noise density ``sigma_c`` [unit/sqrt(Hz)] -> discrete std ``sigma_c / sqrt(dt_sample)``;
  * bias = turn-on bias (per episode) + Gauss-Markov drift with random-walk intensity
    ``sigma_rw`` [unit/s/sqrt(Hz)] and correlation time ``tau``.
Default IMU numbers are the PX4 / rotors_simulator SITL defaults (ADIS16448-class).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..math import quat_exp_rotvec, quat_mul, quat_rotate_inverse
from .noise import DelayBuffer, GaussMarkov, SampleAndHold

GRAVITY_W = (0.0, 0.0, -9.81)


@dataclass
class ImuNoiseCfg:
    rate_hz: float = 200.0
    gyro_noise_density: float = 3.394e-4  # rad/s/sqrt(Hz)
    gyro_random_walk: float = 3.8785e-5  # rad/s^2/sqrt(Hz)
    gyro_bias_tau: float = 1000.0  # s
    gyro_turn_on_bias: float = 8.7e-3  # rad/s (1 sigma)
    acc_noise_density: float = 4.0e-3  # m/s^2/sqrt(Hz)
    acc_random_walk: float = 6.0e-3  # m/s^3/sqrt(Hz)
    acc_bias_tau: float = 300.0  # s
    acc_turn_on_bias: float = 0.196  # m/s^2 (1 sigma)
    scale_error: float = 0.0  # 1-sigma scale-factor error (calibration)
    misalignment: float = 0.0  # 1-sigma axis misalignment [rad]
    delay_steps_max: int = 0  # latency in simulation steps (uniformly sampled per episode)
    enabled: bool = True


class ImuModel:
    def __init__(self, cfg: ImuNoiseCfg, num_envs: int, sim_dt: float, device):
        self.cfg = cfg
        self.n = num_envs
        self.sim_dt = sim_dt
        self.device = device
        self.sample_dt = 1.0 / cfg.rate_hz
        # bias drift: OU with stationary std sigma_rw*sqrt(tau/2)
        self.gyro_drift = GaussMarkov((num_envs, 3), cfg.gyro_random_walk * math.sqrt(cfg.gyro_bias_tau / 2),
                                      cfg.gyro_bias_tau, device)
        self.acc_drift = GaussMarkov((num_envs, 3), cfg.acc_random_walk * math.sqrt(cfg.acc_bias_tau / 2),
                                     cfg.acc_bias_tau, device)
        self.gyro_bias0 = torch.zeros(num_envs, 3, device=device)
        self.acc_bias0 = torch.zeros(num_envs, 3, device=device)
        self.gyro_mat = torch.eye(3, device=device).repeat(num_envs, 1, 1)
        self.acc_mat = torch.eye(3, device=device).repeat(num_envs, 1, 1)
        self.hold = SampleAndHold(num_envs, (6,), cfg.rate_hz, sim_dt, device)
        self.delay = DelayBuffer(num_envs, max(cfg.delay_steps_max, 0), (6,), device)
        self.gravity = torch.tensor(GRAVITY_W, device=device)
        self.prev_vel_w = torch.zeros(num_envs, 3, device=device)
        self.noise_scale = torch.ones(num_envs, 1, device=device)  # randomizable multiplier

    def _imperfection_matrix(self, n):
        c = self.cfg
        m = torch.eye(3, device=self.device).repeat(n, 1, 1)
        if c.scale_error > 0:
            m = m + torch.diag_embed(torch.randn(n, 3, device=self.device) * c.scale_error)
        if c.misalignment > 0:
            off = torch.randn(n, 3, 3, device=self.device) * c.misalignment
            m = m + off * (1 - torch.eye(3, device=self.device))
        return m

    def reset(self, env_ids, lin_vel_w: torch.Tensor):
        n = len(env_ids)
        c = self.cfg
        self.gyro_bias0[env_ids] = torch.randn(n, 3, device=self.device) * c.gyro_turn_on_bias
        self.acc_bias0[env_ids] = torch.randn(n, 3, device=self.device) * c.acc_turn_on_bias
        self.gyro_drift.reset(env_ids)
        self.acc_drift.reset(env_ids)
        self.gyro_mat[env_ids] = self._imperfection_matrix(n)
        self.acc_mat[env_ids] = self._imperfection_matrix(n)
        self.hold.reset(env_ids)
        self.delay.set_delay(torch.randint(0, c.delay_steps_max + 1, (n,), device=self.device), env_ids)
        self.prev_vel_w[env_ids] = lin_vel_w

    def true_specific_force(self, quat_w, lin_vel_w):
        acc_w = (lin_vel_w - self.prev_vel_w) / self.sim_dt
        self.prev_vel_w = lin_vel_w.clone()
        return quat_rotate_inverse(quat_w, acc_w - self.gravity)

    def step(self, quat_w, lin_vel_w, ang_vel_b) -> torch.Tensor:
        """Call once per physics step. Returns delayed, held measurement (N, 6) = [gyro, acc]."""
        f_b = self.true_specific_force(quat_w, lin_vel_w)
        if not self.cfg.enabled:
            return torch.cat([ang_vel_b, f_b], -1)
        c = self.cfg
        bg = self.gyro_bias0 + self.gyro_drift.step(self.sim_dt)
        ba = self.acc_bias0 + self.acc_drift.step(self.sim_dt)
        s = self.noise_scale
        gyro = (self.gyro_mat @ ang_vel_b[..., None]).squeeze(-1) + bg
        gyro = gyro + s * torch.randn_like(gyro) * c.gyro_noise_density / math.sqrt(self.sample_dt)
        acc = (self.acc_mat @ f_b[..., None]).squeeze(-1) + ba
        acc = acc + s * torch.randn_like(acc) * c.acc_noise_density / math.sqrt(self.sample_dt)
        held, _ = self.hold.step(torch.cat([gyro, acc], -1))
        return self.delay.push(held)


@dataclass
class BaroCfg:
    rate_hz: float = 50.0
    noise_std: float = 0.1  # m
    bias_std: float = 0.5  # m, Gauss-Markov stationary std
    bias_tau: float = 120.0  # s
    enabled: bool = True


class BarometerModel:
    """Barometric altimeter: ISA pressure from altitude, noisy altitude reading [m] and pressure [Pa]."""

    P0, T0, L, G, M, R = 101325.0, 288.15, 0.0065, 9.80665, 0.0289644, 8.3144598

    def __init__(self, cfg: BaroCfg, num_envs: int, sim_dt: float, device):
        self.cfg = cfg
        self.bias = GaussMarkov((num_envs, 1), cfg.bias_std, cfg.bias_tau, device)
        self.hold = SampleAndHold(num_envs, (1,), cfg.rate_hz, sim_dt, device)
        self.sim_dt = sim_dt

    @classmethod
    def pressure_from_altitude(cls, h):
        return cls.P0 * (1 - cls.L * h / cls.T0) ** (cls.G * cls.M / (cls.R * cls.L))

    @classmethod
    def altitude_from_pressure(cls, p):
        return cls.T0 / cls.L * (1 - (p / cls.P0) ** (cls.R * cls.L / (cls.G * cls.M)))

    def reset(self, env_ids):
        self.bias.reset(env_ids)
        self.hold.reset(env_ids)

    def step(self, altitude: torch.Tensor) -> torch.Tensor:
        h = altitude[:, None]
        if self.cfg.enabled:
            h = h + self.bias.step(self.sim_dt) + torch.randn_like(h) * self.cfg.noise_std
        held, _ = self.hold.step(h)
        return held


@dataclass
class MagCfg:
    field_w: tuple[float, float, float] = (0.21, 0.0, -0.43)  # Gauss, mid-latitude (N-up frame approx.)
    noise_std: float = 0.005
    enabled: bool = True


class MagnetometerModel:
    def __init__(self, cfg: MagCfg, num_envs: int, device):
        self.cfg = cfg
        self.field = torch.tensor(cfg.field_w, device=device).expand(num_envs, 3)

    def measure(self, quat_w):
        m = quat_rotate_inverse(quat_w, self.field)
        if self.cfg.enabled:
            m = m + torch.randn_like(m) * self.cfg.noise_std
        return m


@dataclass
class OdometryCfg:
    """Noise of the onboard state estimate (EKF / VIO / motion capture) seen by the policy."""

    pos_noise_std: float = 0.01  # m, white
    pos_drift_rw: float = 0.002  # m/sqrt(s) random walk (VIO-like drift; 0 for mocap)
    vel_noise_std: float = 0.03  # m/s
    att_noise_std: float = 0.01  # rad (small-angle, white)
    ang_vel_from_imu: bool = True  # policy body rates come from the (noisy) gyro
    delay_steps_max: int = 2  # physics steps of estimator latency (uniform per episode)
    enabled: bool = True
    # EKF-like correlated errors (all 0 = white-noise model above only). Identified from PX4 EKF2 in SITL
    # (docs/PROGRESS.md "estimator model"): per-flight position bias, slow Gauss-Markov position error,
    # correlated velocity error.
    pos_bias_std: tuple[float, float, float] = (0.0, 0.0, 0.0)  # m, constant per episode, per axis
    pos_gm_std: float = 0.0  # m, stationary std of the Gauss-Markov position error
    pos_gm_tau: float = 2.5  # s
    vel_gm_std: float = 0.0  # m/s
    vel_gm_tau: float = 0.5  # s


PROFILES = {
    "perfect": OdometryCfg(0.0, 0.0, 0.0, 0.0, False, 0, False),
    "mocap": OdometryCfg(0.001, 0.0, 0.01, 0.003, True, 2, True),
    "vio": OdometryCfg(0.02, 0.005, 0.05, 0.01, True, 4, True),
    "gps_ins": OdometryCfg(0.5, 0.0, 0.1, 0.01, True, 4, True),
}


class OdometryModel:
    """Noisy, delayed full-state estimate: position, velocity (world), attitude (quaternion)."""

    def __init__(self, cfg: OdometryCfg, num_envs: int, sim_dt: float, device):
        self.cfg = cfg
        self.sim_dt = sim_dt
        self.drift = GaussMarkov((num_envs, 3), 0.0, None, device, random_walk_sigma=cfg.pos_drift_rw)
        self.pos_gm = GaussMarkov((num_envs, 3), cfg.pos_gm_std, cfg.pos_gm_tau, device)
        self.vel_gm = GaussMarkov((num_envs, 3), cfg.vel_gm_std, cfg.vel_gm_tau, device)
        self.bias_std = torch.tensor(cfg.pos_bias_std, device=device)
        self.bias = torch.zeros(num_envs, 3, device=device)
        self.delay = DelayBuffer(num_envs, max(cfg.delay_steps_max, 0), (13,), device)
        self.device = device
        self.noise_scale = torch.ones(num_envs, 1, device=device)

    def reset(self, env_ids, state13: torch.Tensor):
        self.drift.reset(env_ids, initial_std=0.0)
        # only draw random numbers for active terms: the v0 (white-noise) model keeps its exact random stream
        if self.cfg.pos_gm_std > 0:
            self.pos_gm.reset(env_ids)  # stationary start
        if self.cfg.vel_gm_std > 0:
            self.vel_gm.reset(env_ids)
        if any(x > 0 for x in self.cfg.pos_bias_std):
            self.bias[env_ids] = torch.randn(len(env_ids), 3, device=self.device) * self.bias_std
        n = len(env_ids)
        self.delay.set_delay(torch.randint(0, self.cfg.delay_steps_max + 1, (n,), device=self.device), env_ids)
        self.delay.reset(env_ids, state13)

    def step(self, pos_w, quat_w, lin_vel_w, ang_vel_b) -> torch.Tensor:
        """Returns the delayed noisy estimate (N, 13) = [pos(3), quat(4), lin_vel_w(3), ang_vel_b(3)]."""
        c = self.cfg
        if c.enabled:
            s = self.noise_scale
            # v0 terms with the exact v0 arithmetic and random-draw order (bit-identical v0 results); the EKF-like terms
            # (zero unless configured) are added separately
            pos_w = pos_w + self.drift.step(self.sim_dt) + s * torch.randn_like(pos_w) * c.pos_noise_std
            pos_w = pos_w + s * (self.bias + self.pos_gm.step(self.sim_dt))
            lin_vel_w = lin_vel_w + s * torch.randn_like(lin_vel_w) * c.vel_noise_std
            lin_vel_w = lin_vel_w + s * self.vel_gm.step(self.sim_dt)
            dq = quat_exp_rotvec(s * torch.randn_like(pos_w) * c.att_noise_std)
            quat_w = quat_mul(quat_w, dq)
        return self.delay.push(torch.cat([pos_w, quat_w, lin_vel_w, ang_vel_b], -1))


@dataclass
class GpsCfg:
    rate_hz: float = 10.0
    pos_noise_std: tuple[float, float, float] = (0.5, 0.5, 1.0)
    vel_noise_std: float = 0.1
    enabled: bool = True


class GpsModel:
    def __init__(self, cfg: GpsCfg, num_envs: int, sim_dt: float, device):
        self.cfg = cfg
        self.std = torch.tensor(cfg.pos_noise_std, device=device)
        self.hold = SampleAndHold(num_envs, (6,), cfg.rate_hz, sim_dt, device)

    def reset(self, env_ids):
        self.hold.reset(env_ids)

    def step(self, pos_w, lin_vel_w):
        p = pos_w + torch.randn_like(pos_w) * self.std if self.cfg.enabled else pos_w
        v = lin_vel_w + torch.randn_like(lin_vel_w) * self.cfg.vel_noise_std if self.cfg.enabled else lin_vel_w
        held, _ = self.hold.step(torch.cat([p, v], -1))
        return held
