"""Policy action interfaces: normalized action -> per-rotor thrust commands on the onboard flight controller.

Each interface is selected by name (``cfg.control.mode``) from :data:`quadsim.registry.ACTION_INTERFACES` and runs
at the physics rate on the (delayed) policy action, using only onboard estimates (state estimate, gyro) and the
nominal vehicle model, exactly like the flight controller of a real vehicle.

Built-in interfaces (``A`` = action dimension, all actions normalized to [-1, 1]):

=============  ===  =====================================================================================
``motor``      R    per-rotor thrust in [0, T_max]                     (no onboard loop; R = number of rotors)
``ctbr``       4    collective thrust + body rates                     -> rate PID -> mixer
``ctatt``      4    collective thrust + roll, pitch + yaw rate         -> attitude P -> rate PID -> mixer
``velocity``   4    velocity in the yaw-aligned frame + yaw rate       -> velocity PI -> attitude -> rate -> mixer
``position``   4    position set-point in a box around ``position_center`` + yaw rate  -> full PX4-like cascade
=============  ===  =====================================================================================

A custom interface subclasses :class:`ActionInterface` and registers itself::

    @ACTION_INTERFACES.register("my_interface")
    class MyInterface(ActionInterface):
        def __call__(self, a, quat, est, gyro):
            ...  # return (N, R) rotor thrusts [N], e.g. via self.fc.mixer.allocate(collective, torque)
"""

from __future__ import annotations

import torch

from quadsim.registry import ACTION_INTERFACES

from .. import math as qm
from ..params import QuadrotorParams
from .cascade import AttitudeController, PositionController, RateController, thrust_vector_to_setpoints
from .mixer import Mixer


class FlightController:
    """Onboard flight-controller state shared by the action interfaces (controllers use the *nominal* model).

    Attributes used by the interfaces: ``mixer``, ``rate_ctrl``, ``att_ctrl``, ``pos_ctrl``, ``max_rate``,
    ``max_vel``, ``t_max_nominal``, ``yaw_sp`` (yaw set-point integrated from yaw-rate commands) and
    ``thrust_gain`` (per-env thrust-map gain, domain randomization)."""

    def __init__(self, params: QuadrotorParams, ctrl_cfg, num_envs: int, dt: float, device,
                 thrust_gain_enabled: bool = False):
        c = ctrl_cfg
        self.params, self.cfg, self.num_envs, self.dt, self.device = params, c, num_envs, dt, device
        self.mixer = Mixer(params, device)
        self.rate_ctrl = RateController(params, num_envs, dt, c.rate_gains, device)
        self.att_ctrl = AttitudeController(c.attitude_gains, device)
        self.pos_ctrl = PositionController(params, num_envs, dt, c.position_gains, device)
        self.max_rate = torch.tensor(c.max_body_rate, device=device)
        self.max_vel = torch.tensor(c.max_velocity, device=device)
        self.t_max_nominal = params.max_total_thrust
        self.thrust_gain = torch.ones(num_envs, device=device)
        self.thrust_gain_enabled = thrust_gain_enabled
        self.yaw_sp = torch.zeros(num_envs, device=device)

    def collective_from_action(self, a0: torch.Tensor) -> torch.Tensor:
        lo, hi = self.cfg.collective_range
        c = (lo + (a0 + 1.0) * 0.5 * (hi - lo)) * self.t_max_nominal
        if self.thrust_gain_enabled:  # thrust-map gain error around nominal hover
            c_h = self.params.weight
            c = torch.clamp(c_h + self.thrust_gain * (c - c_h), min=0.0)
        return c

    def integrate_yaw(self, yaw_rate_action: torch.Tensor) -> torch.Tensor:
        self.yaw_sp = qm.wrap_to_pi(self.yaw_sp + yaw_rate_action * self.cfg.max_yaw_rate * self.dt)
        return self.yaw_sp

    def reset(self, env_ids, quat: torch.Tensor, ang_vel: torch.Tensor):
        self.rate_ctrl.reset(env_ids, ang_vel)
        self.pos_ctrl.reset(env_ids)
        self.yaw_sp[env_ids] = qm.yaw_from_quat(quat)


class ActionInterface:
    """Base class: maps the delayed normalized action to rotor thrust commands at the physics rate."""

    def __init__(self, fc: FlightController):
        self.fc = fc

    @classmethod
    def action_dim(cls, ctrl_cfg, params: QuadrotorParams) -> int:
        return 4

    def hover_action(self) -> torch.Tensor:
        """Action that holds the nominal vehicle in hover (initial action, ``last_action`` at reset)."""
        return torch.zeros(self.action_dim(self.fc.cfg, self.fc.params), device=self.fc.device)

    def reset(self, env_ids):
        """Interface-specific state (the shared flight-controller state is reset by the environment)."""

    def __call__(self, a: torch.Tensor, quat: torch.Tensor, est: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        """``a`` (N, A) delayed action, ``quat`` (N, 4) estimated attitude, ``est`` (N, 13) state estimate
        ``[pos, quat, vel_w, ang_vel]``, ``gyro`` (N, 3). Returns rotor thrusts (N, R) [N]."""
        raise NotImplementedError

    def _collective_hover_action(self) -> torch.Tensor:
        a = torch.zeros(4, device=self.fc.device)
        frac = self.fc.params.weight / self.fc.t_max_nominal
        lo, hi = self.fc.cfg.collective_range
        a[0] = 2.0 * (frac - lo) / (hi - lo) - 1.0
        return a


@ACTION_INTERFACES.register("motor")
class MotorThrust(ActionInterface):
    @classmethod
    def action_dim(cls, ctrl_cfg, params):
        return params.num_rotors

    def hover_action(self):
        frac = self.fc.params.weight / self.fc.t_max_nominal
        return torch.full((self.fc.params.num_rotors,), 2.0 * frac - 1.0, device=self.fc.device)

    def __call__(self, a, quat, est, gyro):
        return (a + 1.0) * 0.5 * self.fc.mixer.t_max


@ACTION_INTERFACES.register("ctbr")
class CollectiveThrustBodyRates(ActionInterface):
    def hover_action(self):
        return self._collective_hover_action()

    def __call__(self, a, quat, est, gyro):
        fc = self.fc
        collective = fc.collective_from_action(a[:, 0])
        rate_sp = a[:, 1:4] * fc.max_rate
        torque = fc.rate_ctrl(rate_sp, gyro)
        return fc.mixer.allocate(collective, torque)


@ACTION_INTERFACES.register("ctatt")
class CollectiveThrustAttitude(ActionInterface):
    def hover_action(self):
        return self._collective_hover_action()

    def __call__(self, a, quat, est, gyro):
        fc = self.fc
        collective = fc.collective_from_action(a[:, 0])
        roll, pitch = a[:, 1] * fc.cfg.max_tilt, a[:, 2] * fc.cfg.max_tilt
        yaw_sp = fc.integrate_yaw(a[:, 3])
        q_sp = qm.quat_from_euler_xyz(roll, pitch, yaw_sp)
        rate_sp = fc.att_ctrl(quat, q_sp)
        torque = fc.rate_ctrl(rate_sp, gyro)
        return fc.mixer.allocate(collective, torque)


def _yaw_frame_to_world(v_h: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([cy * v_h[:, 0] - sy * v_h[:, 1], sy * v_h[:, 0] + cy * v_h[:, 1], v_h[:, 2]], -1)


@ACTION_INTERFACES.register("velocity")
class VelocityYawRate(ActionInterface):
    def __call__(self, a, quat, est, gyro):
        fc = self.fc
        yaw = qm.yaw_from_quat(quat)
        vel_sp = _yaw_frame_to_world(a[:, :3] * fc.max_vel, yaw)
        yaw_sp = fc.integrate_yaw(a[:, 3])
        thrust_w = fc.pos_ctrl.velocity_to_thrust(vel_sp, est[:, 7:10])
        collective, q_sp = thrust_vector_to_setpoints(thrust_w, quat, yaw_sp)
        rate_sp = fc.att_ctrl(quat, q_sp)
        torque = fc.rate_ctrl(rate_sp, gyro)
        return fc.mixer.allocate(collective, torque)


@ACTION_INTERFACES.register("position")
class PositionYawRate(ActionInterface):
    """Absolute position set-point ``p_sp = position_center + a[:3] * position_range`` in the local (env) frame,
    tracked by the onboard P(position) -> PI(velocity) -> attitude -> rate cascade; ``a[3]`` = yaw rate."""

    def __init__(self, fc):
        super().__init__(fc)
        self.center = torch.tensor(fc.cfg.position_center, device=fc.device)
        self.range = torch.tensor(fc.cfg.position_range, device=fc.device)

    def __call__(self, a, quat, est, gyro):
        fc = self.fc
        pos_sp = self.center + a[:, :3] * self.range
        yaw_sp = fc.integrate_yaw(a[:, 3])
        thrust_w = fc.pos_ctrl(pos_sp, est[:, :3], est[:, 7:10])
        collective, q_sp = thrust_vector_to_setpoints(thrust_w, quat, yaw_sp)
        rate_sp = fc.att_ctrl(quat, q_sp)
        torque = fc.rate_ctrl(rate_sp, gyro)
        return fc.mixer.allocate(collective, torque)
