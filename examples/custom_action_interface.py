"""Example: a new control interface - acceleration commands (a common high-level interface in drone RL).

    python scripts/rl/train.py --headless --task Quadsim-Hover-v0 --plugins examples/custom_action_interface.py \
        env.control.mode=acceleration

The policy outputs a desired acceleration in the yaw-aligned frame (a[:3] * max_acc) and a yaw rate; the onboard
flight controller converts it to a thrust vector m (a + g), then to a collective thrust and attitude set-point,
which the attitude and body-rate loops track (all at the physics rate, from onboard estimates only).
"""

import torch

from quadsim.core import math as qm
from quadsim.core.controllers.cascade import thrust_vector_to_setpoints
from quadsim.core.controllers.interfaces import ActionInterface, _yaw_frame_to_world
from quadsim.registry import ACTION_INTERFACES


@ACTION_INTERFACES.register("acceleration", overwrite=True)
class AccelerationYawRate(ActionInterface):
    max_acc = (6.0, 6.0, 4.0)  # m/s^2 at |a| = 1

    def __init__(self, fc):
        super().__init__(fc)
        self.scale = torch.tensor(self.max_acc, device=fc.device)

    def __call__(self, a, quat, est, gyro):
        fc = self.fc
        acc = _yaw_frame_to_world(a[:, :3] * self.scale, qm.yaw_from_quat(quat))
        acc[:, 2] = (acc[:, 2] + 9.81).clamp(min=0.2 * 9.81)
        yaw_sp = fc.integrate_yaw(a[:, 3])
        collective, q_sp = thrust_vector_to_setpoints(fc.params.mass * acc, quat, yaw_sp)
        return fc.mixer.allocate(collective, fc.rate_ctrl(fc.att_ctrl(quat, q_sp), gyro))
