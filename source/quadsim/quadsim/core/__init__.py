"""Framework-agnostic quadrotor physics (pure PyTorch, importable without Isaac Sim)."""

from .dynamics import ReferenceQuadrotorSim, RigidState, integrate_rigid_body
from .model import QuadrotorModel, allocation_matrix
from .params import PRESETS, QuadrotorParams, get_preset, make_crazyflie, make_x500
