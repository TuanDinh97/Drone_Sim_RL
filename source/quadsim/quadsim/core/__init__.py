"""Framework-agnostic quadrotor physics (pure PyTorch, importable without Isaac Sim)."""

from .dynamics import ReferenceQuadrotorSim, RigidState, integrate_rigid_body
from .model import QuadrotorModel, allocation_matrix
from .params import PRESETS, QuadrotorParams, get_preset, make_crazyflie, make_x500
from .vehicles import get_vehicle, load_vehicle, multirotor_layout, params_from_dict
