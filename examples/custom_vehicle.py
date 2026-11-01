"""Example: register custom vehicles (Python factory + YAML file) and fly them in any task.

    python scripts/rl/train.py --headless --task Quadsim-Hover-v0 --plugins examples/custom_vehicle.py \
        env.dynamics.preset=quad_250 --max_iterations 300
    python scripts/rl/evaluate.py --headless --task Quadsim-Hover-v0 --plugins examples/custom_vehicle.py \
        --controllers cascade env.dynamics.preset=hexa_550

A vehicle is a :class:`quadsim.core.params.QuadrotorParams`; everything else (spawned USD body, mass/inertia written
to PhysX, motor and aerodynamic model, mixer / allocation, onboard controllers, hover action) is derived from it.
"""

from quadsim.core.vehicles import load_vehicle, multirotor_layout, params_from_dict
from quadsim.registry import VEHICLES


@VEHICLES.register("quad_250", overwrite=True)
def make_quad_250():
    """A 250-class quadrotor built from the frame generator (PX4 Quad-X rotor order)."""
    pos, spin = multirotor_layout(4, arm_length=0.125, layout="x")
    return params_from_dict({
        "name": "quad_250", "mass": 0.55, "rotor_positions": pos, "rotor_spin": spin,
        "thrust_to_weight": 4.0, "omega_max": 2800.0, "k_m": 0.018, "tau_up": 0.02, "tau_down": 0.025,
        "rotor_radius": 0.0635, "rotor_inertia": 3e-6, "rotor_drag_coeff": 1.5e-5,
        "body_drag_quad": (0.006, 0.006, 0.012), "body_size": (0.09, 0.09, 0.04), "arm_width": 0.012,
        "inertia": "auto", "motor_mass": 0.035, "source": "example (not identified)"})


# a YAML vehicle file can also be registered under a short name
VEHICLES.register("hexa_550", lambda: load_vehicle("examples/vehicles/hexa_550.yaml"), overwrite=True)
