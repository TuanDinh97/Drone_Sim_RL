"""Vehicle definitions: registry of presets, YAML/JSON vehicle files and generic multirotor frames.

A vehicle is a :class:`~quadsim.core.params.QuadrotorParams`. It can come from

* a registered preset name (``x500``, ``crazyflie``, ``racer5``, ``hexa_x``, or anything registered in
  :data:`quadsim.registry.VEHICLES`),
* a YAML/JSON file (``cfg.dynamics.preset = "examples/vehicles/my_drone.yaml"``), or
* a Python dict (:func:`params_from_dict`), e.g. converted from an Isaac Lab ``MultirotorCfg``
  (:func:`from_isaaclab_multirotor`).

Vehicle-file schema (all keys optional except what is needed to define thrust)::

    base: x500                 # start from a registered preset, then override fields
    name: my_hexa
    mass: 2.4                  # kg
    inertia: [0.03, 0.03, 0.055]   # kg m^2, or "auto" (box body + point-mass motors, needs motor_mass)
    frame:                     # generic symmetric frame -> rotor_positions / rotor_spin
      num_rotors: 6
      arm_length: 0.275        # m, hub distance from the CoM
      layout: x                # x (no arm on the nose) | plus (first arm on the nose)
      rotor_z: 0.0
    rotor_positions: [[...], ...]  # explicit alternative to ``frame`` (body FLU, m)
    rotor_spin: [1, -1, ...]       # +1 = CCW seen from above
    k_f: 8.5e-6                # N/(rad/s)^2   - or give thrust_to_weight (+ omega_max) and k_f is derived (not both)
    thrust_to_weight: 2.0
    omega_max: 1000.0
    motor_mass: 0.05           # kg per motor+prop, only for inertia: auto
    ... any other QuadrotorParams field (k_m, tau_up, tau_down, rotor_radius, rotor_drag_coeff, body_drag_quad, ...)

Rotor ordering: 4-rotor ``x`` frames follow PX4 Quad-X (0 FR ccw, 1 RL ccw, 2 FL cw, 3 RR cw), so they match the
PX4 bridge. Other frames are numbered counter-clockwise starting at the first arm with alternating spin (CCW first)
and need an even number of rotors. The mixer (pseudo-inverse allocation) handles any rotor geometry whose
zero-torque hover split is positive on every rotor (CoM inside the rotor polygon), including asymmetric ones.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import fields

from quadsim.registry import VEHICLES

from .params import QuadrotorParams, _composite_inertia, make_crazyflie, make_x500

GRAVITY = 9.81
_FIELDS = {f.name for f in fields(QuadrotorParams)}
_DEFAULTS = QuadrotorParams()


def _coerce(name: str, v):
    """Convert a file value to the field's type (YAML 1.1 reads ``1e-5`` - no decimal point - as a string)."""
    default = getattr(_DEFAULTS, name)
    if isinstance(default, bool):
        if not isinstance(v, bool):
            raise TypeError(f"vehicle field '{name}' must be true or false, got {v!r}")
        return v
    if isinstance(default, float):
        return float(v)
    if name == "rotor_positions":
        return tuple(tuple(float(c) for c in r) for r in v)
    if name == "rotor_spin":
        return tuple(int(s) for s in v)
    if isinstance(default, tuple):
        return tuple(float(x) for x in v)
    return v


# ------------------------------------------------------------------------------------------------ geometry
def multirotor_layout(num_rotors: int, arm_length: float, layout: str = "x", rotor_z: float = 0.0):
    """Rotor hub positions and spin directions of a symmetric multirotor frame."""
    if num_rotors < 3:
        raise ValueError("a multirotor needs at least 3 rotors")
    if num_rotors == 4 and layout == "x":
        a = arm_length / math.sqrt(2.0)
        pos = ((a, -a, rotor_z), (-a, a, rotor_z), (a, a, rotor_z), (-a, -a, rotor_z))  # PX4 Quad-X order
        return pos, (1, 1, -1, -1)
    if layout not in ("x", "plus"):
        raise ValueError(f"unknown frame layout '{layout}' (x | plus)")
    if num_rotors % 2:
        raise ValueError("generic frames need an even number of rotors (alternating spin directions)")
    th0 = math.pi / num_rotors if layout == "x" else 0.0
    pos, spin = [], []
    for i in range(num_rotors):
        th = th0 + 2.0 * math.pi * i / num_rotors
        pos.append((arm_length * math.cos(th), arm_length * math.sin(th), rotor_z))
        spin.append(1 if i % 2 == 0 else -1)
    return tuple(pos), tuple(spin)


def estimate_inertia(mass: float, body_size, motor_mass: float, rotor_positions):
    """Principal inertia of a uniform box body plus point-mass motors (engineering estimate for new airframes)."""
    n = len(rotor_positions)
    m_body = mass - n * motor_mass
    if m_body <= 0:
        raise ValueError("motor_mass * num_rotors must be smaller than the total mass")
    x, y, z = body_size
    box = (m_body * (y * y + z * z) / 12.0, m_body * (x * x + z * z) / 12.0, m_body * (x * x + y * y) / 12.0)
    _, inertia, _ = _composite_inertia(m_body, box, motor_mass, (0.0, 0.0, 0.0), rotor_positions)
    return inertia


# ------------------------------------------------------------------------------------------------ dict / file
def params_from_dict(d: dict) -> QuadrotorParams:
    """Build vehicle parameters from a (YAML/JSON-style) dict, see the module docstring for the schema."""
    d = dict(d)
    base = d.pop("base", None)
    p = get_vehicle(base) if base else QuadrotorParams()
    frame = d.pop("frame", None)
    ttw = d.pop("thrust_to_weight", None)
    motor_mass = d.pop("motor_mass", None)
    inertia = d.pop("inertia", None)
    unknown = set(d) - _FIELDS
    if unknown:
        raise KeyError(f"unknown vehicle fields {sorted(unknown)}; valid: {sorted(_FIELDS)}")
    if ttw is not None and "k_f" in d:
        raise ValueError("give either k_f or thrust_to_weight (k_f is then derived), not both")
    changes = {k: _coerce(k, v) for k, v in d.items()}
    if frame is not None:
        pos, spin = multirotor_layout(int(frame["num_rotors"]), float(frame["arm_length"]),
                                      frame.get("layout", "x"), float(frame.get("rotor_z", 0.0)))
        changes.setdefault("rotor_positions", pos)
        changes.setdefault("rotor_spin", spin)
    p = p.copy(**changes)
    if len(p.rotor_spin) != len(p.rotor_positions):
        raise ValueError(f"{len(p.rotor_positions)} rotor positions but {len(p.rotor_spin)} spin directions")
    if ttw is not None and "k_f" not in d:
        p = p.copy(k_f=float(ttw) * p.mass * GRAVITY / (p.num_rotors * p.omega_max ** 2))
    if inertia == "auto":
        if motor_mass is None:
            raise ValueError("inertia: auto needs motor_mass")
        p = p.copy(inertia=estimate_inertia(p.mass, p.body_size, float(motor_mass), p.rotor_positions))
        p.notes = {**p.notes, "inertia": "estimated: box body + point-mass motors"}
    elif inertia is not None:
        p = p.copy(inertia=tuple(float(x) for x in inertia))
    validate_params(p)
    return p


def load_vehicle(path: str) -> QuadrotorParams:
    with open(path) as f:
        if path.endswith(".json"):
            d = json.load(f)
        else:
            import yaml

            d = yaml.safe_load(f)
    d.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    base = d.get("base")
    if isinstance(base, str) and base.endswith((".yaml", ".yml", ".json")) and not os.path.isabs(base):
        local = os.path.join(os.path.dirname(path), base)  # relative to this file (else to the working directory)
        if os.path.exists(local):
            d["base"] = local
    p = params_from_dict(d)
    p.source = p.source or f"vehicle file {path}"
    return p


def validate_params(p: QuadrotorParams) -> None:
    """Physical sanity checks (raise ``ValueError`` with an explanation)."""
    if p.mass <= 0 or min(p.inertia) <= 0:
        raise ValueError(f"{p.name}: mass and inertia must be positive")
    i = p.inertia
    if i[0] + i[1] < i[2] * 0.999 or i[0] + i[2] < i[1] * 0.999 or i[1] + i[2] < i[0] * 0.999:
        raise ValueError(f"{p.name}: inertia {i} violates the triangle inequality")
    if p.thrust_to_weight <= 1.05:
        raise ValueError(f"{p.name}: thrust-to-weight {p.thrust_to_weight:.2f} is too low to fly")
    if p.tau_up <= 0 or p.tau_down <= 0 or p.k_f <= 0 or p.omega_max <= p.omega_min:
        raise ValueError(f"{p.name}: motor parameters must be positive (k_f, tau, omega range)")
    if any(s not in (1, -1) for s in p.rotor_spin):
        raise ValueError(f"{p.name}: rotor_spin entries must be +1 (CCW) or -1 (CW), got {p.rotor_spin}")
    import numpy as np

    pos = np.asarray(p.rotor_positions, dtype=np.float64)
    spin = np.asarray(p.rotor_spin, dtype=np.float64)
    a = np.stack([np.ones_like(spin), pos[:, 1], -pos[:, 0], -spin * p.k_m])
    if np.linalg.matrix_rank(a) < 4:
        raise ValueError(f"{p.name}: rotor geometry cannot produce independent thrust/roll/pitch/yaw (rank < 4)")
    split = np.linalg.pinv(a)[:, 0]
    if split.min() <= 0:
        raise ValueError(f"{p.name}: hovering without torque needs a non-positive thrust on some rotor (CoM outside "
                         f"the rotor polygon); zero-torque split per newton {np.round(split, 3).tolist()}")


# ------------------------------------------------------------------------------------------------ Isaac Lab
def from_isaaclab_multirotor(cfg, mass: float, inertia, name: str = "isaaclab_multirotor", **overrides):
    """Convert an Isaac Lab ``isaaclab_contrib`` :class:`MultirotorCfg` (allocation matrix + ``ThrusterCfg``) into
    quadsim parameters. Mass and inertia live in the robot's USD in Isaac Lab and are therefore passed explicitly.

    Isaac Lab's thruster model is ``T = c * rps^2`` with a first-order rotor-speed lag (``tau_inc``/``tau_dec``);
    the allocation matrix rows are ``[Fx, Fy, Fz, Mx, My, Mz]`` per thruster. Randomization ranges are collapsed to
    their means (quadsim randomizes through ``DomainRandCfg``)."""
    th = next(iter(cfg.actuators.values()))
    a = [list(r) for r in cfg.allocation_matrix]
    n = len(a[0])
    c = sum(th.thrust_const_range) / 2.0
    k_f = c / (2.0 * math.pi) ** 2  # rps^2 -> (rad/s)^2
    t_max = th.thrust_range[1]
    omega_max = math.sqrt(t_max / k_f)
    pos, spin = [], []
    for i in range(n):
        if abs(a[2][i] - 1.0) > 1e-6 or abs(a[0][i]) > 1e-6 or abs(a[1][i]) > 1e-6:
            raise ValueError("only thrusters along body +z are supported")
        pos.append((-a[4][i], a[3][i], 0.0))  # Mx = y T, My = -x T
        spin.append(-1 if a[5][i] > 0 else 1)  # Mz = -spin k_m T
    k_m = sum(abs(a[5][i]) for i in range(n)) / n
    d = dict(name=name, mass=mass, inertia=tuple(inertia), rotor_positions=tuple(pos), rotor_spin=tuple(spin),
              k_f=k_f, k_m=k_m or th.torque_to_thrust_ratio, omega_min=math.sqrt(th.thrust_range[0] / k_f),
              omega_max=omega_max, tau_up=sum(th.tau_inc_range) / 2.0, tau_down=sum(th.tau_dec_range) / 2.0,
              source="converted from an Isaac Lab MultirotorCfg")
    d.update(overrides)
    p = QuadrotorParams(**d)
    validate_params(p)
    return p


# ------------------------------------------------------------------------------------------------ presets
def make_racer5() -> QuadrotorParams:
    """5-inch racing quadrotor (Agilicious-class), engineering approximation for racing research, **not identified**:
    0.75 kg, 0.11 m arms (220 mm motor-to-motor diagonal), thrust-to-weight 5, 5-inch propellers."""
    pos, spin = multirotor_layout(4, 0.11, "x")
    omega_max = 3000.0
    mass = 0.75
    return QuadrotorParams(
        name="racer5", mass=mass, inertia=(2.5e-3, 2.1e-3, 4.3e-3), rotor_positions=pos, rotor_spin=spin,
        k_f=5.0 * mass * GRAVITY / (4 * omega_max ** 2), k_m=0.022, omega_min=0.0, omega_max=omega_max,
        tau_up=0.025, tau_down=0.03, rotor_inertia=2.0e-6, rotor_radius=0.0635, rotor_drag_coeff=1.0e-5,
        rolling_moment_coeff=0.0, body_drag_quad=(0.008, 0.008, 0.016), body_size=(0.10, 0.10, 0.04),
        arm_width=0.012, source="engineering approximation (Agilicious/Foehn 2022-class 5-inch racer)",
        notes={"status": "approximate, not system-identified"},
    )


def make_hexa_x() -> QuadrotorParams:
    """Generic 2.4 kg X-hexarotor (550 mm class) built from the frame generator; engineering approximation."""
    return params_from_dict({
        "name": "hexa_x", "mass": 2.4, "frame": {"num_rotors": 6, "arm_length": 0.275, "layout": "x"},
        "thrust_to_weight": 2.2, "omega_max": 900.0, "k_m": 0.016, "tau_up": 0.015, "tau_down": 0.03,
        "rotor_radius": 0.127, "rotor_inertia": 2.6e-5, "rotor_drag_coeff": 8.0e-5, "rolling_moment_coeff": 1e-6,
        "body_drag_quad": (0.04, 0.04, 0.08), "body_size": (0.25, 0.25, 0.10), "arm_width": 0.025,
        "inertia": "auto", "motor_mass": 0.07, "source": "engineering approximation (generic 550 mm hexarotor)",
    })


for _name, _fn in (("x500", make_x500), ("crazyflie", make_crazyflie), ("racer5", make_racer5),
                   ("hexa_x", make_hexa_x)):
    VEHICLES.register(_name, _fn, overwrite=True)


def get_vehicle(name_or_path: str) -> QuadrotorParams:
    """Registered preset name, or a path to a ``.yaml``/``.yml``/``.json`` vehicle file."""
    if isinstance(name_or_path, QuadrotorParams):
        return name_or_path
    if name_or_path.endswith((".yaml", ".yml", ".json")):
        return load_vehicle(name_or_path)
    p = VEHICLES.get(name_or_path)()
    validate_params(p)  # also covers plugin factories that build QuadrotorParams directly
    return p
