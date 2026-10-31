"""Physical parameters of multirotor vehicles and reference presets.

Rotor ordering and spin directions follow the PX4 *Quadrotor X* convention:
    index 0: front-right, CCW      index 2: front-left, CW
    index 1: rear-left,   CCW      index 3: rear-right, CW
``spin = +1`` means the propeller spins counter-clockwise seen from above
(positive about body +z). Its aerodynamic reaction torque on the body is
``-spin * k_m * T`` about body +z.

All quantities are SI. Thrust model: ``T = k_f * omega^2`` with ``omega`` in rad/s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass
class QuadrotorParams:
    name: str = "generic"

    # rigid body ---------------------------------------------------------------
    mass: float = 1.0
    """Total vehicle mass [kg]."""
    inertia: tuple[float, float, float] = (0.01, 0.01, 0.02)
    """Principal moments of inertia about the CoM, body FLU axes [kg m^2]."""

    # rotors -------------------------------------------------------------------
    rotor_positions: tuple[tuple[float, float, float], ...] = (
        (0.15, -0.15, 0.0),
        (-0.15, 0.15, 0.0),
        (0.15, 0.15, 0.0),
        (-0.15, -0.15, 0.0),
    )
    """Rotor hub positions relative to the CoM, body frame [m]."""
    rotor_spin: tuple[int, ...] = (1, 1, -1, -1)
    k_f: float = 1e-5
    """Thrust coefficient [N / (rad/s)^2]."""
    k_m: float = 0.016
    """Yaw-moment-to-thrust ratio [m]."""
    omega_min: float = 0.0
    omega_max: float = 1000.0
    """Rotor speed limits [rad/s]."""
    tau_up: float = 0.0125
    tau_down: float = 0.025
    """First-order rotor-speed time constants when accelerating / decelerating [s]."""
    rotor_inertia: float = 0.0
    """Rotor + motor-bell inertia about the spin axis [kg m^2] (gyroscopic & reaction effects)."""
    rotor_radius: float = 0.1
    """Propeller radius [m] (ground effect, visuals and collision)."""

    # aerodynamics -------------------------------------------------------------
    rotor_drag_coeff: float = 0.0
    """Linear rotor (blade-flapping/induced) drag: F = -c * omega * v_perp [N s / (m rad)]."""
    rolling_moment_coeff: float = 0.0
    """Rotor rolling moment: M = -c * omega * v_perp [N s / rad]."""
    body_drag_lin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Linear parasitic drag, body axes [N / (m/s)]."""
    body_drag_quad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Quadratic parasitic drag 0.5*rho*Cd*A, body axes [N / (m/s)^2]."""
    ground_effect: bool = True
    """Cheeseman-Bennett in-ground-effect thrust augmentation."""

    # geometry for spawning ----------------------------------------------------
    body_size: tuple[float, float, float] = (0.15, 0.15, 0.06)
    """Central-body collision box (x, y, z) [m]."""
    arm_width: float = 0.02

    # bookkeeping --------------------------------------------------------------
    source: str = ""
    notes: dict = field(default_factory=dict)

    # ---------------------------------------------------------------------- derived
    @property
    def num_rotors(self) -> int:
        return len(self.rotor_positions)

    @property
    def weight(self) -> float:
        return self.mass * 9.81

    @property
    def max_thrust_per_rotor(self) -> float:
        return self.k_f * self.omega_max**2

    @property
    def max_total_thrust(self) -> float:
        return self.num_rotors * self.max_thrust_per_rotor

    @property
    def thrust_to_weight(self) -> float:
        return self.max_total_thrust / self.weight

    @property
    def hover_omega(self) -> float:
        return math.sqrt(self.weight / (self.num_rotors * self.k_f))

    @property
    def arm_length(self) -> float:
        x, y, _ = self.rotor_positions[0]
        return math.hypot(x, y)

    def copy(self, **changes) -> QuadrotorParams:
        return replace(self, **changes)


def _composite_inertia(
    base_mass: float,
    base_inertia: tuple[float, float, float],
    part_mass: float,
    part_inertia: tuple[float, float, float],
    part_positions,
) -> tuple[float, tuple[float, float, float], float]:
    """Total mass, principal inertia about the composite CoM and CoM height (point-mass rotors)."""
    n = len(part_positions)
    m_tot = base_mass + n * part_mass
    com = [sum(part_mass * p[i] for p in part_positions) / m_tot for i in range(3)]
    ixx = base_inertia[0] + base_mass * (com[1] ** 2 + com[2] ** 2)
    iyy = base_inertia[1] + base_mass * (com[0] ** 2 + com[2] ** 2)
    izz = base_inertia[2] + base_mass * (com[0] ** 2 + com[1] ** 2)
    for p in part_positions:
        dx, dy, dz = p[0] - com[0], p[1] - com[1], p[2] - com[2]
        ixx += part_inertia[0] + part_mass * (dy**2 + dz**2)
        iyy += part_inertia[1] + part_mass * (dx**2 + dz**2)
        izz += part_inertia[2] + part_mass * (dx**2 + dy**2)
    return m_tot, (ixx, iyy, izz), com[2]


def make_x500() -> QuadrotorParams:
    """Holybro X500 as modelled in PX4 SITL (PX4-gazebo-models ``x500``/``x500_base``).

    Base link 2.0 kg, inertia diag(0.021667, 0.021667, 0.04); four 0.016077 kg rotors at
    (+-0.174, +-0.174, 0.06) m. Rotor inertia: the blade inertia averaged over a revolution
    about the in-plane axes (ixx=3.85e-7, iyy=2.61e-5 -> 1.33e-5) and izz=2.65e-5 about the spin axis.
    """
    base_rotor_pos = ((0.174, -0.174, 0.06), (-0.174, 0.174, 0.06), (0.174, 0.174, 0.06), (-0.174, -0.174, 0.06))
    j_spin = 2.649858234714004e-05
    j_inplane = 0.5 * (3.8464910483993325e-07 + 2.6115851691700804e-05)
    mass, inertia, z_com = _composite_inertia(
        2.0, (0.02166666666666667, 0.02166666666666667, 0.04), 0.016076923076923075,
        (j_inplane, j_inplane, j_spin), base_rotor_pos,
    )
    rotor_pos = tuple((x, y, z - z_com) for x, y, z in base_rotor_pos)
    return QuadrotorParams(
        name="x500",
        mass=mass,
        inertia=inertia,
        rotor_positions=rotor_pos,
        rotor_spin=(1, 1, -1, -1),
        k_f=8.54858e-06,
        k_m=0.016,
        omega_min=0.0,
        omega_max=1000.0,
        tau_up=0.0125,
        tau_down=0.025,
        rotor_inertia=j_spin,
        rotor_radius=0.127,  # 10-inch propellers
        rotor_drag_coeff=8.06428e-05,
        rolling_moment_coeff=1e-06,
        body_drag_lin=(0.0, 0.0, 0.0),
        # 0.5 * rho * Cd * A with Cd~1, frontal ~0.06 m^2, top ~0.12 m^2 (engineering estimate, not identified)
        body_drag_quad=(0.035, 0.035, 0.07),
        ground_effect=True,
        body_size=(0.24, 0.24, 0.10),
        arm_width=0.025,
        source="PX4-gazebo-models x500/x500_base model.sdf (gz-sim multicopter motor model)",
        notes={"body_drag_quad": "engineering estimate", "base_link_z_offset_m": z_com},
    )


def make_crazyflie() -> QuadrotorParams:
    """Bitcraze Crazyflie 2.x (Foerster 2015 identification as used by gym-pybullet-drones ``cf2x``).

    k_f = 3.16e-10 N/rpm^2 -> 2.8816e-8 N/(rad/s)^2; k_m = 7.94e-12/3.16e-10 = 0.025127 m.
    omega_max from thrust-to-weight 2.25. Motor time constant is an approximate literature value.
    """
    rpm2rad = 2.0 * math.pi / 60.0
    k_f = 3.16e-10 / rpm2rad**2
    mass = 0.027
    omega_hover = math.sqrt(mass * 9.81 / (4 * k_f))
    a = 0.028
    return QuadrotorParams(
        name="crazyflie",
        mass=mass,
        inertia=(1.4e-5, 1.4e-5, 2.17e-5),
        rotor_positions=((a, -a, 0.0), (-a, a, 0.0), (a, a, 0.0), (-a, -a, 0.0)),
        rotor_spin=(1, 1, -1, -1),
        k_f=k_f,
        k_m=7.94e-12 / 3.16e-10,
        omega_min=0.0,
        omega_max=omega_hover * math.sqrt(2.25),
        tau_up=0.03,
        tau_down=0.03,
        rotor_inertia=1.0e-8,
        rotor_radius=0.0231348,
        rotor_drag_coeff=9.1785e-7 / rpm2rad,
        rolling_moment_coeff=0.0,
        body_drag_lin=(0.0, 0.0, 0.0),
        body_drag_quad=(0.0003, 0.0003, 0.0006),
        ground_effect=True,
        body_size=(0.05, 0.05, 0.02),
        arm_width=0.006,
        source="Foerster 2015 (ETH SysID of Crazyflie 2.0); gym-pybullet-drones cf2x.urdf",
        notes={"tau": "approximate", "body_drag_quad": "engineering estimate"},
    )


PRESETS = {"x500": make_x500, "crazyflie": make_crazyflie}


def get_preset(name: str) -> QuadrotorParams:
    try:
        return PRESETS[name]()
    except KeyError as e:
        raise KeyError(f"Unknown quadrotor preset '{name}'. Available: {list(PRESETS)}") from e
