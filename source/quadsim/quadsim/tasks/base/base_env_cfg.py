"""Configuration of the generic quadrotor environment (shared by all tasks)."""

from __future__ import annotations

from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from quadsim.core.controllers import AttitudeGains, PositionGains, RateGains
from quadsim.core.disturbances import PushCfg, WindCfg
from quadsim.core.params import get_preset
from quadsim.core.sensors import BaroCfg, DepthNoiseCfg, ImuNoiseCfg, OdometryCfg, RgbNoiseCfg
from quadsim.isaac.spawn import QuadrotorSpawnerCfg


@configclass
class DynamicsCfg:
    preset: str = "x500"
    """Vehicle preset name (see :data:`quadsim.core.params.PRESETS`)."""
    rotor_drag: bool = True
    body_drag: bool = True
    ground_effect: bool = True
    rotor_gyroscopics: bool = True
    """Rotor gyroscopic torque and spin-up reaction torque (needs ``rotor_inertia``)."""


@configclass
class ControlCfg:
    mode: str = "ctbr"
    """Policy action interface:
    ``motor``    : 4 normalized rotor thrusts in [-1, 1] -> [0, T_max]
    ``ctbr``     : collective thrust + body rates (onboard PX4-like rate PID at physics rate)
    ``ctatt``    : collective thrust + roll, pitch + yaw rate (attitude + rate loops onboard)
    ``velocity`` : velocity set-point in the yaw-aligned frame + yaw rate (full cascade onboard)
    """
    max_body_rate: tuple[float, float, float] = (6.0, 6.0, 3.0)  # rad/s at |a| = 1
    max_tilt: float = 0.6  # rad (ctatt)
    max_velocity: tuple[float, float, float] = (3.0, 3.0, 2.0)  # m/s (velocity)
    max_yaw_rate: float = 2.0  # rad/s (ctatt, velocity)
    collective_range: tuple[float, float] = (0.0, 1.0)  # fraction of nominal max total thrust
    action_delay_steps: tuple[int, int] = (0, 2)  # physics steps, uniform per episode
    rate_gains: RateGains = field(default_factory=RateGains)
    attitude_gains: AttitudeGains = field(default_factory=AttitudeGains)
    position_gains: PositionGains = field(default_factory=PositionGains)
    use_imu_gyro_for_rate_loop: bool = True
    """The onboard rate controller uses the (noisy, biased) gyro measurement, as on a real FC."""


@configclass
class DomainRandCfg:
    enabled: bool = True
    # startup (written to PhysX once): inertial properties
    mass_scale: tuple[float, float] = (0.9, 1.1)
    inertia_scale: tuple[float, float] = (0.85, 1.15)
    com_offset_xy: float = 0.01  # m, uniform +-
    # per episode (pure model side)
    payload_weight_frac: tuple[float, float] = (-0.05, 0.10)  # extra weight as fraction of mg (constant force)
    k_f_scale: tuple[float, float] = (0.9, 1.1)  # per rotor
    k_m_scale: tuple[float, float] = (0.85, 1.15)
    motor_tau_scale: tuple[float, float] = (0.7, 1.6)
    omega_max_scale: tuple[float, float] = (0.92, 1.0)  # battery sag
    drag_scale: tuple[float, float] = (0.5, 1.5)
    sensor_noise_scale: tuple[float, float] = (0.5, 1.5)


@configclass
class SensorsCfg:
    imu: ImuNoiseCfg = field(default_factory=ImuNoiseCfg)
    odometry: OdometryCfg = field(default_factory=lambda: OdometryCfg(0.01, 0.002, 0.03, 0.01, True, 2, True))
    baro: BaroCfg = field(default_factory=BaroCfg)
    noise_enabled: bool = True


@configclass
class VisionCfg:
    """Exteroceptive configuration. ``mode`` selects the policy's image input."""

    mode: str = "none"  # none | depth | rgb | rgbd
    source: str = "raycast"  # raycast (analytic, fast, depth only) | render (Isaac TiledCamera)
    width: int = 64
    height: int = 48
    hfov_deg: float = 87.0  # RealSense D435 depth HFOV
    offset_pos: tuple[float, float, float] = (0.12, 0.0, 0.02)  # body frame
    pitch_deg: float = 0.0  # positive pitches the camera up
    depth_noise: DepthNoiseCfg = field(default_factory=DepthNoiseCfg)
    rgb_noise: RgbNoiseCfg = field(default_factory=RgbNoiseCfg)
    update_period_steps: int = 1  # policy steps between camera updates (rate = policy_rate / this)


@configclass
class QuadrotorEnvCfg(DirectRLEnvCfg):
    # timing: physics 200 Hz, policy 50 Hz
    decimation: int = 4
    episode_length_s: float = 10.0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=8.0, replicate_physics=True)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
        debug_vis=False,
    )

    # spaces (set by tasks)
    action_space: int = 4
    observation_space: int = 0
    state_space: int = 0

    dynamics: DynamicsCfg = DynamicsCfg()
    control: ControlCfg = ControlCfg()
    randomization: DomainRandCfg = DomainRandCfg()
    sensors: SensorsCfg = SensorsCfg()
    vision: VisionCfg = VisionCfg()
    wind: WindCfg = field(default_factory=WindCfg)
    push: PushCfg = field(default_factory=PushCfg)

    robot: RigidObjectCfg | None = None  # built from the preset in __post_init__ when None

    # generic termination
    min_height: float = 0.05  # m above ground for the CoM
    max_distance: float = 12.0  # m from the env origin
    max_tilt_terminate: float = 3.2  # rad (> pi disables)
    max_lin_vel: float = 20.0
    max_ang_vel: float = 35.0

    # observation options
    include_last_action: bool = True
    debug_vis: bool = False

    def __post_init__(self):
        if self.robot is None:
            params = get_preset(self.dynamics.preset)
            self.robot = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Robot",
                spawn=QuadrotorSpawnerCfg(
                    params=params,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        linear_damping=0.0,
                        angular_damping=0.0,
                        max_linear_velocity=1000.0,
                        max_angular_velocity=1e4,  # deg/s in USD; effectively unlimited
                        max_depenetration_velocity=3.0,
                        enable_gyroscopic_forces=True,
                        solver_position_iteration_count=4,
                        solver_velocity_iteration_count=0,
                        sleep_threshold=0.0,
                        stabilization_threshold=0.0,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
                    activate_contact_sensors=False,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
            )
