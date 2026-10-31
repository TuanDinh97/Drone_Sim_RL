"""Procedural USD quadrotor spawner.

The vehicle is a single PhysX rigid body whose mass, inertia tensor and centre of mass are
written *explicitly* from :class:`~quadsim.core.params.QuadrotorParams` (USD ``MassAPI``), so the
simulated inertial properties are exactly the identified ones and not derived from the visual
geometry. Children are simple shapes (central box, arms, rotor discs, heading marker) used for
rendering and collision only. No network assets are required.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING

from pxr import Gf, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass

from quadsim.core.params import QuadrotorParams


def _box(stage, path, size, pos, yaw=0.0, collide=False, material=None):
    create_prim(
        path,
        "Cube",
        translation=pos,
        orientation=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
        scale=(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0),
        attributes={"size": 2.0},
        stage=stage,
    )
    if collide:
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))
    if material is not None:
        bind_visual_material(path, material, stage=stage)


def _cylinder(stage, path, radius, height, pos, collide=False, material=None):
    create_prim(
        path, "Cylinder", translation=pos, attributes={"radius": radius, "height": height, "axis": "Z"}, stage=stage
    )
    if collide:
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))
    if material is not None:
        bind_visual_material(path, material, stage=stage)


@clone
def spawn_quadrotor(
    prim_path: str,
    cfg: QuadrotorSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    p: QuadrotorParams = cfg.params
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)

    looks = prim_path + "/Looks"
    mat_body = f"{looks}/body"
    mat_front = f"{looks}/front"
    mat_rotor_ccw = f"{looks}/rotor_ccw"
    mat_rotor_cw = f"{looks}/rotor_cw"
    for path, color in (
        (mat_body, cfg.body_color),
        (mat_front, (0.9, 0.15, 0.1)),
        (mat_rotor_ccw, (0.15, 0.45, 0.95)),
        (mat_rotor_cw, (0.95, 0.75, 0.1)),
    ):
        mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.1, roughness=0.6)
        mat_cfg.func(path, mat_cfg)

    geo = prim_path + "/geometry"
    create_prim(geo, "Xform", stage=stage)
    collide = cfg.collision_props is not None
    _box(stage, f"{geo}/body", p.body_size, (0.0, 0.0, 0.0), collide=collide, material=mat_body)
    # heading marker (visual only)
    _box(stage, f"{geo}/nose", (0.04, 0.03, 0.02), (p.body_size[0] / 2.0, 0.0, 0.0), material=mat_front)
    for i, (pos, spin) in enumerate(zip(p.rotor_positions, p.rotor_spin)):
        x, y, z = pos
        length = math.hypot(x, y)
        _box(stage, f"{geo}/arm{i}", (length, p.arm_width, p.arm_width), (x / 2.0, y / 2.0, 0.0),
             yaw=math.atan2(y, x), collide=False, material=mat_body)
        _cylinder(stage, f"{geo}/rotor{i}", p.rotor_radius, 0.01, (x, y, z), collide=collide,
                  material=mat_rotor_ccw if spin > 0 else mat_rotor_cw)
    if collide:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(geo)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                schemas.define_collision_properties(str(prim.GetPath()), cfg.collision_props, stage=stage)

    # explicit inertial properties on the rigid-body prim
    root = stage.GetPrimAtPath(prim_path)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr().Set(float(p.mass))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(v) for v in p.inertia]))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # body-level MassAPI values take precedence over collider-derived mass (UsdPhysics rules);
    # this is verified at runtime against the PhysX tensor view (scripts/validation/m2_*).
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    if cfg.activate_contact_sensors:
        schemas.activate_contact_sensors(prim_path, threshold=0.0, stage=stage)
    return root


@configclass
class QuadrotorSpawnerCfg(RigidObjectSpawnerCfg):
    func: Callable = spawn_quadrotor
    params: QuadrotorParams = MISSING
    body_color: tuple[float, float, float] = (0.12, 0.12, 0.14)
