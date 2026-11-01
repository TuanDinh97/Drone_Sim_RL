"""Runtime patches for upstream issues in the pinned Isaac Lab release (applied without editing Isaac Lab).

1. WrenchComposer eager pose read (Isaac Lab v2.3.2, isaaclab 0.54.x)
   ``WrenchComposer.__init__`` detects the asset type with ``hasattr(asset.data, "body_link_pos_w")``.
   ``hasattr`` *evaluates* the property, i.e. queries PhysX for link transforms during asset initialization.
   When a viewport is rendered (GUI, ``--video``, livestream) the PLAY callback can run before the PhysX GPU
   buffers exist; the query raises ``Exception("Failed to get rigid body transforms from backend")`` which
   ``hasattr`` does not swallow, so every RigidObject/Articulation fails to initialize. Reproduced on this
   machine with a plain Isaac Lab cuboid (no quadsim code). The patch performs the same detection on the
   *class* (no getter call); pose reads stay lazy exactly as upstream intends (``_link_poses_updated``).
"""

from __future__ import annotations

_APPLIED = False


def _patched_wrench_composer_init(self, asset) -> None:
    import torch
    import warp as wp

    self.num_envs = asset.num_instances
    self.num_bodies = asset.num_bodies if hasattr(asset, "num_bodies") else asset.num_objects
    self.device = asset.device
    self._asset = asset
    self._active = False
    data_cls = type(asset.data)
    if hasattr(data_cls, "body_link_pos_w") and hasattr(data_cls, "body_link_quat_w"):
        self._get_link_position_fn = lambda a=asset: a.data.body_link_pos_w[..., :3]
        self._get_link_quaternion_fn = lambda a=asset: a.data.body_link_quat_w[..., :4]
    elif hasattr(data_cls, "object_link_pos_w") and hasattr(data_cls, "object_link_quat_w"):
        self._get_link_position_fn = lambda a=asset: a.data.object_link_pos_w[..., :3]
        self._get_link_quaternion_fn = lambda a=asset: a.data.object_link_quat_w[..., :4]
    else:
        raise ValueError(f"Unsupported asset type: {asset.__class__.__name__}")
    self._composed_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
    self._composed_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
    self._ALL_ENV_INDICES_WP = wp.from_torch(
        torch.arange(self.num_envs, dtype=torch.int32, device=self.device), dtype=wp.int32)
    self._ALL_BODY_INDICES_WP = wp.from_torch(
        torch.arange(self.num_bodies, dtype=torch.int32, device=self.device), dtype=wp.int32)
    self._composed_force_b_torch = wp.to_torch(self._composed_force_b)
    self._composed_torque_b_torch = wp.to_torch(self._composed_torque_b)
    self._ALL_ENV_INDICES_TORCH = wp.to_torch(self._ALL_ENV_INDICES_WP)
    self._ALL_BODY_INDICES_TORCH = wp.to_torch(self._ALL_BODY_INDICES_WP)
    self._link_poses_updated = False


def apply_isaaclab_patches() -> list[str]:
    """Apply version-gated patches once. Returns the list of applied patch names."""
    global _APPLIED
    if _APPLIED:
        return []
    _APPLIED = True
    applied = []
    try:
        from importlib.metadata import version

        v = version("isaaclab")
    except Exception:
        return applied
    if v.startswith("0.54."):
        import inspect

        from isaaclab.utils import wrench_composer

        src = inspect.getsource(wrench_composer.WrenchComposer.__init__)
        if 'hasattr(self._asset.data, "body_link_pos_w")' in src:  # only patch the known-buggy code
            wrench_composer.WrenchComposer.__init__ = _patched_wrench_composer_init
            applied.append("wrench_composer_lazy_pose_detection")
    return applied
