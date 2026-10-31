"""Synchronized HDF5 recording of quadrotor episodes (state, sensors, actions, images).

Layout (T = recorded policy steps, K = recorded environments)::

    /time                    (T,)          s, simulation time
    /step                    (T,)          policy step index
    /true/{pos,quat,lin_vel_w,ang_vel_b}   (T, K, .)  ground truth (env-local position)
    /est/odom                (T, K, 13)    onboard state estimate [pos, quat, vel_w, ang_vel]
    /sensors/imu             (T, K, 6)     [gyro, accel] (body)
    /sensors/baro            (T, K, 1)
    /action                  (T, K, A)     normalized policy action
    /motor/omega             (T, K, R)     rotor speeds [rad/s]
    /env/wind_w              (T, K, 3)
    /reward, /terminated, /truncated  (T, K)
    /image                   (T, K, C, H, W)  uint16 depth [mm] (C=1) or uint8 RGB(-D)
    attrs: config (YAML), vehicle params (JSON), dt, git commit, seed, task

All arrays share the same time axis, so every row is one synchronized sample taken right after the
policy step (images are those the policy saw at that step).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict

import h5py
import numpy as np
import torch


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


class EpisodeRecorder:
    def __init__(self, path: str, env, env_ids=(0,), record_images: bool = True, chunk: int = 256,
                 meta: dict | None = None):
        self.env = env
        self.ids = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
        self.k = len(self.ids)
        self.f = h5py.File(path, "w")
        self.chunk = chunk
        self.t = 0
        self.record_images = record_images and getattr(env, "vision_enabled", False)
        self._ds: dict[str, h5py.Dataset] = {}
        a = self.f.attrs
        a["dt"] = env.step_dt
        a["physics_dt"] = env.physics_dt
        a["git_commit"] = _git_commit()
        a["vehicle_params"] = json.dumps(asdict(env.params), default=str)
        try:
            import yaml

            a["config"] = yaml.safe_dump(env.cfg.to_dict(), default_flow_style=False)
        except Exception as e:  # cfg serialization should never block recording
            a["config"] = f"<unserializable: {e}>"
        for k, v in (meta or {}).items():
            a[k] = v

    def _write(self, name: str, arr: np.ndarray):
        if name not in self._ds:
            shape = (0,) + arr.shape
            self._ds[name] = self.f.create_dataset(
                name, shape=shape, maxshape=(None,) + arr.shape, dtype=arr.dtype,
                chunks=(min(self.chunk, 64),) + arr.shape, compression="gzip", compression_opts=4,
            )
        ds = self._ds[name]
        ds.resize(self.t + 1, axis=0)
        ds[self.t] = arr

    def record(self, reward=None, terminated=None, truncated=None):
        env, ids = self.env, self.ids
        d = env.robot.data

        def np_(x):
            return x[ids].detach().cpu().numpy()

        self._write("time", np.array(env.common_step_counter * env.step_dt, dtype=np.float64))
        self._write("step", np.array(env.common_step_counter, dtype=np.int64))
        self._write("true/pos", np_(d.root_pos_w - env.scene.env_origins))
        self._write("true/quat", np_(d.root_quat_w))
        self._write("true/lin_vel_w", np_(d.root_lin_vel_w))
        self._write("true/ang_vel_b", np_(d.root_ang_vel_b))
        self._write("est/odom", np_(env.odom_meas))
        self._write("sensors/imu", np_(env.imu_meas))
        self._write("sensors/baro", np_(env.baro_meas))
        self._write("action", np_(env.actions))
        self._write("motor/omega", np_(env.model.omega))
        self._write("env/wind_w", np_(env.wind_w))
        if reward is not None:
            self._write("reward", np_(reward).astype(np.float32))
            self._write("terminated", np_(terminated).astype(np.bool_))
            self._write("truncated", np_(truncated).astype(np.bool_))
        if self.record_images:
            img = env.image[ids].detach()
            if img.shape[1] == 1:  # depth, normalized [0,1] -> mm
                arr = (img * env.cfg.vision.depth_noise.max_range * 1000.0).round().clamp(0, 65535).to(torch.int32)
                self._write("image", arr.cpu().numpy().astype(np.uint16))
            else:
                self._write("image", (img * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy())
        self.t += 1

    def close(self):
        self.f.attrs["num_samples"] = self.t
        self.f.close()


def load(path: str) -> dict:
    """Read a recording into a nested dict of numpy arrays (+ ``attrs``)."""
    out = {"attrs": {}}
    with h5py.File(path, "r") as f:
        out["attrs"] = dict(f.attrs)

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]

        f.visititems(visit)
    return out
