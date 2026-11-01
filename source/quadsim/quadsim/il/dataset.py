"""Episode-structured demonstration datasets (HDF5) for imitation learning and VLA research.

File layout (one file per dataset)::

    attrs: format="quadsim-demos/1", task, step_dt, expert, obs_layout (JSON [[name, dim], ...]), action_semantics,
           vehicle (JSON), config (YAML), git_commit, seed, num_episodes
    /episodes/ep_000000/
        obs/policy   (T, P) float32   deployable policy observation at step t (what the policy saw)
        obs/critic   (T, Q) float32   privileged observation (optional)
        image        (T, C, H, W) uint8  onboard camera (optional; RGB in [0, 255])
        action       (T, A) float32   action applied at step t (normalized [-1, 1])
        reward       (T,)   float32
        true/pos, true/quat, true/lin_vel_w, true/ang_vel_b   ground truth (env-local)
        est/odom     (T, 13)          onboard state estimate
        attrs: instruction, success, length, env_id, metrics (JSON)

All arrays share the time index t: ``action[t]`` was computed from ``obs/*[t]`` / ``image[t]``.
"""

from __future__ import annotations

import json
import subprocess

import h5py
import numpy as np
import torch

FORMAT = "quadsim-demos/1"


def _git_commit():
    """HEAD commit, with ``-dirty`` if tracked files have uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        stderr=subprocess.DEVNULL).decode().strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


class DemoWriter:
    """Collects vectorized rollouts per environment and writes finished episodes that pass ``keep``."""

    def __init__(self, path: str, env, expert: str, obs_layout, record_critic=True, record_images=True, meta=None):
        self.env = env.unwrapped
        self.f = h5py.File(path, "w")
        self.n_saved, self.n_seen = 0, 0
        self.record_critic, self.record_images = record_critic, record_images
        self.buf: list[dict[str, list]] = [self._empty() for _ in range(self.env.num_envs)]
        self.instr = [""] * self.env.num_envs
        a = self.f.attrs
        a["format"] = FORMAT
        a["task"] = (meta or {}).get("task", type(self.env).__name__)
        a["step_dt"] = self.env.step_dt
        a["expert"] = expert
        a["obs_layout"] = json.dumps(obs_layout)
        a["git_commit"] = _git_commit()
        a["control_mode"] = self.env.cfg.control.mode
        from dataclasses import asdict

        a["vehicle"] = json.dumps(asdict(self.env.params), default=str)
        try:
            import yaml

            a["config"] = yaml.safe_dump(self.env.cfg.to_dict(), default_flow_style=False)
        except Exception as e:
            a["config"] = f"<unserializable: {e}>"
        for k, v in (meta or {}).items():
            a[k] = v if isinstance(v, (int, float, str)) else json.dumps(v)
        self.f.create_group("episodes")
        self.refresh_instructions(range(self.env.num_envs))

    @staticmethod
    def _empty():
        return {k: [] for k in ("policy", "critic", "image", "action", "reward", "pos", "quat", "vel", "angvel",
                                "odom")}

    def refresh_instructions(self, env_ids):
        ins = getattr(self.env, "instructions", None)
        for e in env_ids:
            self.instr[e] = ins[e] if ins is not None else ""

    def add(self, obs: dict, action: torch.Tensor):
        """Record step t (call *before* ``env.step``): the observation and the (label) action, plus the state."""
        u = self.env
        d = u.robot.data
        cpu = lambda x: x.detach().float().cpu().numpy()  # noqa: E731
        cols = {"policy": cpu(obs["policy"]), "action": cpu(action),
                "pos": cpu(d.root_pos_w - u.env_origins), "quat": cpu(d.root_quat_w), "vel": cpu(d.root_lin_vel_w),
                "angvel": cpu(d.root_ang_vel_b), "odom": cpu(u.odom_meas)}
        if self.record_critic and "critic" in obs:
            cols["critic"] = cpu(obs["critic"])
        if self.record_images and "image" in obs:
            cols["image"] = (obs["image"].detach() * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        for e in range(u.num_envs):
            b = self.buf[e]
            for k, v in cols.items():
                b[k].append(v[e])

    def add_reward(self, reward: torch.Tensor):
        """Reward of the step recorded by the last :meth:`add` (call right after ``env.step``)."""
        r = reward.detach().float().cpu().numpy()
        for e in range(self.env.num_envs):
            self.buf[e]["reward"].append(r[e])

    def end_episodes(self, env_ids, metrics: dict[int, dict], keep=lambda m: True) -> int:
        """Close the episodes of ``env_ids`` (they were reset by the env); returns the number saved."""
        saved = 0
        for e in env_ids:
            b, m = self.buf[e], metrics.get(e, {})
            self.n_seen += 1
            if b["action"] and keep(m):
                self._write(e, b, m)
                saved += 1
            self.buf[e] = self._empty()
        self.refresh_instructions(env_ids)
        return saved

    def _write(self, e, b, m):
        g = self.f["episodes"].create_group(f"ep_{self.n_saved:06d}")
        st = lambda k, dt=np.float32: np.stack(b[k]).astype(dt)  # noqa: E731
        g.create_dataset("obs/policy", data=st("policy"), compression="gzip")
        if b["critic"]:
            g.create_dataset("obs/critic", data=st("critic"), compression="gzip")
        if b["image"]:
            g.create_dataset("image", data=st("image", np.uint8), compression="gzip", chunks=True)
        g.create_dataset("action", data=st("action"))
        g.create_dataset("reward", data=st("reward"))
        for k, name in (("pos", "true/pos"), ("quat", "true/quat"), ("vel", "true/lin_vel_w"),
                        ("angvel", "true/ang_vel_b"), ("odom", "est/odom")):
            g.create_dataset(name, data=st(k))
        g.attrs["instruction"] = self.instr[e]
        g.attrs["env_id"] = e
        g.attrs["length"] = len(b["action"])
        g.attrs["success"] = float(m.get("success", float("nan")))
        g.attrs["metrics"] = json.dumps({k: float(v) for k, v in m.items()})
        self.n_saved += 1

    def close(self):
        self.f.attrs["num_episodes"] = self.n_saved
        self.f.attrs["episodes_seen"] = self.n_seen
        self.f.close()


class DemoDataset(torch.utils.data.Dataset):
    """Transitions of a demo file (all episodes in memory), split by episode into train / val."""

    def __init__(self, path: str, split: str = "train", val_frac: float = 0.1, seed: int = 0, images: bool = True,
                 device="cpu"):
        with h5py.File(path, "r") as f:
            self.attrs = dict(f.attrs)
            names = sorted(f["episodes"])
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(names))
            n_val = int(round(val_frac * len(names)))
            keep = perm[:n_val] if split == "val" else perm[n_val:] if split == "train" else perm
            self.episodes = [names[i] for i in sorted(keep)]
            obs, act, img, ep, instr = [], [], [], [], []
            for i, name in enumerate(self.episodes):
                g = f["episodes"][name]
                obs.append(g["obs/policy"][()])
                act.append(g["action"][()])
                if images and "image" in g:
                    img.append(g["image"][()])
                ep.append(np.full(len(g["action"]), i))
                instr.append(g.attrs.get("instruction", ""))
        self.obs = torch.as_tensor(np.concatenate(obs), device=device)
        self.action = torch.as_tensor(np.concatenate(act), device=device)
        self.image = torch.as_tensor(np.concatenate(img), device=device) if img else None
        self.episode = torch.as_tensor(np.concatenate(ep), device=device)
        self.instructions = instr
        self.obs_layout = json.loads(self.attrs["obs_layout"])

    def __len__(self):
        return len(self.action)

    def __getitem__(self, i):
        out = {"obs": self.obs[i], "action": self.action[i]}
        if self.image is not None:
            out["image"] = self.image[i]
        return out


def summarize(path: str) -> dict:
    """Episode statistics of a demo file (count, lengths, success rate, instructions)."""
    with h5py.File(path, "r") as f:
        eps = f["episodes"]
        lens = [eps[k].attrs["length"] for k in eps]
        succ = [eps[k].attrs["success"] for k in eps]
        instr = [eps[k].attrs.get("instruction", "") for k in eps]
        first = eps[sorted(eps)[0]] if len(eps) else None
        return {"file": path, "task": f.attrs.get("task"), "expert": f.attrs.get("expert"),
                "episodes": len(lens), "episodes_seen": int(f.attrs.get("episodes_seen", len(lens))),
                "transitions": int(np.sum(lens)) if lens else 0,
                "mean_length": float(np.mean(lens)) if lens else 0.0,
                "success_rate": float(np.nanmean(succ)) if succ else float("nan"),
                "distinct_instructions": len(set(instr)),
                "keys": sorted(_keys(first)) if first is not None else []}


def _keys(g, prefix=""):
    out = []
    for k, v in g.items():
        if isinstance(v, h5py.Group):
            out += _keys(v, prefix + k + "/")
        else:
            out.append(prefix + k)
    return out
