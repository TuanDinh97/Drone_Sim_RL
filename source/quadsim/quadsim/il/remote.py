"""Remote policies: evaluate a model that runs in another process, Python environment or machine.

The wire protocol is the one of openpi's policy server (``scripts/serve_policy.py`` in Physical Intelligence's openpi,
also used by Isaac Lab-Arena's remote policies): a WebSocket connection over which the server first sends a metadata
dict, then answers every observation dict with an action dict; messages are msgpack with numpy arrays packed as
``{b"__ndarray__": True, b"data", b"dtype", b"shape"}``. quadsim's server (``scripts/il/serve_policy.py``) speaks the
same protocol, so the same client evaluates a quadsim checkpoint, an openpi / LeRobot model or any custom server:

    python scripts/il/serve_policy.py --checkpoint logs/il/<name>/model.pt --port 8000        # or openpi's server
    QUADSIM_POLICY_SERVER=localhost:8000 python scripts/rl/evaluate.py --headless --task Quadsim-LangNav-v0 \\
        --controllers policy --algo vla --checkpoint quadsim.il.remote:make_adapter

Observation sent per inference (keys configurable): ``observation/image`` (H, W, 3) uint8 - or the exact float
(3, H, W) image with ``image_format="float_chw"`` -, ``observation/state`` the deployable proprioception,
``observation/policy`` the full deployable policy observation and ``prompt`` the instruction. Expected reply:
``{"actions": (A,) or (horizon, A)}``; the first action of a chunk is executed (the server is queried every step).
Dependencies (optional): ``websockets`` (>= 12) and ``msgpack``.
"""

from __future__ import annotations

import os
import threading
import time
import traceback

import numpy as np
import torch

try:
    import msgpack
except ImportError:  # optional dependency
    msgpack = None


# ------------------------------------------------------------------------------------------------ serialization
def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _require_msgpack():
    if msgpack is None:
        raise ImportError("remote policies need msgpack: pip install msgpack -c env/constraints.txt")


def packb(obj) -> bytes:
    _require_msgpack()
    return msgpack.packb(obj, default=_pack_array)


def unpackb(data: bytes):
    _require_msgpack()
    return msgpack.unpackb(data, object_hook=_unpack_array)


# ------------------------------------------------------------------------------------------------ client / server
class PolicyClient:
    """Synchronous client of an openpi-protocol policy server (usable as a context manager)."""

    def __init__(self, host: str = "localhost", port: int = 8000, wait_s: float = 60.0):
        from websockets.sync.client import connect

        uri, box = f"ws://{host}:{port}", {}

        def _connect():  # in a daemon thread: the connection's receive thread inherits daemon=True, so an unclosed
            t0 = time.time()  # client never keeps the interpreter from exiting
            while True:
                try:
                    box["ws"] = connect(uri, compression=None, max_size=None, open_timeout=wait_s)
                    return
                except OSError as e:  # server not up yet
                    if time.time() - t0 > wait_s:
                        box["error"] = e
                        return
                    time.sleep(0.5)

        th = threading.Thread(target=_connect, daemon=True)
        th.start()
        th.join()
        if "error" in box:
            raise box["error"]
        self.ws = box["ws"]
        self.metadata = unpackb(self.ws.recv())

    def infer(self, obs: dict) -> dict:
        self.ws.send(packb(obs))
        reply = self.ws.recv()
        if isinstance(reply, str):  # the server sends its traceback as text
            raise RuntimeError(f"policy server error:\n{reply}")
        return unpackb(reply)

    def close(self):
        self.ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_server(policy_fn, metadata: dict | None = None, host: str = "0.0.0.0", port: int = 8000):
    """Server of ``policy_fn(obs: dict) -> dict`` (openpi protocol); call ``serve_forever()`` / ``shutdown()``."""
    from websockets.sync.server import serve

    def handler(ws):
        ws.send(packb(metadata or {}))
        for message in ws:
            try:
                ws.send(packb(policy_fn(unpackb(message))))
            except Exception:
                ws.send(traceback.format_exc())
                break

    return serve(handler, host, port, compression=None, max_size=None)


# ------------------------------------------------------------------------------------------------ evaluation adapter
class RemotePolicyAdapter:
    """:class:`~quadsim.il.vla.VLAAdapter` that queries a policy server. ``batched=True`` sends the whole batch of
    environments in one message (quadsim's server); otherwise one message per environment (openpi servers)."""

    wants_policy_obs = True

    def __init__(self, client: PolicyClient, batched: bool = False, image_format: str = "uint8_hwc",
                 keys: dict | None = None):
        self.client, self.batched, self.image_format = client, batched, image_format
        self.keys = {"image": "observation/image", "state": "observation/state", "policy": "observation/policy",
                     "prompt": "prompt", "actions": "actions", **(keys or {})}

    def reset(self, env_ids):
        pass  # the protocol has no reset; stateful servers track episodes themselves

    def _images(self, images):
        if images is None:
            return None
        if self.image_format == "float_chw":
            return images.detach().float().cpu().numpy()
        return (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    def __call__(self, images, instructions, proprio, policy_obs=None):
        k = self.keys
        img = self._images(images)
        state = proprio.detach().float().cpu().numpy()
        pol = None if policy_obs is None else policy_obs.detach().float().cpu().numpy()
        if self.batched:
            obs = {k["state"]: state, k["prompt"]: list(instructions)}
            if img is not None:
                obs[k["image"]] = img
            if pol is not None:
                obs[k["policy"]] = pol
            acts = np.array(self.client.infer(obs)[k["actions"]], dtype=np.float32)
            acts = acts[:, 0] if acts.ndim == 3 else acts  # (N, H, A) action chunks: first action of each
        else:
            rows = []
            for i in range(state.shape[0]):
                obs = {k["state"]: state[i], k["prompt"]: instructions[i]}
                if img is not None:
                    obs[k["image"]] = img[i]
                if pol is not None:
                    obs[k["policy"]] = pol[i]
                a = np.array(self.client.infer(obs)[k["actions"]], dtype=np.float32)
                rows.append(a[0] if a.ndim == 2 else a)  # action chunk: execute its first action, re-plan next step
            acts = np.stack(rows)
        return torch.as_tensor(acts, device=proprio.device).clamp(-1.0, 1.0)


def make_adapter(env):
    """Factory for ``evaluate.py --algo vla --checkpoint quadsim.il.remote:make_adapter``. Configuration through
    environment variables: ``QUADSIM_POLICY_SERVER`` (host:port, default localhost:8000), ``QUADSIM_REMOTE_BATCHED``
    (1 = one message per batch), ``QUADSIM_REMOTE_IMAGE`` (uint8_hwc | float_chw)."""
    host, _, port = os.environ.get("QUADSIM_POLICY_SERVER", "localhost:8000").rpartition(":")
    client = PolicyClient(host or "localhost", int(port))
    print(f"[remote] connected to {host}:{port}, server metadata: {client.metadata}", flush=True)
    return RemotePolicyAdapter(client, batched=os.environ.get("QUADSIM_REMOTE_BATCHED", "0") == "1",
                               image_format=os.environ.get("QUADSIM_REMOTE_IMAGE", "uint8_hwc"))


# ------------------------------------------------------------------------------------------------ serving quadsim BC
def bc_policy_fn(checkpoint: str, device: str = "cpu"):
    """``policy_fn`` serving a quadsim BC checkpoint: reads ``observation/policy`` (+ ``observation/image`` as float
    (3, H, W) or uint8 (H, W, 3), + ``prompt`` for pretrained-text policies), batched or single."""
    from quadsim.il.bc import BCPolicy

    pol = BCPolicy.load(checkpoint, device)
    enc = {}

    def embed(texts):
        missing = [t for t in dict.fromkeys(texts) if t not in pol.text_cache]
        if missing:
            if "e" not in enc:
                from quadsim.il.language import PretrainedTextEncoder

                name, _, sub = pol.text_encoder.partition(":")
                enc["e"] = PretrainedTextEncoder(name, sub or None)
            for t, e in zip(missing, enc["e"].embed(missing)):
                pol.text_cache[t] = e
        return torch.stack([pol.text_cache[t] for t in texts]).to(device)

    @torch.no_grad()
    def policy_fn(obs: dict) -> dict:
        o = torch.as_tensor(np.array(obs["observation/policy"], dtype=np.float32), device=device)  # writable copy
        single = o.dim() == 1
        o = o[None] if single else o
        img = obs.get("observation/image")
        if img is not None:
            img = torch.as_tensor(np.array(img), device=device)
            img = img[None] if single else img
            if img.dtype == torch.uint8:  # (N, H, W, 3) uint8 -> (N, 3, H, W) float
                img = img.permute(0, 3, 1, 2).float() / 255.0
        text = None
        if pol.text_proj is not None:
            prompts = obs.get("prompt")
            text = embed([prompts] if isinstance(prompts, str) else list(prompts))
        a = pol(o, img, text).cpu().numpy()
        return {"actions": a[0] if single else a}

    meta = {"policy": "quadsim-bc", "checkpoint": checkpoint, "action_dim": int(pol.config["act_dim"]),
            "obs_layout": [[n, int(d)] for n, d in pol.layout]}
    return policy_fn, meta
