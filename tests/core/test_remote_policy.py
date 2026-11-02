"""Remote policies over the openpi policy-server protocol (CPU, local sockets)."""

import socket
import threading

import numpy as np
import pytest
import torch

msgpack = pytest.importorskip("msgpack")  # optional dependencies of quadsim.il.remote
pytest.importorskip("websockets.sync.client")

from quadsim.il.bc import BCPolicy
from quadsim.il.remote import PolicyClient, RemotePolicyAdapter, bc_policy_fn, make_server, packb, unpackb


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    def __init__(self, policy_fn, meta=None):
        self.port = _free_port()
        self.server = make_server(policy_fn, meta, "127.0.0.1", self.port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.thread.join(timeout=5)


def test_wire_format_matches_openpi_numpy_packing():
    obs = {"observation/state": np.arange(4, dtype=np.float32), "prompt": "go to the red pillar",
           "n": np.int64(3), "img": np.zeros((2, 3, 3), np.uint8), "nested": [{"a": np.ones(2)}]}
    raw = msgpack.unpackb(packb(obs))  # no object hook: the openpi on-the-wire representation
    a = raw["observation/state"]
    assert set(a) == {b"__ndarray__", b"data", b"dtype", b"shape"} and a[b"dtype"] == "<f4" and a[b"shape"] == [4]
    assert raw["n"][b"__npgeneric__"] and raw["n"][b"data"] == 3
    back = unpackb(packb(obs))
    assert np.array_equal(back["observation/state"], obs["observation/state"]) and back["prompt"] == obs["prompt"]
    assert back["img"].dtype == np.uint8 and back["img"].shape == (2, 3, 3) and back["n"] == 3
    assert np.array_equal(back["nested"][0]["a"], np.ones(2))


def test_client_server_round_trip_and_errors():
    def policy_fn(obs):
        if obs.get("fail"):
            raise ValueError("boom")
        return {"actions": obs["observation/state"][:2] * 2.0}

    with _Server(policy_fn, {"policy": "test"}) as srv, PolicyClient("127.0.0.1", srv.port, wait_s=5) as client:
        assert client.metadata == {"policy": "test"}
        out = client.infer({"observation/state": np.array([1.0, 2.0, 3.0], np.float32)})
        assert np.array_equal(out["actions"], np.array([2.0, 4.0], np.float32))
        with pytest.raises(RuntimeError, match="boom"):
            client.infer({"fail": True})


def _small_bc(tmp_path):
    torch.manual_seed(0)
    pol = BCPolicy([["state", 6], ["tokens", 4]], 3, vocab_size=12, hidden=(16,))
    path = str(tmp_path / "bc.pt")
    pol.save(path)
    return BCPolicy.load(path), path


def test_remote_bc_policy_reproduces_local_actions(tmp_path):
    pol, path = _small_bc(tmp_path)
    fn, meta = bc_policy_fn(path)
    assert meta["action_dim"] == 3 and meta["obs_layout"][1] == ["tokens", 4]
    obs = torch.cat([torch.randn(5, 6), torch.randint(2, 12, (5, 4)).float()], -1)
    proprio = obs[:, :6]
    local = pol(obs).detach().clamp(-1, 1)
    with _Server(fn, meta) as srv:
        with PolicyClient("127.0.0.1", srv.port, wait_s=5) as client:
            batched = RemotePolicyAdapter(client, batched=True)(None, ["go to the red pillar"] * 5, proprio, policy_obs=obs)
            single = RemotePolicyAdapter(client, batched=False)(None, ["go to the red pillar"] * 5, proprio, policy_obs=obs)
    assert torch.equal(batched, local)  # same batch, same kernels: bit-identical
    assert torch.allclose(single, local, atol=1e-6)  # one observation per call: matmul of batch 1 (last-bit differences)


def test_adapter_executes_first_action_of_a_chunk():
    seen = {}

    def chunk_policy(obs):
        seen["image"] = obs["observation/image"]
        return {"actions": np.stack([obs["observation/state"][:2], -obs["observation/state"][:2]])}

    with _Server(chunk_policy) as srv, PolicyClient("127.0.0.1", srv.port, wait_s=5) as client:
        out = RemotePolicyAdapter(client, batched=False)(torch.rand(2, 3, 4, 5), ["a", "b"],
                                                         torch.tensor([[0.5, -0.25, 9.0], [0.1, 0.2, 9.0]]))
    assert torch.allclose(out, torch.tensor([[0.5, -0.25], [0.1, 0.2]]))
    assert seen["image"].dtype == np.uint8 and seen["image"].shape == (4, 5, 3)  # openpi image convention: HWC uint8

    def batched_chunk_policy(obs):  # (N, H, A): one chunk per observation
        s = obs["observation/state"][:, :2]
        return {"actions": np.stack([s, -s], axis=1)}

    with _Server(batched_chunk_policy) as srv, PolicyClient("127.0.0.1", srv.port, wait_s=5) as client:
        out = RemotePolicyAdapter(client, batched=True)(None, ["a", "b"], torch.tensor([[0.5, -0.25, 9.0], [0.1, 0.2, 9.0]]))
    assert torch.allclose(out, torch.tensor([[0.5, -0.25], [0.1, 0.2]]))


def test_unclosed_client_does_not_block_interpreter_exit():
    import subprocess
    import sys

    port = _free_port()
    code = (f"import threading; from quadsim.il.remote import PolicyClient, make_server; "
            f"s = make_server(lambda o: o, None, '127.0.0.1', {port}); "
            f"threading.Thread(target=s.serve_forever, daemon=True).start(); "
            f"c = PolicyClient('127.0.0.1', {port}, wait_s=5); print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


@pytest.mark.parametrize("batched", [True, False])
def test_action_horizon_executes_chunks_open_loop_and_resets(batched):
    calls = []

    def chunk_policy(obs):  # chunk[h] = (state_0 + 0.1 h, 0.1 * call number); batched or single observations
        s = np.atleast_2d(obs["observation/state"])
        calls.append(len(s))
        h = 0.1 * np.arange(3, dtype=np.float32)[None, :, None]
        a = np.concatenate([s[:, None, :1] + h, np.full((len(s), 3, 1), 0.1 * len(calls), np.float32)], -1)
        return {"actions": a if obs["observation/state"].ndim == 2 else a[0]}

    s1 = torch.tensor([[0.0, 9.0], [0.5, 9.0]])
    s2 = torch.tensor([[0.2, 9.0], [0.7, 9.0]])
    with _Server(chunk_policy) as srv, PolicyClient("127.0.0.1", srv.port, wait_s=5) as client:
        ad = RemotePolicyAdapter(client, batched=batched, action_horizon=2)
        a1 = ad(None, ["a", "b"], s1)  # both envs query: first action of their chunks
        ad.reset(torch.tensor([1]))  # env 1 starts a new episode: fresh chunk
        a2 = ad(None, ["a", "b"], s2)  # env 0 executes its second action open loop, env 1 queries
        a3 = ad(None, ["a", "b"], s2)  # env 0 re-plans (horizon 2 reached), env 1 executes its second action
    m = [1, 1, 2, 3] if batched else [1, 2, 3, 4]  # message that produced: env 0 / env 1 at call 1, env 1 at call 2, env 0 at call 3
    assert torch.allclose(a1, torch.tensor([[0.0, 0.1 * m[0]], [0.5, 0.1 * m[1]]]))
    assert torch.allclose(a2, torch.tensor([[0.1, 0.1 * m[0]], [0.7, 0.1 * m[2]]]))
    assert torch.allclose(a3, torch.tensor([[0.2, 0.1 * m[3]], [0.8, 0.1 * m[2]]]))
    assert calls == ([2, 1, 1] if batched else [1, 1, 1, 1])  # only environments without pending actions are queried
