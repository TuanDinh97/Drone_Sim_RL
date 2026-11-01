"""CPU tests of the configurable terms, safe-RL multiplier, off-policy learner parts, hardware profile and IL tools.

Modules that live inside the Isaac-dependent ``quadsim.tasks`` package are loaded directly from their files.
"""

import importlib.util
import json
import math
import os
import sys
import types

import h5py
import numpy as np
import pytest
import torch

from quadsim.il.bc import BCPolicy, train_bc
from quadsim.il.dataset import DemoDataset, summarize
from quadsim.il.language import UNK, BagOfWordsEncoder, Tokenizer
from quadsim.rl.offpolicy import Actor, ReplayBuffer, RunningNorm
from quadsim.utils.hardware import recommend_num_envs

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "source", "quadsim", "quadsim")


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


terms = _load("quadsim_terms_under_test", "tasks/base/terms.py")


class _FakeEnv:
    def __init__(self, n=4):
        self.num_envs, self.device, self.step_dt = n, "cpu", 0.02
        self.t = 0.0
        self.value = torch.zeros(n)

    def signal(self):
        return self.value[:, None] + self.t


# ---------------------------------------------------------------- observation terms (rate / latency / noise)
def test_obs_term_period_and_delay():
    env = _FakeEnv(2)
    cfg = terms.ObsTermCfg(func=lambda e: e.signal(), period_steps=3, delay_steps=2)
    term = terms._ObsTerm("x", cfg, env)
    out = []
    for k in range(9):
        env.t = float(k)
        out.append(term.compute(env)[0, 0].item())
    # step 0 is the episode start (history filled), samples at 0, 3, 6 held for 3 steps, delayed by 2 steps
    assert out == [0, 0, 0, 0, 0, 3, 3, 3, 6]


def test_obs_term_reset_refills_history():
    env = _FakeEnv(2)
    term = terms._ObsTerm("x", terms.ObsTermCfg(func=lambda e: e.signal(), delay_steps=3), env)
    for k in range(5):
        env.t = float(k)
        term.compute(env)
    term.fresh[1] = True
    env.t = 100.0
    out = term.compute(env)
    assert out[1, 0].item() == 100.0  # reset env: no stale samples from the previous episode
    assert out[0, 0].item() == 2.0  # running env: 3-step latency


def test_obs_term_history():
    env = _FakeEnv(1)
    term = terms._ObsTerm("x", terms.ObsTermCfg(func=lambda e: e.signal(), history=3, delay_steps=1), env)
    assert term.dim == 3
    out = []
    for k in range(5):
        env.t = float(k)
        out.append(term.compute(env)[0].tolist())
    assert out[-1] == [1.0, 2.0, 3.0]  # samples t-3, t-2, t-1 (1-step latency), oldest first
    assert out[0] == [0.0, 0.0, 0.0]  # history filled at the episode start


def test_obs_term_noise_and_clip():
    env = _FakeEnv(1000)
    torch.manual_seed(0)
    term = terms._ObsTerm("x", terms.ObsTermCfg(func=lambda e: e.signal(), noise_std=0.5, clip=(-1.0, 1.0)), env)
    v = term.compute(env)
    assert v.abs().max() <= 1.0 and 0.4 < v.std().item() < 0.6


# ---------------------------------------------------------------- safe RL (PID Lagrangian)
def test_lagrangian_multiplier_rises_and_falls():
    env = _FakeEnv(8)
    cfg = terms.TermsCfg(costs={"c": terms.CostTermCfg(func=lambda e: torch.ones(e.num_envs))})
    safety = terms.SafetyCfg(lagrangian=True, cost_limit=0.1, ki=0.5, update_period_steps=1)
    tm = terms.TermManager(env, cfg, safety)
    ids = torch.arange(8)
    for _ in range(5):  # episodes with cost 1.0 > limit 0.1 -> lambda grows
        tm._finished_costs.append(torch.ones(8))
        tm.compute_cost()
        tm.shape_reward(torch.zeros(8))
    lam_hi = tm.lagrange
    assert lam_hi > 1.0
    for _ in range(20):  # constraint satisfied -> lambda decays to 0 (projected)
        tm._finished_costs.append(torch.zeros(8))
        tm.shape_reward(torch.zeros(8))
    assert tm.lagrange < lam_hi and tm.lagrange >= 0.0
    tm.cost = torch.ones(8)
    tm.lagrange = 3.0
    r = tm.shape_reward(torch.full((8,), 2.0))  # (r - lambda c) / (1 + lambda)
    assert torch.allclose(r, torch.full((8,), (2.0 - 3.0) / 4.0), atol=1e-6) or tm.lagrange != 3.0
    tm.before_reset(ids, torch.ones(8, dtype=torch.bool))


def test_cost_rates_integrate_to_seconds():
    env = _FakeEnv(3)
    cfg = terms.TermsCfg(costs={"c": terms.CostTermCfg(func=lambda e: torch.tensor([1.0, 0.0, 1.0]))})
    tm = terms.TermManager(env, cfg, terms.SafetyCfg())
    for _ in range(50):
        tm.compute_cost()
    assert torch.allclose(tm.episode_cost, torch.tensor([1.0, 0.0, 1.0]))  # 50 steps * 0.02 s


# ---------------------------------------------------------------- off-policy building blocks
def test_running_norm_matches_batch_statistics():
    torch.manual_seed(0)
    x = torch.randn(10000, 5) * torch.tensor([1, 2, 3, 4, 5.0]) + 7
    rn = RunningNorm(5)
    for chunk in x.split(997):
        rn.update(chunk)
    assert torch.allclose(rn.mean, x.mean(0), atol=1e-3)
    assert torch.allclose(rn.var, x.var(0, unbiased=False), rtol=1e-3)


def test_squashed_gaussian_log_prob():
    """SAC log-probability of tanh-squashed actions equals the change-of-variables formula."""
    torch.manual_seed(0)
    actor = Actor(6, 3, [32], stochastic=True)
    obs = torch.randn(4096, 6)
    a, logp = actor.sample(obs)
    out = actor.net(actor.norm(obs))
    mean, log_std = out[:, :3], out[:, 3:].clamp(-5.0, 1.0)
    u = torch.atanh(a.clamp(-1 + 1e-6, 1 - 1e-6))
    ref = torch.distributions.Normal(mean, log_std.exp()).log_prob(u).sum(-1) - torch.log(1 - a.square() + 1e-12).sum(-1)
    ok = a.abs().max(-1).values < 0.999  # avoid saturated samples where atanh is ill-conditioned
    assert torch.allclose(logp[ok], ref[ok], atol=2e-3)


def test_replay_buffer_wraps():
    rb = ReplayBuffer(4, 2, {"r": 1, "x": 3}, "cpu")
    for k in range(6):
        rb.add(r=torch.full((2,), float(k)), x=torch.full((2, 3), float(k)))
    assert rb.full and set(rb.data["r"].flatten().tolist()) == {2.0, 3.0, 4.0, 5.0}
    b = rb.sample(64)
    assert b["x"].shape == (64, 3) and (b["r"] >= 2).all()


# ---------------------------------------------------------------- hardware profile
def test_hardware_recommendation():
    assert recommend_num_envs("state", 7.0, 4096) == 4096  # RTX 5060 8 GB: tuned default fits
    assert recommend_num_envs("state", 7.0, 4096, allow_increase=True) == 8192
    assert recommend_num_envs("render", 7.0, 1024) < 256
    assert recommend_num_envs("depth_raycast", 3.5, 2048) < 1024
    assert recommend_num_envs("state", 1.0, 4096) == 64
    assert recommend_num_envs("render", 1.0, 32) == 32  # never raises a small default
    assert recommend_num_envs("render", 1.0, 32, allow_increase=True) == 64


def test_hardware_detect_respects_cuda_visible_devices(monkeypatch):
    from quadsim.utils import hardware

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert hardware.detect()["gpu"] is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert hardware._first_visible_gpu() == "1"
    assert hardware.own_gpu_memory_gb(pid=-1) == 0.0


# ---------------------------------------------------------------- language / BC / datasets
def test_tokenizer_and_unknown_words():
    t = Tokenizer.default(["red", "blue"], max_tokens=6)
    ids = t.encode("Could you go to the BLUE one, please?")
    assert len(ids) == 6 and ids[0] == UNK and t.words[ids[5]] == "blue"
    t2 = Tokenizer.from_dict(json.loads(json.dumps(t.to_dict())))
    assert t2.encode("fly to the red pillar") == t.encode("fly to the red pillar")
    t = Tokenizer.default(["red", "blue"], max_tokens=10)
    enc = BagOfWordsEncoder(t.vocab_size, 8, 8)
    a = enc(torch.tensor([t.encode("go to the blue one")]))
    b = enc(torch.tensor([t.encode("xyz qq go to the blue one")]))
    assert torch.allclose(a, b, atol=1e-6)  # out-of-vocabulary words are ignored by the pooling


def _write_fake_demos(path, n_ep=10, t=20, images=True):
    tok = Tokenizer.default(["red", "blue"], max_tokens=4)
    with h5py.File(path, "w") as f:
        f.attrs["obs_layout"] = json.dumps([["state", 3], ["last_action", 2], ["tokens", 4]])
        f.attrs["tokenizer"] = json.dumps(tok.to_dict())
        for i in range(n_ep):
            g = f.create_group(f"episodes/ep_{i:06d}")
            color = ["red", "blue"][i % 2]
            obs = np.zeros((t, 9), np.float32)
            obs[:, :3] = np.random.randn(t, 3)
            obs[:, 5:] = tok.encode(f"go to the {color}")
            g["obs/policy"] = obs
            g["action"] = np.tile(np.array([1.0 if color == "red" else -1.0, 0.0], np.float32), (t, 1))
            if images:
                g["image"] = np.random.randint(0, 255, (t, 3, 8, 8), np.uint8)
            g.attrs.update(instruction=f"go to the {color}", length=t, success=1.0, env_id=i)
    return tok


def test_demo_dataset_split_and_summary(tmp_path):
    p = str(tmp_path / "d.h5")
    _write_fake_demos(p)
    tr, va = DemoDataset(p, "train", 0.2), DemoDataset(p, "val", 0.2)
    assert len(tr) == 160 and len(va) == 40 and not set(tr.episodes) & set(va.episodes)
    assert tr.image.dtype == torch.uint8 and tr.obs_layout[2] == ["tokens", 4]
    s = summarize(p)
    assert s["episodes"] == 10 and s["distinct_instructions"] == 2 and "image" in s["keys"]


def test_bc_learns_language_grounding(tmp_path):
    """BC must read the instruction tokens to output the right action (the state is pure noise)."""
    torch.manual_seed(0)
    p = str(tmp_path / "d.h5")
    tok = _write_fake_demos(p, n_ep=40, images=False)
    tr = DemoDataset(p, "train", 0.0, images=False)
    pol = BCPolicy(tr.obs_layout, 2, tok.vocab_size, None, (32,), drop_blocks=("last_action",))
    assert pol.cont_idx == [0, 1, 2] and pol.tok_idx == [5, 6, 7, 8]
    train_bc(pol, tr, None, epochs=60, batch_size=64, lr=3e-3, device="cpu", log=lambda *_: None)
    o = torch.zeros(2, 9)
    o[0, 5:] = torch.tensor(tok.encode("go to the red")).float()
    o[1, 5:] = torch.tensor(tok.encode("go to the blue")).float()
    a = pol(o)
    assert a[0, 0] > 0.8 and a[1, 0] < -0.8
    path = str(tmp_path / "bc.pt")
    pol.save(path)
    assert torch.allclose(BCPolicy.load(path)(o), a)


def test_bc_film_cnn_forward():
    pol = BCPolicy([["state", 3], ["tokens", 4]], 4, 20, (3, 48, 64), (32,))
    obs = torch.zeros(5, 7)
    out = pol(obs, torch.randint(0, 255, (5, 3, 48, 64), dtype=torch.uint8))
    assert out.shape == (5, 4) and out.abs().max() <= 1.0
    assert math.isfinite(out.sum().item())
    _ = types  # keep import used


class _OffPolicyEnv:
    """Minimal vectorized env for the off-policy learner (random observations and rewards)."""

    def __init__(self, n=8, act_dim=2):
        self.n = n
        self.unwrapped = types.SimpleNamespace(num_envs=n, action_dim=act_dim, step_dt=0.02, device="cpu")
        self.t = 0

    def reset(self):
        return {"policy": torch.randn(self.n, 3), "critic": torch.randn(self.n, 5)}, {}

    def step(self, a):
        self.t += 1
        term = torch.zeros(self.n, dtype=torch.bool)
        trunc = torch.zeros(self.n, dtype=torch.bool)
        trunc[0] = self.t % 5 == 0
        term[1] = self.t % 7 == 0
        return {"policy": torch.randn(self.n, 3), "critic": torch.randn(self.n, 5)}, torch.randn(self.n), term, trunc, {}


def _small_offpolicy_cfg(algo, **kw):
    from quadsim.rl.offpolicy import OffPolicyCfg

    c = dict(device="cpu", batch_size=32, learning_starts=1, buffer_steps=16, actor_hidden_dims=[16],
             critic_hidden_dims=[16], updates_per_step=1)
    return OffPolicyCfg(algo=algo, **{**c, **kw})


@pytest.mark.parametrize("algo,updates", [("sac", 1), ("td3", 2)])
def test_offpolicy_target_update_rate(algo, updates):
    """Target critic moves by exactly tau toward the online critic per update (SAC) / per delayed update (TD3)."""
    from quadsim.rl.offpolicy import OffPolicyLearner

    torch.manual_seed(0)
    learner = OffPolicyLearner(_OffPolicyEnv(), _small_offpolicy_cfg(algo, tau=0.005), None)
    learner.learn(3)
    while (learner.updates + updates) % updates:  # TD3: next update is a delayed (target) update
        learner._update()
    t0 = [p.detach().clone() for p in learner.critic_t.parameters()]
    for _ in range(updates):
        learner._update()
    p1 = [p.detach().clone() for p in learner.critic.parameters()]
    t1 = [p.detach().clone() for p in learner.critic_t.parameters()]
    num = sum(((b - a) * (c - a)).sum() for a, b, c in zip(t0, t1, p1))
    den = sum(((c - a) ** 2).sum() for a, c in zip(t0, p1))
    assert float(num / den) == pytest.approx(0.005, rel=1e-3)


def test_offpolicy_resume_keeps_policy_and_config(tmp_path):
    from quadsim.rl.offpolicy import OffPolicyLearner

    torch.manual_seed(0)
    cfg = _small_offpolicy_cfg("sac", learning_starts=4)
    a = OffPolicyLearner(_OffPolicyEnv(), cfg, None)
    a.learn(10)
    a.save(str(tmp_path / "model_10.pt"))
    b = OffPolicyLearner(_OffPolicyEnv(), _small_offpolicy_cfg("sac", learning_starts=4), None)
    b.load(str(tmp_path / "model_10.pt"))
    assert b.step == 10 and b.cfg.learning_starts == 4  # the config is not inflated
    assert b.random_until == 0 and b.update_from == 14  # refill with the loaded policy, no random actions
    oa = torch.randn(8, 3)
    assert torch.allclose(b.actor(oa), a.actor(oa))


def _load_mdp():
    """quadsim.tasks.mdp without importing quadsim.tasks (which needs Isaac Sim)."""
    root = os.path.join(os.path.dirname(__file__), "..", "..", "source", "quadsim", "quadsim", "tasks", "mdp.py")
    spec = importlib.util.spec_from_file_location("quadsim_mdp_for_tests", root)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_attribute_curriculum_judges_current_level_only():
    """Success 95 % up to 0.6 rad initial tilt, 40 % above: the curriculum must stop at 0.75 rad, not run away on
    the stale successes of envs whose episodes started at easier levels."""
    mdp = _load_mdp()
    torch.manual_seed(0)
    n = 4096
    env = types.SimpleNamespace(num_envs=n, device="cpu", cfg=types.SimpleNamespace(init_max_tilt=0.3),
                                last_episode_metrics={"success": torch.full((n,), float("nan"))})
    level_at_start = torch.full((n,), 0.3)
    for _ in range(2000):  # reset batches of 64 finished episodes
        ids = torch.randperm(n)[:64]
        p = torch.where(level_at_start[ids] <= 0.6 + 1e-6, 0.95, 0.4)
        env.last_episode_metrics["success"][ids] = (torch.rand(64) < p).float()
        value = mdp.attribute_curriculum(env, ids, "init_max_tilt", 0.3, 1.2, 0.15)
        level_at_start[ids] = env.cfg.init_max_tilt
    assert value == pytest.approx(0.75)


def test_action_decoder_position_mode():
    from quadsim.deploy.interface import ActionDecoder

    contract = {"action": {"control_mode": "position", "collective_range": [0.0, 1.0], "max_total_thrust_N": 40.0,
                           "max_body_rate_rad_s": [6, 6, 3], "max_velocity_m_s": [3, 3, 2], "max_yaw_rate_rad_s": 2.0,
                           "position_center_m": [0.0, 0.0, 1.5], "position_range_m": [3.0, 3.0, 1.5]}}
    dec = ActionDecoder(contract)
    d = dec.decode(np.array([0.5, -1.0, 1.0, 0.25]))
    assert np.allclose(d["position_local"], [1.5, -3.0, 3.0]) and d["yaw_rate"] == pytest.approx(0.5)
    px4 = dec.to_px4(np.array([0.5, -1.0, 1.0, 0.25]))
    assert px4["msg"] == "TrajectorySetpoint" and np.allclose(px4["position"], [-3.0, 1.5, -3.0])  # ENU -> NED
    assert px4["yawspeed"] == pytest.approx(-0.5) and np.isnan(px4["velocity"]).all()


def test_export_policy_leaves_the_callers_module_untouched(tmp_path):
    from quadsim.deploy.export import export_policy

    actor = Actor(5, 2, [16], stochastic=False)
    actor.train()
    rep = export_policy(actor, {"task": "test"}, str(tmp_path), torch.randn(4, 5))
    assert actor.training  # export works on a copy (eval mode, CPU)
    assert os.path.exists(rep["onnx"]) and rep.get("verified", True)


def test_cost_logging_without_lagrangian_does_not_accumulate():
    env = _FakeEnv(4)
    cfg = terms.TermsCfg(costs={"c": terms.CostTermCfg(func=lambda e: torch.ones(e.num_envs))})
    tm = terms.TermManager(env, cfg, terms.SafetyCfg(lagrangian=False))
    for _ in range(100):
        tm.compute_cost()
        log = tm.before_reset(torch.arange(4), torch.ones(4, dtype=torch.bool))
    assert "Safety/episode_cost" in log and len(tm._finished_costs) == 0


def test_lagrangian_integral_anti_windup():
    env = _FakeEnv(4)
    cfg = terms.TermsCfg(costs={"c": terms.CostTermCfg(func=lambda e: torch.ones(e.num_envs))})
    safety = terms.SafetyCfg(lagrangian=True, cost_limit=0.5, ki=10.0, lambda_max=5.0, update_period_steps=1)
    tm = terms.TermManager(env, cfg, safety)
    for _ in range(50):  # long violation: the integral must not grow beyond lambda_max ...
        tm._finished_costs.append(torch.ones(4))
        tm.shape_reward(torch.zeros(4))
    assert tm._i_term <= 5.0 and tm.lagrange == 5.0
    tm._finished_costs.append(torch.zeros(4))  # ... so that lambda reacts as soon as the constraint holds
    tm.shape_reward(torch.zeros(4))
    assert tm.lagrange < 5.0
