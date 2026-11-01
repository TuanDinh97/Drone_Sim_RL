"""Loading trained rsl_rl policies for evaluation, recording and export."""

from __future__ import annotations

import os

import torch
from tensordict import TensorDict


def agent_cfg_for(task: str):
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    return load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")


def build_runner(env, agent_cfg, log_dir=None, device="cuda:0"):
    """OnPolicyRunner around an (unwrapped or gym-wrapped) quadsim env."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from .networks import register_rsl_rl_modules

    register_rsl_rl_modules()
    vec = env if isinstance(env, RslRlVecEnvWrapper) else RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    return OnPolicyRunner(vec, agent_cfg.to_dict(), log_dir=log_dir, device=device), vec


def load_policy(checkpoint: str, env, task: str | None = None, agent_cfg=None):
    """Returns ``policy(obs_dict) -> actions`` (deterministic mean action, normalizers included)."""
    if agent_cfg is None:
        task = task or getattr(getattr(env, "spec", None), "id", None)
        agent_cfg = agent_cfg_for(task)
    runner, _ = build_runner(env, agent_cfg, None, str(env.unwrapped.device))
    pol = runner.alg.policy
    sd = torch.load(checkpoint, map_location=str(env.unwrapped.device), weights_only=False)["model_state_dict"]
    own = pol.state_dict()
    critic_mismatch = any(k.startswith("critic") and k in own and own[k].shape != v.shape for k, v in sd.items())
    if critic_mismatch:
        # evaluating in an env whose privileged (critic) observation differs from training, e.g. a v0 policy in a v1
        # env: only the actor (+ its normalizer) is needed for inference
        actor = {k: v for k, v in sd.items() if not k.startswith("critic")}
        missing = [k for k in own if not k.startswith("critic") and k not in actor]
        if missing:
            raise RuntimeError(f"actor parameters missing in {checkpoint}: {missing}")
        pol.load_state_dict(actor, strict=False)
        print(f"[policy_io] critic shape differs from {checkpoint}: loaded the actor only", flush=True)
    else:
        runner.load(checkpoint, load_optimizer=False)
    pol.eval()

    def act(obs: dict) -> torch.Tensor:
        td = obs if isinstance(obs, TensorDict) else TensorDict(obs, batch_size=[next(iter(obs.values())).shape[0]])
        return pol.act_inference(td)

    act.policy = pol
    act.runner = runner
    return act


def latest_checkpoint(run_dir: str) -> str:
    pts = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not pts:
        raise FileNotFoundError(f"no model_*.pt in {run_dir}")
    return os.path.join(run_dir, max(pts, key=lambda f: int(f[6:-3])))
