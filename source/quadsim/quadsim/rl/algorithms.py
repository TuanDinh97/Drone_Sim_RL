"""Training back-ends behind one interface (``scripts/rl/train.py --algo <name>``).

An algorithm is registered in :data:`quadsim.registry.ALGORITHMS` as an :class:`Algorithm` subclass:

* ``default_cfg(task)``  configuration object (a configclass, so ``agent.<field>=value`` overrides work);
* ``train(env, cfg, log_dir, resume)``  runs the training loop and writes checkpoints to ``log_dir``;
* ``load_policy(checkpoint, env, task)``  returns ``act(obs_dict) -> actions`` (deterministic, for evaluation);
* ``actor_module(act)``  the exportable network behind a loaded policy (ONNX export).

Built-in: ``ppo`` (rsl_rl 3.1.2, asymmetric actor-critic, CNN encoder for image tasks), ``sac`` and ``td3``
(:mod:`quadsim.rl.offpolicy`, asymmetric twin critics). ``examples/custom_algorithm.py`` registers a new one.
"""

from __future__ import annotations

import os

import torch

from quadsim.registry import ALGORITHMS


class Algorithm:
    name = ""
    log_root = "logs"

    def default_cfg(self, task: str):
        raise NotImplementedError

    def train(self, env, cfg, log_dir: str, resume: str | None = None):
        raise NotImplementedError

    def load_policy(self, checkpoint: str, env, task: str | None = None):
        raise NotImplementedError

    def actor_module(self, act):
        return getattr(act, "policy", None)

    def log_dir_root(self, cfg) -> str:
        return os.path.join(self.log_root, self.name, cfg.experiment_name)


@ALGORITHMS.register("ppo")
class RslRlPPO(Algorithm):
    name = "ppo"

    def log_dir_root(self, cfg):
        return os.path.join("logs", "rsl_rl", cfg.experiment_name)  # unchanged layout of the validated runs

    def default_cfg(self, task):
        from .policy_io import agent_cfg_for

        return agent_cfg_for(task)

    def train(self, env, cfg, log_dir, resume=None):
        from .policy_io import build_runner, latest_checkpoint

        runner, _ = build_runner(env, cfg, log_dir, cfg.device)
        runner.add_git_repo_to_log(__file__)
        if resume:
            ckpt = resume if resume.endswith(".pt") else latest_checkpoint(resume)
            print(f"[train] resuming from {ckpt}")
            runner.load(ckpt)
        runner.learn(num_learning_iterations=cfg.max_iterations, init_at_random_ep_len=True)

    def load_policy(self, checkpoint, env, task=None):
        from .policy_io import load_policy

        return load_policy(checkpoint, env, task=task)


class _OffPolicy(Algorithm):
    def default_cfg(self, task):
        from .offpolicy import OffPolicyCfg
        from .policy_io import agent_cfg_for

        cfg = OffPolicyCfg(algo=self.name)
        try:
            cfg.experiment_name = agent_cfg_for(task).experiment_name
        except Exception:
            cfg.experiment_name = task
        return cfg

    def train(self, env, cfg, log_dir, resume=None):
        from .offpolicy import OffPolicyLearner
        from .policy_io import latest_checkpoint

        torch.manual_seed(cfg.seed)
        learner = OffPolicyLearner(env, cfg, log_dir)
        if resume:
            ckpt = resume if resume.endswith(".pt") else latest_checkpoint(resume)
            print(f"[train] resuming from {ckpt}")
            learner.load(ckpt)
        learner.learn(cfg.max_iterations - learner.step if resume else cfg.max_iterations)

    def load_policy(self, checkpoint, env, task=None):
        from .offpolicy import load_actor

        actor = load_actor(checkpoint, str(env.unwrapped.device))

        def act(obs):
            return actor(obs["policy"])

        act.policy = actor
        return act


@ALGORITHMS.register("sac")
class SAC(_OffPolicy):
    name = "sac"


@ALGORITHMS.register("td3")
class TD3(_OffPolicy):
    name = "td3"


def algorithm_of_checkpoint(checkpoint: str) -> str:
    """Infer the algorithm that wrote a checkpoint (``model_*.pt``)."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "algo" in ck:
        return ck["algo"]
    if isinstance(ck, dict) and "model_state_dict" in ck:
        return "ppo"
    raise ValueError(f"cannot infer the algorithm of {checkpoint}")


def load_any_policy(checkpoint: str, env, task: str | None = None, algo: str | None = None):
    """Policy from any registered algorithm's checkpoint (run directory -> latest ``model_*.pt``)."""
    from .policy_io import latest_checkpoint

    ckpt = latest_checkpoint(checkpoint) if os.path.isdir(checkpoint) else checkpoint
    algo = algo or algorithm_of_checkpoint(ckpt)
    return ALGORITHMS.get(algo)().load_policy(ckpt, env, task=task)


def _register_optional():
    """Algorithms living in other sub-packages (imported lazily to keep ``quadsim.rl`` light)."""
    from quadsim.il import bc, vla  # noqa: F401  (registers "bc", "vla")


_register_optional()
