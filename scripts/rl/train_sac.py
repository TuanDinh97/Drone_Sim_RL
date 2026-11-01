"""Off-policy training with Soft Actor-Critic (skrl 1.4.3) on any quadsim state-based task.

Example: python scripts/rl/train_sac.py --headless --task Quadsim-Hover-v0 --num_envs 256 --timesteps 20000
Writes logs/skrl/<task>_sac/<timestamp>/ (checkpoints + TensorBoard) and an evaluation JSON
(deterministic policy, same per-episode metrics as scripts/rl/evaluate.py) in that directory.
"""

import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Quadsim-Hover-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--timesteps", type=int, default=20000, help="vectorized env steps")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval_episodes", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG  # noqa: E402
from skrl.envs.wrappers.torch import wrap_env  # noqa: E402
from skrl.memories.torch import RandomMemory  # noqa: E402
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model  # noqa: E402
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa: E402
from skrl.trainers.torch import SequentialTrainer  # noqa: E402
from skrl.utils import set_seed  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.rl.overrides import apply_overrides  # noqa: E402
from quadsim.rl.policy_io import agent_cfg_for  # noqa: E402


class Actor(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-5.0, max_log_std=1.0)
        self.net = nn.Sequential(nn.Linear(self.num_observations, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU())
        self.mean = nn.Linear(128, self.num_actions)
        self.log_std = nn.Linear(128, self.num_actions)

    def compute(self, inputs, role):
        h = self.net(inputs["states"])
        return torch.tanh(self.mean(h)), self.log_std(h), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512), nn.ELU(),
                                 nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)), {}


def main():
    set_seed(args.seed)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    apply_overrides(cfg, agent_cfg_for(args.task), overrides)
    cfg.seed = args.seed
    genv = gym.make(args.task, cfg=cfg)
    env = wrap_env(genv, wrapper="isaaclab")
    dev = env.device
    log_dir = os.path.join("logs", "skrl", f"{args.task}_sac", time.strftime("%Y-%m-%d_%H-%M-%S"))
    models = {"policy": Actor(env.observation_space, env.action_space, dev)}
    for k in ("critic_1", "critic_2", "target_critic_1", "target_critic_2"):
        models[k] = Critic(env.observation_space, env.action_space, dev)
    sac_cfg = SAC_DEFAULT_CONFIG.copy()
    sac_cfg.update({
        "gradient_steps": 1, "batch_size": 4096, "discount_factor": 0.99, "polyak": 0.005,
        "actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4, "learning_starts": 200,
        "learn_entropy": True, "initial_entropy_value": 0.1,
        "state_preprocessor": RunningStandardScaler, "state_preprocessor_kwargs": {"size": env.observation_space, "device": dev},
    })
    sac_cfg["experiment"].update({"directory": log_dir, "experiment_name": "", "write_interval": 200,
                                  "checkpoint_interval": 5000})
    memory = RandomMemory(memory_size=2000, num_envs=env.num_envs, device=dev)
    agent = SAC(models=models, memory=memory, cfg=sac_cfg, observation_space=env.observation_space,
                action_space=env.action_space, device=dev)
    t0 = time.time()
    SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": True, "disable_progressbar": True},
                      env=env, agents=agent).train()
    train_s = time.time() - t0

    # ---- deterministic evaluation on the env's per-episode metrics
    u = genv.unwrapped
    u.seed(args.seed + 1)
    obs, _ = genv.reset()
    u.episode_metric_buffer.clear()
    u.collect_episode_metrics = True
    agent.set_running_mode("eval")
    steps = 0
    max_steps = int(math.ceil(args.eval_episodes / u.num_envs) + 2) * int(u.max_episode_length)
    while sum(len(r["env_id"]) for r in u.episode_metric_buffer) < args.eval_episodes and steps < max_steps:
        with torch.no_grad():
            s = agent._state_preprocessor(obs["policy"])
            mean, _, _ = agent.policy.compute({"states": s}, role="policy")
        obs, *_ = genv.step(mean)
        steps += 1
    u.collect_episode_metrics = False
    recs = u.episode_metric_buffer
    res = {"task": args.task, "algorithm": "SAC (skrl 1.4.3)", "num_envs": args.num_envs, "timesteps": args.timesteps,
           "env_transitions": args.timesteps * args.num_envs, "train_wall_s": train_s, "log_dir": log_dir}
    for k in recs[0]:
        if k != "env_id":
            v = torch.cat([r[k] for r in recs])
            v = v[torch.isfinite(v)]
            res[k] = float(v.mean()) if len(v) else float("nan")
    res["episodes"] = int(sum(len(r["env_id"]) for r in recs))
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("[SAC] " + json.dumps(res, indent=2), flush=True)
    genv.close()


if __name__ == "__main__":
    main()
    app.close()
