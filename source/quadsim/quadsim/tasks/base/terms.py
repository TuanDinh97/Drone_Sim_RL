"""Configuration-driven MDP terms that extend any quadsim task without subclassing it.

Every task (``QuadrotorEnvCfg.terms``) accepts extra terms in the spirit of Isaac Lab's manager-based environments
(same ``func(env, **params)`` signature, so Isaac Lab-style term functions can be reused)::

    cfg.terms.observations["down_range"] = ObsTermCfg(func=mdp.rangefinder_down, params={"max_range": 8.0},
                                                      noise_std=0.02, period_steps=2, delay_steps=1)
    cfg.terms.rewards["energy"] = RewardTermCfg(func=mdp.rotor_power, weight=-1e-4)
    cfg.terms.terminations["ceiling"] = TerminationTermCfg(func=mdp.above_height, params={"height": 3.0})
    cfg.terms.costs["speed"] = CostTermCfg(func=mdp.speed_violation, params={"limit": 2.0})
    cfg.terms.events["bump"] = EventTermCfg(func=mdp.random_velocity_kick, mode="interval", interval_steps=100)
    cfg.terms.curricula["tilt"] = CurriculumTermCfg(func=mdp.attribute_curriculum, params={"attr": "init_max_tilt", ...})

Conventions (identical to the built-in task terms):
* policy-step rewards are ``weight * func(env) * step_dt`` (Isaac Lab convention); costs likewise
  ``weight * func(env) * step_dt``, so an indicator cost integrates to *seconds of violation per episode*;
* observation terms are deployable (appended to the ``policy`` group) unless ``privileged=True`` (``critic`` only).
  Deployable terms must only use onboard information (estimates, sensors, task data) - the framework cannot check
  this for user functions;
* termination terms with ``time_out=True`` truncate (the value is bootstrapped), otherwise they are failures.

Safe RL (:class:`SafetyCfg`): with ``lagrangian=True`` the environment returns the Lagrangian reward
``(r - lambda c) / (1 + lambda)`` and adapts ``lambda >= 0`` with a PID rule on the mean episode cost of finished
episodes vs ``cost_limit`` (PID Lagrangian, Stooke et al. 2020). It works with any RL algorithm because it only
changes the reward the algorithm sees; the raw reward and cost stay available in ``extras``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, dataclass, field

import torch


@dataclass
class ObsTermCfg:
    func: Callable = MISSING
    """``func(env, **params) -> (N, d)`` tensor."""
    params: dict = field(default_factory=dict)
    privileged: bool = False
    """False: appended to the deployable ``policy`` observation (and the critic's); True: critic only."""
    scale: float = 1.0
    noise_std: float = 0.0
    """Additive white Gaussian noise (std, after scaling)."""
    clip: tuple[float, float] | None = None
    period_steps: int = 1
    """Sample-and-hold: the term is refreshed every ``period_steps`` policy steps (sensor rate)."""
    delay_steps: int = 0
    """Latency in policy steps."""
    history: int = 1
    """Number of most recent (delayed) samples concatenated, oldest first (e.g. action/gyro histories for implicit
    system identification / adaptive policies)."""


@dataclass
class RewardTermCfg:
    func: Callable = MISSING
    """``func(env, **params) -> (N,)``; the reward is ``weight * func * step_dt``."""
    params: dict = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class TerminationTermCfg:
    func: Callable = MISSING
    """``func(env, **params) -> (N,) bool``."""
    params: dict = field(default_factory=dict)
    time_out: bool = False


@dataclass
class CostTermCfg:
    func: Callable = MISSING
    """``func(env, **params) -> (N,)`` non-negative cost rate; the step cost is ``weight * func * step_dt``."""
    params: dict = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class EventTermCfg:
    func: Callable = MISSING
    """``func(env, env_ids, **params)``: randomization / perturbation applied to ``env_ids``."""
    params: dict = field(default_factory=dict)
    mode: str = "reset"
    """``startup`` (once, all envs) | ``reset`` (every episode start) | ``interval`` (every ``interval_steps``)."""
    interval_steps: int = 50


@dataclass
class CurriculumTermCfg:
    func: Callable = MISSING
    """``func(env, env_ids, **params) -> float | None``, called before ``env_ids`` (the environments whose episode just
    finished) are reset; their metrics are in ``env.last_episode_metrics[...][env_ids]``. The returned value is
    logged as ``Curriculum/<name>``."""
    params: dict = field(default_factory=dict)


@dataclass
class TermsCfg:
    observations: dict = field(default_factory=dict)
    rewards: dict = field(default_factory=dict)
    terminations: dict = field(default_factory=dict)
    costs: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    curricula: dict = field(default_factory=dict)


@dataclass
class SafetyCfg:
    lagrangian: bool = False
    cost_limit: float = 0.0
    """Budget for the mean episode cost (sum of step costs, i.e. seconds of violation for indicator costs)."""
    lambda_init: float = 0.0
    kp: float = 0.0
    ki: float = 0.05
    kd: float = 0.0
    lambda_max: float = 100.0
    update_period_steps: int = 32
    """Policy steps between multiplier updates (one rsl_rl rollout by default)."""
    warmup_steps: int = 0
    """Policy steps before the multiplier starts to move (it stays at ``lambda_init`` and the costs of that period
    are discarded): lets the policy learn the task before the constraint is enforced."""
    cost_limit_start: float | None = None
    """Constraint curriculum: after the warm-up the budget starts here and decreases geometrically to ``cost_limit``
    over ``anneal_steps`` policy steps (None: constant budget)."""
    anneal_steps: int = 0
    normalize: bool = True
    """Divide the Lagrangian reward by (1 + lambda) to keep its scale bounded."""


class _ObsTerm:
    def __init__(self, name, cfg: ObsTermCfg, env):
        self.name, self.cfg = name, cfg
        v = self._eval(env)
        self.base_dim = v.shape[-1]
        self.dim = self.base_dim * max(cfg.history, 1)
        self.hist = torch.zeros(cfg.delay_steps + max(cfg.history, 1), env.num_envs, self.base_dim, device=env.device)
        self.hold = torch.zeros(env.num_envs, self.base_dim, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.counter = 0

    def _eval(self, env):
        v = self.cfg.func(env, **self.cfg.params)
        if v.dim() == 1:
            v = v[:, None]
        return v * self.cfg.scale

    def compute(self, env) -> torch.Tensor:
        c = self.cfg
        sample = self.counter % max(c.period_steps, 1) == 0
        self.counter += 1
        if sample or self.fresh.any():
            v = self._eval(env)
            if c.noise_std > 0:
                v = v + c.noise_std * torch.randn_like(v)
            if c.clip is not None:
                v = v.clamp(*c.clip)
            self.hold = v if sample else torch.where(self.fresh[:, None], v, self.hold)
        if len(self.hist) > 1:
            self.hist = torch.roll(self.hist, 1, 0)
        self.hist[0] = self.hold
        if self.fresh.any():
            self.hist[:, self.fresh] = self.hold[self.fresh]
            self.fresh[:] = False
        h = max(c.history, 1)
        if h == 1:
            return self.hist[-1]
        # the h samples ending ``delay_steps`` steps ago, oldest first
        return self.hist[c.delay_steps: c.delay_steps + h].flip(0).permute(1, 0, 2).reshape(self.hist.shape[1], -1)


class TermManager:
    """Evaluates the configured extra terms of a :class:`~quadsim.tasks.base.QuadrotorEnv`."""

    def __init__(self, env, cfg: TermsCfg, safety: SafetyCfg):
        self.env, self.cfg, self.safety = env, cfg, safety
        n, dev = env.num_envs, env.device
        obs = [_ObsTerm(k, c, env) for k, c in cfg.observations.items()]
        self.policy_obs = [o for o in obs if not o.cfg.privileged]
        self.critic_obs = [o for o in obs if o.cfg.privileged]
        self.policy_dim = sum(o.dim for o in self.policy_obs)
        self.critic_dim = sum(o.dim for o in self.critic_obs)
        self.has_costs = len(cfg.costs) > 0
        self.cost = torch.zeros(n, device=dev)
        self.episode_cost = torch.zeros(n, device=dev)
        self.cost_sums: dict[str, torch.Tensor] = {k: torch.zeros(n, device=dev) for k in cfg.costs}
        self.lagrange = float(safety.lambda_init)
        self._i_term = float(safety.lambda_init)
        self._prev_cost = 0.0
        self._finished_costs: list[torch.Tensor] = []
        self._steps = 0
        self.step_counter = 0
        for k, e in cfg.events.items():
            if e.mode == "startup":
                e.func(env, torch.arange(n, device=dev), **e.params)

    # ---------------------------------------------------------------- observations
    def policy_terms(self) -> list[torch.Tensor]:
        return [o.compute(self.env) for o in self.policy_obs]

    def critic_terms(self) -> list[torch.Tensor]:
        return [o.compute(self.env) for o in self.critic_obs]

    # ---------------------------------------------------------------- rewards / costs
    def rewards(self) -> dict[str, torch.Tensor]:
        dt = self.env.step_dt
        return {k: c.weight * c.func(self.env, **c.params) * dt for k, c in self.cfg.rewards.items()}

    def compute_cost(self) -> torch.Tensor:
        dt = self.env.step_dt
        self.cost = torch.zeros_like(self.cost)
        for k, c in self.cfg.costs.items():
            v = c.weight * c.func(self.env, **c.params) * dt
            self.cost_sums[k] += v
            self.cost += v
        self.episode_cost += self.cost
        return self.cost

    def current_cost_limit(self) -> float:
        """Budget of the constraint at the current step (constant unless a curriculum is configured)."""
        s = self.safety
        if s.cost_limit_start is None or s.anneal_steps <= 0:
            return s.cost_limit
        f = min(max((self._steps - s.warmup_steps) / s.anneal_steps, 0.0), 1.0)
        return s.cost_limit_start * (s.cost_limit / s.cost_limit_start) ** f

    def shape_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Lagrangian reward (safe RL); also performs the periodic multiplier update."""
        s = self.safety
        self._steps += 1
        if self._steps <= s.warmup_steps:
            self._finished_costs.clear()
        elif self._steps % max(s.update_period_steps, 1) == 0 and self._finished_costs:
            jc = torch.cat(self._finished_costs).mean().item()
            self._finished_costs.clear()
            err = jc - self.current_cost_limit()
            self._i_term = min(max(0.0, self._i_term + s.ki * err), s.lambda_max)  # anti-windup
            d = max(0.0, jc - self._prev_cost)
            self._prev_cost = jc
            self.lagrange = min(max(0.0, s.kp * err + self._i_term + s.kd * d), s.lambda_max)
        r = reward - self.lagrange * self.cost
        return r / (1.0 + self.lagrange) if s.normalize else r

    # ---------------------------------------------------------------- terminations
    def terminations(self) -> tuple[torch.Tensor, torch.Tensor]:
        n, dev = self.env.num_envs, self.env.device
        fail = torch.zeros(n, dtype=torch.bool, device=dev)
        trunc = torch.zeros(n, dtype=torch.bool, device=dev)
        for c in self.cfg.terminations.values():
            v = c.func(self.env, **c.params).bool()
            if c.time_out:
                trunc |= v
            else:
                fail |= v
        return fail, trunc

    # ---------------------------------------------------------------- events / curricula / reset
    def interval_events(self):
        self.step_counter += 1
        for e in self.cfg.events.values():
            if e.mode == "interval" and self.step_counter % max(e.interval_steps, 1) == 0:
                e.func(self.env, torch.arange(self.env.num_envs, device=self.env.device), **e.params)

    def before_reset(self, env_ids, played) -> dict:
        """Curricula (see finished-episode metrics) and episode-cost logging; returns log entries."""
        log = {}
        for k, c in self.cfg.curricula.items():
            v = c.func(self.env, env_ids[played], **c.params)
            if v is not None:
                log[f"Curriculum/{k}"] = float(v)
        if self.has_costs:
            done = self.episode_cost[env_ids][played]
            if len(done):
                if self.safety.lagrangian:  # consumed (and cleared) by the multiplier update
                    self._finished_costs.append(done.clone())
                log["Safety/episode_cost"] = done.mean().item()
            log["Safety/lambda"] = self.lagrange
            log["Safety/cost_limit"] = self.current_cost_limit()
            for k, v in self.cost_sums.items():
                log[f"Episode_Cost/{k}"] = v[env_ids].mean().item()
                v[env_ids] = 0.0
        return log

    def after_reset(self, env_ids):
        self.episode_cost[env_ids] = 0.0
        for o in self.policy_obs + self.critic_obs:
            o.fresh[env_ids] = True
        for e in self.cfg.events.values():
            if e.mode == "reset":
                e.func(self.env, env_ids, **e.params)
