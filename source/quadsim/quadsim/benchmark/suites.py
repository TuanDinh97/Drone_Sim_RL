"""Standardized benchmark suites: task x evaluation conditions x metrics, with in-distribution / OOD splits.

A suite fixes everything that must be identical for a fair comparison (task, episodes, seeds, conditions); the
runner (``scripts/benchmark/run.py``) evaluates any set of controllers (RL / BC / VLA checkpoints of any registered
algorithm, classical baselines) on every condition with ``scripts/rl/evaluate.py`` (fresh simulator per condition,
same seed for every controller) and reports

* ``id``  conditions = the training distribution (training performance),
* ``ood`` conditions = perturbations / layouts / phrasings never seen in training (generalization), and the
  generalization gap ``id - mean(ood)`` of the primary metric.

Register new suites with ``BENCHMARKS.register(name, Suite(...))`` (e.g. from a ``--plugins`` module).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quadsim.registry import BENCHMARKS


@dataclass
class Condition:
    name: str
    split: str = "id"  # id | ood
    overrides: list[str] = field(default_factory=list)
    task: str | None = None  # evaluate on another registered task (e.g. unseen track layouts)
    note: str = ""


@dataclass
class Suite:
    name: str
    task: str
    conditions: list[Condition]
    primary: str = "success"
    higher_is_better: bool = True
    metrics: list[str] = field(default_factory=lambda: ["success", "crash"])
    episodes: int = 500
    num_envs: int = 250
    seed: int = 1234
    baselines: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)
    description: str = ""


_HARSH_DYN = ["env.randomization.mass_scale=(0.8,1.25)", "env.randomization.motor_tau_scale=(1.5,2.5)",
              "env.randomization.k_f_scale=(0.8,1.0)"]
_WIND = ["env.wind.enabled=True", "env.push.enabled=True"]
_SENSOR = ["env.randomization.sensor_noise_scale=(2.0,3.0)"]
_LATENCY = ["env.control.action_delay_steps=(4,6)", "env.sensors.odometry.delay_steps_max=6"]


def _robustness(prefix=()):
    p = list(prefix)
    return [Condition("nominal", "id", p, note="training distribution"),
            Condition("wind_push", "ood", p + _WIND, note="wind 0-3 m/s + turbulence + gusts, pushes <= 0.3 mg"),
            Condition("harsh_dynamics", "ood", p + _HARSH_DYN, note="mass x0.8-1.25, motor tau x1.5-2.5, k_f x0.8-1"),
            Condition("sensor_noise", "ood", p + _SENSOR, note="estimate/IMU noise x2-3"),
            Condition("latency", "ood", p + _LATENCY, note="20-30 ms command + estimate delay")]


for _s in (
    Suite("hover", "Quadsim-Hover-v0", _robustness(), metrics=["success", "final_pos_err", "final_yaw_err", "crash"],
          episodes=1000, num_envs=500, baselines=["cascade"],
          description="recover from up to 57 deg tilt / 3 rad/s and hold a pose"),
    Suite("track", "Quadsim-Track-v0", _robustness() + [
        Condition("moderate_refs", "ood", ["env.max_speed=2.0", "env.freq_range=(0.1,0.35)"],
                  note="slower references (easier, but outside the training mix)")],
          primary="track_rmse", higher_is_better=False, metrics=["track_rmse", "track_max_err", "crash"],
          baselines=["cascade", "geometric"], description="track random smooth 3-D references (<= 4 m/s)"),
    Suite("waypoint", "Quadsim-Waypoint-v0", _robustness(), metrics=["success", "time_to_finish", "crash"],
          baselines=["cascade"], description="fly through 5 random waypoints as fast as possible"),
    Suite("nav_range", "Quadsim-Nav-Range-v0", [
        Condition("full_density", "id", ["env.curriculum=False"], note="4-24 obstacles (training end state)"),
        Condition("dense_only", "ood", ["env.curriculum=False", "env.active_obstacles_range=(20,24)"],
                  note="only dense forests"),
        Condition("wind", "ood", ["env.curriculum=False", "env.wind.enabled=True"]),
        Condition("latency", "ood", ["env.curriculum=False"] + _LATENCY)],
          metrics=["success", "collision", "time_to_goal"], baselines=["potential_field"],
          description="reach a goal through a cylinder forest with a 2-D range ring"),
    Suite("race", "Quadsim-Race-v0", [
        Condition("random_tracks", "id", note="random elliptic circuits (training distribution)"),
        Condition("ellipse", "ood", task="Quadsim-Race-Ellipse-v0", note="fixed reference circuit"),
        Condition("figure8", "ood", task="Quadsim-Race-Figure8-v0", note="unseen layout family with a crossing"),
        Condition("wind", "ood", ["env.wind.enabled=True"])],
          metrics=["success", "lap_time", "gates_passed", "crash"], baselines=["cascade"],
          description="2 laps through 7 gates on a 5-inch racer"),
    Suite("swarm", "Quadsim-Swarm-v0", [
        Condition("swap4", "id", note="4-agent antipodal swap"),
        Condition("wind", "ood", ["env.wind.enabled=True"]),
        Condition("latency", "ood", _LATENCY)],
          metrics=["success", "agent_collision", "min_separation", "time_to_goal"], num_envs=256,
          baselines=["cascade", "swarm_potential"], description="decentralized multi-agent position swap"),
    Suite("langnav", "Quadsim-LangNav-v0", [
        Condition("train_phrasing", "id"),
        Condition("heldout_phrasing", "ood", task="Quadsim-LangNav-Heldout-v0", note="unseen instruction templates"),
        Condition("wind", "ood", ["env.wind.enabled=True"])],
          metrics=["success", "wrong_landmark", "pillar_collision", "time_to_success"], baselines=["langnav_expert"],
          description="fly to the pillar named in a natural-language instruction (symbolic map)"),
):
    BENCHMARKS.register(_s.name, _s, overwrite=True)
