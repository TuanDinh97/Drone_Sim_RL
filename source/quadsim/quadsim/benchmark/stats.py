"""Aggregate metrics with interval estimates for multi-run (multi-seed) results.

Follows Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice" (NeurIPS 2021) and uses
the same definitions as their ``rliable`` library, implemented with numpy only:

* ``scores`` is an array (runs, tasks) of per-run scores - e.g. the success rate of every seed (rows) on every
  evaluation condition (columns);
* :func:`iqm` - interquartile mean of all run x task scores (mean of the middle 50 %, robust to outliers, more
  statistically efficient than the median);
* :func:`stratified_bootstrap_ci` - percentile confidence interval of any aggregate, resampling runs with replacement
  independently for every task;
* :func:`probability_of_improvement` - P(X > Y) for a random run of algorithm X vs a random run of Y on a random task
  (Mann-Whitney statistic averaged over tasks; ties count 1/2), with a stratified bootstrap CI;
* :func:`optimality_gap` - mean shortfall below a target score.

With few runs (3 seeds) the intervals are wide - that is the point: they show how much a result could move with
other seeds. Report them next to the per-seed values.
"""

from __future__ import annotations

import numpy as np


def _as_2d(scores) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    return s[:, None] if s.ndim == 1 else s


def mean(scores) -> float:
    return float(np.mean(_as_2d(scores)))


def median(scores) -> float:
    return float(np.median(_as_2d(scores)))


def iqm(scores) -> float:
    """Interquartile mean (``scipy.stats.trim_mean(x, 0.25)`` of the flattened scores)."""
    x = np.sort(_as_2d(scores).ravel())
    k = int(0.25 * x.size)
    return float(x[k: x.size - k].mean())


def optimality_gap(scores, target: float = 1.0) -> float:
    return float(np.mean(np.maximum(target - _as_2d(scores), 0.0)))


def stratified_bootstrap_ci(scores, aggregate=iqm, reps: int = 5000, alpha: float = 0.05, seed: int = 0):
    """(point estimate, lower, upper) of ``aggregate`` with a (1 - alpha) percentile interval; runs are resampled with
    replacement independently for every task (column)."""
    s = _as_2d(scores)
    rng = np.random.default_rng(seed)
    n, t = s.shape
    idx = rng.integers(0, n, size=(reps, n, t))
    boot = np.array([aggregate(np.take_along_axis(s, idx[r], axis=0)) for r in range(reps)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(aggregate(s)), float(lo), float(hi)


def _poi(x: np.ndarray, y: np.ndarray) -> float:
    per_task = []
    for j in range(x.shape[1]):
        a, b = x[:, j][:, None], y[:, j][None, :]
        per_task.append(np.mean((a > b) + 0.5 * (a == b)))
    return float(np.mean(per_task))


def probability_of_improvement(x, y, reps: int = 5000, alpha: float = 0.05, seed: int = 0):
    """(P(X > Y), lower, upper): ``x`` (runs_x, tasks) and ``y`` (runs_y, tasks) on the same tasks."""
    x, y = _as_2d(x), _as_2d(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must cover the same tasks")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(reps):
        xi = np.take_along_axis(x, rng.integers(0, x.shape[0], size=x.shape), axis=0)
        yi = np.take_along_axis(y, rng.integers(0, y.shape[0], size=y.shape), axis=0)
        boot.append(_poi(xi, yi))
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return _poi(x, y), float(lo), float(hi)


def summarize_runs(scores, reps: int = 5000, seed: int = 0) -> dict:
    """Per-run values plus mean / IQM with stratified bootstrap 95 % CIs (``scores`` (runs, tasks) or (runs,))."""
    s = _as_2d(scores)
    m = stratified_bootstrap_ci(s, mean, reps=reps, seed=seed)
    q = stratified_bootstrap_ci(s, iqm, reps=reps, seed=seed)
    return {"runs": int(s.shape[0]), "tasks": int(s.shape[1]), "mean": m[0], "mean_ci95": [m[1], m[2]],
            "iqm": q[0], "iqm_ci95": [q[1], q[2]]}
