"""Aggregate statistics for multi-seed results (rliable definitions), checked against scipy references."""

import numpy as np
import pytest

sps = pytest.importorskip("scipy.stats")  # reference implementations

from quadsim.benchmark import stats


def test_iqm_matches_scipy_trim_mean():
    rng = np.random.default_rng(1)
    for shape in [(3,), (3, 4), (10, 7), (5, 1)]:
        x = rng.normal(size=shape)
        assert stats.iqm(x) == pytest.approx(sps.trim_mean(x.ravel(), 0.25))


def test_stratified_bootstrap_ci_properties():
    const = np.full((3, 4), 0.7)
    assert stats.stratified_bootstrap_ci(const, reps=200) == pytest.approx((0.7, 0.7, 0.7))
    s = np.array([[0.54, 0.98], [0.43, 0.97], [0.68, 0.97]])
    point, lo, hi = stats.stratified_bootstrap_ci(s, stats.mean, reps=2000, seed=3)
    assert lo <= point <= hi and point == pytest.approx(s.mean())
    assert stats.stratified_bootstrap_ci(s, stats.mean, reps=2000, seed=3) == (point, lo, hi)  # deterministic
    # the interval only contains means of resamples of the observed runs, per task
    assert lo >= (0.43 + 0.97) / 2 - 1e-9 and hi <= (0.68 + 0.98) / 2 + 1e-9


def test_probability_of_improvement_matches_mann_whitney():
    rng = np.random.default_rng(2)
    x, y = rng.normal(0.3, 1, size=(6, 1)), rng.normal(0.0, 1, size=(5, 1))
    p, lo, hi = stats.probability_of_improvement(x, y, reps=500)
    u = sps.mannwhitneyu(x[:, 0], y[:, 0]).statistic
    assert p == pytest.approx(u / (6 * 5)) and lo <= p <= hi
    assert stats.probability_of_improvement(np.ones((3, 2)), np.zeros((3, 2)), reps=50)[0] == 1.0
    assert stats.probability_of_improvement(np.ones((3, 2)), np.ones((3, 2)), reps=50)[0] == 0.5  # ties count 1/2
    with pytest.raises(ValueError):
        stats.probability_of_improvement(np.ones((3, 2)), np.ones((3, 3)))


def test_summarize_runs_and_optimality_gap():
    out = stats.summarize_runs([0.738, 0.666, 0.792], reps=500)
    assert out["runs"] == 3 and out["tasks"] == 1 and out["mean"] == pytest.approx(0.732)
    assert out["mean_ci95"][0] <= out["mean"] <= out["mean_ci95"][1]
    assert stats.optimality_gap([[0.9, 1.0], [0.5, 1.2]]) == pytest.approx((0.1 + 0.5) / 4)
