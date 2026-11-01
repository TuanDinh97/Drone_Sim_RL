import math

import numpy as np

from quadsim.deploy.closed_loop_metrics import estimate_delay, fit_fopdt, hover_rms, step_metrics


def test_hover_rms_known_errors():
    t = np.linspace(0, 10, 1001)
    goal = np.array([1.0, 2.0, 3.0])
    pos = goal + np.stack([0.03 * np.sin(2 * np.pi * t), np.zeros_like(t), np.full_like(t, -0.04)], -1)
    r = hover_rms(t, pos, goal, 0.0, 10.0)
    assert abs(r["rms_xy"] - 0.03 / math.sqrt(2)) < 1e-4
    assert abs(r["rms_z"] - 0.04) < 1e-12
    assert abs(r["rms_3d"] - math.sqrt(0.03 ** 2 / 2 + 0.04 ** 2)) < 1e-4
    assert np.allclose(r["bias"], [0.0, 0.0, -0.04], atol=1e-4)


def test_step_metrics_second_order():
    # under-damped 2nd-order step, zeta = 0.5, wn = 4 rad/s: overshoot = exp(-pi z / sqrt(1-z^2)) = 16.3 %
    z, wn = 0.5, 4.0
    t = np.linspace(0, 10, 20001)
    wd = wn * math.sqrt(1 - z * z)
    s = np.clip(t - 2, 0, None)
    y = np.where(t < 2, 0.0, 1 - np.exp(-z * wn * s) * (np.cos(wd * s) + z / math.sqrt(1 - z * z) * np.sin(wd * s)))
    r = step_metrics(t, 5.0 + y, t_step=2.0, y0=5.0, y1=6.0, t_end=10.0, band=0.02)
    assert abs(r["overshoot_pct"] - 100 * math.exp(-math.pi * z / math.sqrt(1 - z * z))) < 0.05
    assert 0.25 < r["rise_10_90_s"] < 0.45  # ~1.64/wn for zeta=0.5 is the 0-100 % time; 10-90 % ~ 0.34 s
    assert abs(r["settling_s"] - 4 / (z * wn)) < 0.4  # 2 % settling ~ 4 / (zeta wn)
    assert abs(r["ss_err"]) < 1e-3
    # downward step is handled symmetrically
    r2 = step_metrics(t, 5.0 - y, t_step=2.0, y0=5.0, y1=4.0, t_end=10.0, band=0.02)
    assert abs(r2["overshoot_pct"] - r["overshoot_pct"]) < 1e-9 and abs(r2["settling_s"] - r["settling_s"]) < 1e-9


def test_step_metrics_never_settles():
    t = np.linspace(0, 5, 501)
    y = np.where(t < 1, 0.0, 0.5)
    r = step_metrics(t, y, 1.0, 0.0, 1.0, 5.0, band=0.05)
    assert math.isnan(r["settling_s"]) and math.isnan(r["rise_10_90_s"]) and abs(r["ss_err"] + 0.5) < 1e-12


def test_estimate_delay_recovers_shift():
    rng = np.random.default_rng(0)
    t = np.arange(0, 20, 0.004)
    x = np.convolve(rng.normal(size=len(t)), np.ones(50) / 50, mode="same")  # band-limited random signal
    t_del = np.arange(0.1, 19.9, 0.01)  # delayed copy sampled at a different rate
    x_del = np.interp(t_del - 0.023, t, x)
    r = estimate_delay(t, x, t_del, x_del, max_lag=0.1)
    assert abs(r["lag_s"] - 0.023) <= 1e-3 and r["rmse_at_lag"] < r["rmse_at_zero"]


def test_fit_fopdt_recovers_parameters():
    rng = np.random.default_rng(1)
    dt_sim = 0.0005
    t = np.arange(0, 30, dt_sim)
    # 50 Hz zero-order-held random set-points (like a policy's rate commands)
    t_u = np.arange(0, 30, 0.02)
    u = np.repeat(rng.normal(size=len(t_u)), 1)
    u_fine = u[np.clip(np.searchsorted(t_u, t, side="right") - 1, 0, len(u) - 1)]
    tau, delay, gain = 0.06, 0.018, 0.9
    y = np.zeros_like(t)
    a = math.exp(-dt_sim / tau)
    n_d = int(round(delay / dt_sim))
    for k in range(len(t) - 1):
        y[k + 1] = a * y[k] + (1 - a) * gain * (u_fine[k - n_d] if k >= n_d else 0.0)
    t_y = t[::8]  # measured at 250 Hz with noise
    y_meas = y[::8] + rng.normal(scale=0.02, size=len(t_y))
    r = fit_fopdt(t_u, u, t_y, y_meas, dt=1e-3)
    assert abs(r["delay_s"] - delay) <= 2e-3
    assert abs(r["tau_s"] - tau) / tau < 0.1
    assert abs(r["gain"] - gain) < 0.03
    assert r["vaf"] > 0.95


def test_roll_pitch_and_scenario_metrics_on_ideal_response():
    from quadsim.deploy.closed_loop_metrics import roll_pitch, scenario_metrics

    r, p = 0.1, -0.2  # q = Rz(0) Ry(p) Rx(r)
    q = np.array([math.cos(r / 2) * math.cos(p / 2), math.sin(r / 2) * math.cos(p / 2),
                  math.cos(r / 2) * math.sin(p / 2), -math.sin(r / 2) * math.sin(p / 2)])
    assert np.allclose(roll_pitch(q), [r, p], atol=1e-12)
    # first-order position response to the scenario schedule, estimate = truth delayed by 10 ms
    t = np.arange(0, 60, 0.005)
    g0 = np.array([0.0, 0.0, 2.5])
    goal = g0 + np.stack([(t >= 30).astype(float), 0 * t, (t >= 45).astype(float)], -1)
    pos = np.zeros_like(goal)
    pos[0] = g0
    a = math.exp(-0.005 / 0.3)
    for k in range(len(t) - 1):
        pos[k + 1] = a * pos[k] + (1 - a) * goal[k]
    roll = 0.05 * np.sin(2 * np.pi * 0.7 * t)
    qs = np.stack([np.cos(roll / 2), np.sin(roll / 2), 0 * t, 0 * t], -1)
    rate_cmd = np.stack([np.sin(2 * np.pi * 0.5 * t), np.cos(2 * np.pi * 0.3 * t), 0 * t], -1)
    true = {"t": t, "pos": pos, "q": qs, "rate": rate_cmd}
    est = {"t": t + 0.01, "pos": pos, "q": qs, "rate": rate_cmd}
    m = scenario_metrics({"goal0": g0, "hover_window": (10, 30), "t_step_x": 30.0, "t_step_z": 45.0, "t_end": 59.99},
                         true, est, {"t": t, "rate": rate_cmd, "collective": 0.2 * np.sin(2 * np.pi * 4.0 * t)})
    assert m["hover_true"]["rms_3d"] < 1e-9 and m["est_pos_err_rms"] < 1e-9
    assert abs(m["est_att_lag"]["roll"]["lag_s"] - 0.01) <= 1e-3
    tau = 0.3
    assert abs(m["step_x_true"]["rise_10_90_s"] - tau * math.log(9)) < 0.01  # first order: tau ln 9
    assert abs(m["step_z_true"]["settling_s"] - tau * math.log(20)) < 0.01  # 5 cm band of a 1 m step
    assert m["step_x_true"]["overshoot_pct"] == 0.0
    assert m["rate_fopdt_roll_true"]["delay_s"] == 0.0 and m["rate_fopdt_roll_true"]["vaf"] > 0.999
    assert abs(m["hover_collective"]["peak_hz"] - 4.0) <= 0.25 and m["hover_heave"]["rms"] < 1e-9


def test_rate_oscillation_detects_limit_cycle():
    from quadsim.deploy.closed_loop_metrics import rate_oscillation

    rng = np.random.default_rng(4)
    t = np.arange(0, 30, 0.02)  # 50 Hz, like the SITL ground truth
    w = 0.15 * np.sin(2 * np.pi * 4.8 * t) + rng.normal(scale=0.01, size=len(t))
    r = rate_oscillation(t, w, 5.0, 25.0)
    assert abs(r["peak_hz"] - 4.8) <= 0.25 and abs(r["rms"] - 0.15 / math.sqrt(2)) < 0.005
    assert r["band_power_frac"] > 0.9
    quiet = rate_oscillation(t, rng.normal(scale=0.01, size=len(t)), 5.0, 25.0)
    assert quiet["band_power_frac"] < 0.5
