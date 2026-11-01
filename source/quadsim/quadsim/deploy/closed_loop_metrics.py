"""Closed-loop metrics shared by the quadsim and PX4 SITL scenario runs (NumPy only).

Both sides produce time series of the same quantities (scripts/sitl/): position (true and estimated), goal,
commanded and measured body rates. These functions turn them into the numbers of the sim-vs-SITL table, so
both columns are computed by identical code.

* :func:`hover_rms`         position error statistics in a hover window
* :func:`step_metrics`      1-D step response (rise time, overshoot, settling time, steady-state error)
* :func:`estimate_delay`    pure time shift between two signals (e.g. estimate vs ground truth)
* :func:`fit_fopdt`         first-order-plus-dead-time model of commanded -> measured body rate
* :func:`rate_oscillation`  RMS and dominant frequency of a body rate (limit-cycle detection)
* :func:`scenario_metrics`  all of the above for the hover / step scenario (scripts/sitl/)
"""

from __future__ import annotations

import numpy as np


def _window(t, t0, t1):
    t = np.asarray(t)
    m = (t >= t0) & (t <= t1)
    if m.sum() < 2:
        raise ValueError(f"window [{t0}, {t1}] contains {int(m.sum())} samples")
    return m


def hover_rms(t, pos, goal, t0: float, t1: float) -> dict:
    """Position error w.r.t. ``goal`` (3,) or (T, 3) in ``[t0, t1]``: RMS (3-D, horizontal, vertical), mean, max."""
    m = _window(t, t0, t1)
    e = np.asarray(pos)[m] - np.broadcast_to(goal, np.shape(pos))[m]
    e3 = np.linalg.norm(e, axis=-1)
    return {"rms_3d": float(np.sqrt(np.mean(e3 ** 2))), "rms_xy": float(np.sqrt(np.mean((e[:, :2] ** 2).sum(-1)))),
            "rms_z": float(np.sqrt(np.mean(e[:, 2] ** 2))), "mean_3d": float(e3.mean()), "max_3d": float(e3.max()),
            "bias": e.mean(0).tolist(), "n": int(m.sum()), "duration_s": float(t1 - t0)}


def step_metrics(t, y, t_step: float, y0: float, y1: float, t_end: float, band: float = 0.05) -> dict:
    """Step ``y0 -> y1`` commanded at ``t_step``, evaluated until ``t_end``.

    rise time 10-90 %, overshoot in % of the step, settling time = last exit from ``|y - y1| <= band``
    (NaN if never settled), steady-state error = mean over the last 20 % of the window.
    """
    t, y = np.asarray(t, dtype=np.float64), np.asarray(y, dtype=np.float64)
    m = _window(t, t_step, t_end)
    ts, ys = t[m] - t_step, y[m]
    d = y1 - y0
    p = (ys - y0) / d  # normalized progress
    i10 = np.argmax(p >= 0.1) if np.any(p >= 0.1) else None
    i90 = np.argmax(p >= 0.9) if np.any(p >= 0.9) else None
    rise = float(ts[i90] - ts[i10]) if i10 is not None and i90 is not None else float("nan")
    overshoot = float(max(0.0, (p.max() - 1.0) * 100.0))
    outside = np.abs(ys - y1) > band
    if outside[-1]:
        settle = float("nan")
    else:
        last_out = np.nonzero(outside)[0]
        settle = float(ts[last_out[-1] + 1]) if len(last_out) else 0.0
    tail = ts >= 0.8 * ts[-1]
    return {"rise_10_90_s": rise, "overshoot_pct": overshoot, "settling_s": settle, "band_m": band,
            "ss_err": float(np.mean(ys[tail]) - y1), "peak": float(ys.max() if d > 0 else ys.min())}


def resample(t, x, t_new):
    """Linear interpolation of (T,) or (T, D) samples at ``t_new``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return np.interp(t_new, t, x)
    return np.stack([np.interp(t_new, t, x[:, i]) for i in range(x.shape[1])], -1)


def estimate_delay(t_ref, x_ref, t_del, x_del, max_lag: float, dt: float = 1e-3) -> dict:
    """Shift ``L`` minimizing ``mean |x_del(t) - x_ref(t - L)|^2`` (positive L: ``x_del`` lags ``x_ref``)."""
    t0 = max(t_ref[0], t_del[0]) + max_lag
    t1 = min(t_ref[-1], t_del[-1])
    tg = np.arange(t0, t1, dt)
    y = resample(t_del, x_del, tg)
    lags = np.arange(-max_lag, max_lag + dt / 2, dt)
    cost = np.array([np.mean((y - resample(t_ref, x_ref, tg - lag)) ** 2) for lag in lags])
    i = int(np.argmin(cost))
    return {"lag_s": float(lags[i]), "rmse_at_lag": float(np.sqrt(cost[i])), "rmse_at_zero": float(np.sqrt(
        cost[np.argmin(np.abs(lags))]))}


def _first_order_response(u, a):
    """y[k+1] = a y[k] + (1 - a) u[k], y[0] = u[0] (unit DC gain)."""
    from scipy.signal import lfilter

    y, _ = lfilter([0.0, 1.0 - a], [1.0, -a], u, zi=[u[0]])  # transposed form: y[0] = zi[0]
    return y


def fit_fopdt(t_u, u, t_y, y, dt: float = 2e-3, max_delay: float = 0.1,
              taus=np.geomspace(0.005, 0.5, 60)) -> dict:
    """Output-error fit of ``y = K e^{-L s} / (tau s + 1) u`` (e.g. commanded -> measured body rate).

    ``u`` is treated as zero-order-held between its samples (set-points), ``y`` is linearly interpolated.
    Grid search over delay ``L`` (step ``dt``) and ``tau``; gain ``K`` by least squares. Returns L, tau, K and
    the variance-accounted-for ``vaf`` (1 = perfect).
    """
    t_u, u, t_y, y = map(lambda a: np.asarray(a, dtype=np.float64), (t_u, u, t_y, y))
    t0, t1 = max(t_u[0], t_y[0]), min(t_u[-1], t_y[-1])
    tg = np.arange(t0, t1, dt)
    idx = np.clip(np.searchsorted(t_u, tg, side="right") - 1, 0, len(u) - 1)
    ug = u[idx]
    yg = np.interp(tg, t_y, y)
    n_skip = int(round(max_delay / dt))
    yt = yg[n_skip:]
    best = None
    for tau in taus:
        base = _first_order_response(ug, float(np.exp(-dt / tau)))
        for d in range(0, n_skip + 1):
            x = base[n_skip - d: len(base) - d]
            k = float(x @ yt / max(x @ x, 1e-12))
            err = float(np.mean((yt - k * x) ** 2))
            if best is None or err < best[0]:
                best = (err, d * dt, float(tau), k)
    err, lag, tau, k = best
    var = float(np.var(yt))
    return {"delay_s": lag, "tau_s": tau, "gain": k, "vaf": 1.0 - err / var if var > 0 else float("nan"),
            "rmse": float(np.sqrt(err)), "n": int(len(yt))}


def rate_oscillation(t, w, t0: float, t1: float, band=(2.0, 8.0)) -> dict:
    """RMS, dominant frequency (0.5 Hz .. Nyquist) and fraction of power in ``band`` of a 1-D signal in [t0, t1]."""
    from scipy.signal import welch

    m = _window(t, t0, t1)
    dt = float(np.median(np.diff(np.asarray(t)[m])))
    tg = np.arange(np.asarray(t)[m][0], np.asarray(t)[m][-1], dt)
    x = np.interp(tg, np.asarray(t)[m], np.asarray(w)[m])
    x = x - x.mean()
    f, p = welch(x, fs=1.0 / dt, nperseg=min(len(x), int(round(4.0 / dt))))
    sel = f >= 0.5
    in_band = (f >= band[0]) & (f <= band[1])
    return {"rms": float(x.std()), "peak_hz": float(f[sel][np.argmax(p[sel])]),
            "band_power_frac": float(p[in_band].sum() / max(p[f > 0.2].sum(), 1e-30))}


def roll_pitch(q):
    """Roll and pitch [rad] of wxyz quaternions (z-up world, FLU body; ZYX Euler)."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return np.stack([roll, pitch], -1)


def scenario_metrics(sched: dict, true: dict, est: dict, cmd: dict, band: float = 0.05) -> dict:
    """Metrics of the hover + step scenario; times in s relative to the policy hand-over.

    sched: ``goal0`` (3,), ``hover_window`` (t0, t1), ``t_step_x``, ``t_step_z``, ``t_end`` (steps of +1 m in
           world x, then +1 m in z; omit the step keys for hover-only runs)
    true / est: ``t``, ``pos`` (T, 3) ENU, ``q`` (T, 4) wxyz ENU/FLU, ``rate`` (T, 3) body FLU; ``est`` also
           ``t_rate`` / ``rate`` for the onboard gyro estimate if sampled differently
    cmd:  ``t``, ``rate`` (T, 3) commanded body rates FLU [rad/s] (zero-order held), optional ``collective``
          (T,) normalized collective action a0
    """
    g0 = np.asarray(sched["goal0"], dtype=np.float64)
    w0, w1 = sched["hover_window"]
    out = {"hover_true": hover_rms(true["t"], true["pos"], g0, w0, w1),
           "hover_est": hover_rms(est["t"], est["pos"], g0, w0, w1)}
    # estimator: position error and attitude lag vs ground truth in the hover window + steps
    tg = np.asarray(true["t"])
    m = (tg >= w0) & (tg <= w1)
    p_est = resample(est["t"], est["pos"], tg[m])
    out["est_pos_err_rms"] = float(np.sqrt(np.mean(np.sum((p_est - true["pos"][m]) ** 2, -1))))
    out["est_pos_bias"] = (p_est - true["pos"][m]).mean(0).tolist()
    t_end = sched.get("t_end", w1)
    rp_true, rp_est = roll_pitch(true["q"]), roll_pitch(est["q"])
    out["est_att_lag"] = {ax: estimate_delay(true["t"], rp_true[:, i], est["t"], rp_est[:, i], max_lag=0.06)
                          for i, ax in enumerate(("roll", "pitch"))}
    if "t_step_x" in sched:
        for key, t_a, t_b, i in (("step_x", sched["t_step_x"], sched["t_step_z"], 0),
                                 ("step_z", sched["t_step_z"], t_end, 2)):
            for src, d in (("true", true), ("est", est)):
                out[f"{key}_{src}"] = step_metrics(d["t"], d["pos"][:, i], t_a, g0[i], g0[i] + 1.0, t_b, band)
    out["hover_rate"] = {ax: rate_oscillation(true["t"], true["rate"][:, i], w0, w1)
                         for i, ax in enumerate(("roll", "pitch"))}
    out["hover_heave"] = rate_oscillation(true["t"], np.gradient(true["pos"][:, 2], true["t"]), w0, w1)
    if "collective" in cmd:
        out["hover_collective"] = rate_oscillation(cmd["t"], cmd["collective"], w0, w1)
    # commanded -> measured body rate (roll, pitch) over the whole flight
    for src, t_r, r in (("true", true["t"], true["rate"]), ("est", est.get("t_rate", est["t"]), est["rate"])):
        for i, ax in enumerate(("roll", "pitch")):
            out[f"rate_fopdt_{ax}_{src}"] = fit_fopdt(cmd["t"], cmd["rate"][:, i], t_r, r[:, i])
    return out
