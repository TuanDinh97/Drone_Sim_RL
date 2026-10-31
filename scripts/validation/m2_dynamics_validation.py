"""Milestone 2: validate the Isaac Lab quadrotor dynamics against analytic models and the
independent reference integrator (quadsim.core.dynamics).

Tests (all with sensor noise, delays, wind and domain randomization disabled):
  T1 inertial properties: PhysX mass/inertia read back == preset (rel. err < 1e-4)
  T2 hover equilibrium: at the analytic hover speed the vehicle holds altitude (|dz| < 5 mm over 2 s)
  T3 vertical thrust step: measured vertical acceleration matches T(omega(t))/m - g using the
     first-order motor response (RMS error < 2% of peak)
  T4 yaw torque: differential CW/CCW command -> yaw acceleration == tau_z / I_zz (rel. err < 3%)
  T5 open-loop trajectory: identical multisine rotor commands applied to PhysX and to the reference
     simulator (semi-implicit and RK4); position error after 1.5 s < 1 cm (semi-implicit),
     attitude error < 0.5 deg
  T6 closed-loop: velocity-mode cascade step response in PhysX vs reference (pos err < 3 cm @ 3 s)

Usage: python scripts/validation/m2_dynamics_validation.py --headless [--preset x500]
Writes logs/validation/m2_dynamics_<preset>.json and a figure to docs/figures/.
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="x500")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--out_dir", default="logs/validation")
parser.add_argument("--fig_dir", default="docs/figures")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from quadsim.core import QuadrotorModel, ReferenceQuadrotorSim  # noqa: E402
from quadsim.core import math as qm  # noqa: E402
from quadsim.core.controllers import Mixer  # noqa: E402
from quadsim.tasks.base import QuadrotorEnv, QuadrotorEnvCfg  # noqa: E402


def make_env(mode: str, rotor_drag=True) -> QuadrotorEnv:
    import omni.usd

    omni.usd.get_context().new_stage()
    cfg = QuadrotorEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 6.0
    cfg.dynamics.preset = args.preset
    cfg.dynamics.rotor_drag = rotor_drag
    cfg.dynamics.body_drag = rotor_drag
    cfg.dynamics.ground_effect = False
    cfg.control.mode = mode
    cfg.control.action_delay_steps = (0, 0)
    cfg.control.use_imu_gyro_for_rate_loop = False
    cfg.randomization.enabled = False
    cfg.sensors.noise_enabled = False
    cfg.episode_length_s = 100.0
    cfg.min_height = -1e3
    cfg.max_distance = 1e3
    cfg.max_lin_vel = 1e3
    cfg.max_ang_vel = 1e3
    cfg.sim.device = args.device
    return QuadrotorEnv(cfg)


def set_state(env, pos, quat=None, lin_vel=None, ang_vel=None):
    n = env.num_envs
    dev = env.device
    quat = torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n, 1) if quat is None else quat
    lin_vel = torch.zeros(n, 3, device=dev) if lin_vel is None else lin_vel
    ang_vel = torch.zeros(n, 3, device=dev) if ang_vel is None else ang_vel
    root = torch.cat([pos + env.scene.env_origins, quat, lin_vel, ang_vel], -1)
    env.robot.write_root_state_to_sim(root)
    env.robot.reset()
    env.model.reset()
    env.action_delay.reset(torch.arange(n, device=dev), env.hover_action.expand(n, -1))
    env.rate_ctrl.reset(None, ang_vel)
    env.pos_ctrl.reset()


def physics_step(env, action):
    """One env (policy) step = decimation physics steps; returns per-physics-step states."""
    env._pre_physics_step(action)
    states = []
    for _ in range(env.cfg.decimation):
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        d = env.robot.data
        states.append(torch.cat([d.root_pos_w - env.scene.env_origins, d.root_quat_w, d.root_lin_vel_w,
                                 d.root_ang_vel_b], -1).clone())
    return states


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    results = {"preset": args.preset}
    env = make_env("motor")
    env.reset()
    p = env.params
    dev = env.device
    n = env.num_envs
    dt = env.physics_dt
    mixer = Mixer(p, dev)

    # ---------------- T1 inertial properties ----------------
    m_px = env.robot.root_physx_view.get_masses()[:, 0]
    i_px = env.robot.root_physx_view.get_inertias()[:, [0, 4, 8]]
    t1_m = (m_px / p.mass - 1).abs().max().item()
    t1_i = (i_px / torch.tensor(p.inertia) - 1).abs().max().item()
    results["T1_inertial"] = {"mass_rel_err": t1_m, "inertia_rel_err": t1_i, "pass": t1_m < 1e-4 and t1_i < 1e-4}

    # ---------------- T2 hover ----------------
    start = torch.tensor([0.0, 0.0, 2.0], device=dev).repeat(n, 1)
    set_state(env, start)
    hover_a = env.hover_action.expand(n, -1)
    for _ in range(int(2.0 / (dt * env.cfg.decimation))):
        states = physics_step(env, hover_a)
    dz = (states[-1][:, 2] - 2.0).abs().max().item()
    results["T2_hover"] = {"max_abs_dz_m": dz, "pass": dz < 5e-3}

    # ---------------- T3 vertical thrust step ----------------
    set_state(env, start)
    a_step = torch.full((n, 4), 2.0 * 0.8 - 1.0, device=dev)  # 80% of max thrust per rotor
    vz, t_hist = [], []
    model_ref = QuadrotorModel(p, 1, dev)
    model_ref.reset()
    pred_acc = []
    for k in range(int(0.3 / (dt * env.cfg.decimation))):
        states = physics_step(env, a_step)
        for s in states:
            vz.append(s[0, 9].item())
            omega_cmd = mixer.thrust_to_omega((a_step[:1] + 1) * 0.5 * mixer.t_max)
            model_ref.step_motors(omega_cmd, dt)
            drag = p.body_drag_quad[2] * vz[-1] * abs(vz[-1])  # body drag at the start of the step
            pred_acc.append(((model_ref.k_f * model_ref.omega**2).sum().item() - drag) / p.mass - 9.81)
    meas_acc = np.diff(np.array([0.0] + vz)) / dt
    pred_acc = np.array(pred_acc)
    rms = float(np.sqrt(np.mean((meas_acc - pred_acc) ** 2)))
    peak = float(np.abs(pred_acc).max())
    results["T3_thrust_step"] = {"rms_err_mps2": rms, "peak_mps2": peak, "rel": rms / peak, "pass": rms / peak < 0.02}

    # ---------------- T4 yaw torque ----------------
    # rotor drag damps yaw (tangential rotor velocity); isolate the pure reaction-torque model here.
    saved = {k: getattr(env.model, k).clone() for k in ("rotor_drag_coeff", "rolling_moment_coeff", "body_drag_quad")}
    for k in saved:
        getattr(env.model, k)[:] = 0.0
    set_state(env, start)
    w_h = p.hover_omega
    t_h = p.k_f * w_h**2
    d_t = 0.15 * t_h
    thr = torch.tensor([t_h - d_t, t_h - d_t, t_h + d_t, t_h + d_t], device=dev)  # CW rotors (2,3) up
    a_yaw = (2 * thr / mixer.t_max - 1).expand(n, -1)
    yaw_rates = []
    for _ in range(int(0.4 / (dt * env.cfg.decimation))):
        states = physics_step(env, a_yaw)
        yaw_rates += [s[0, 12].item() for s in states]
    # steady thrusts after motor transient: fit slope over the last 0.2 s
    yr = np.array(yaw_rates)
    t = np.arange(len(yr)) * dt
    sel = t > 0.2
    slope = float(np.polyfit(t[sel], yr[sel], 1)[0])
    tau_z = p.k_m * 4 * d_t
    pred = tau_z / p.inertia[2]
    results["T4_yaw"] = {"yaw_acc_meas": slope, "yaw_acc_pred": pred, "rel_err": abs(slope / pred - 1),
                         "pass": abs(slope / pred - 1) < 0.03}
    for k, v in saved.items():
        getattr(env.model, k)[:] = v

    # ---------------- T5 open-loop vs reference ----------------
    set_state(env, start, lin_vel=torch.tensor([0.5, -0.3, 0.0], device=dev).repeat(n, 1))
    refs = {}
    for method in ("semi_implicit", "rk4"):
        m = QuadrotorModel(p, 1, dev)
        m.reset()
        r = ReferenceQuadrotorSim(m, dt, method=method, substeps=1 if method == "semi_implicit" else 10)
        r.state.pos[:] = start[:1]
        r.state.lin_vel[:] = torch.tensor([[0.5, -0.3, 0.0]], device=dev)
        refs[method] = (m, r)
    hist = {"isaac": [], "semi_implicit": [], "rk4": []}
    steps = int(1.5 / (dt * env.cfg.decimation))
    for k in range(steps):
        t = k * dt * env.cfg.decimation
        u = torch.full((1, 4), 2 * (p.weight / 4) / mixer.t_max - 1, device=dev)
        u[0, 0] += 0.04 * math.sin(2 * math.pi * 1.3 * t)
        u[0, 1] += 0.04 * math.sin(2 * math.pi * 0.7 * t + 1.0)
        u[0, 2] += 0.03 * math.sin(2 * math.pi * 2.1 * t + 0.3)
        u[0, 3] += 0.03 * math.cos(2 * math.pi * 1.7 * t)
        states = physics_step(env, u.expand(n, -1))
        omega_cmd = mixer.thrust_to_omega((u + 1) * 0.5 * mixer.t_max)
        for s in states:
            hist["isaac"].append(s[0].cpu().numpy())
        for method, (m, r) in refs.items():
            for _ in range(env.cfg.decimation):
                r.step(omega_cmd)
                hist[method].append(torch.cat([r.state.pos[0], r.state.quat[0], r.state.lin_vel[0],
                                               r.state.ang_vel[0]]).cpu().numpy())
    H = {k: np.array(v) for k, v in hist.items()}
    def att_err_deg(a, b):
        qa, qb = torch.tensor(a[:, 3:7]), torch.tensor(b[:, 3:7])
        return np.degrees(qm.quat_log_rotvec(qm.quat_mul(qm.quat_conjugate(qa), qb)).norm(dim=-1).numpy())
    e_si = np.linalg.norm(H["isaac"][:, :3] - H["semi_implicit"][:, :3], axis=1)
    e_rk = np.linalg.norm(H["isaac"][:, :3] - H["rk4"][:, :3], axis=1)
    a_si = att_err_deg(H["isaac"], H["semi_implicit"])
    a_rk = att_err_deg(H["isaac"], H["rk4"])
    travel = float(np.linalg.norm(H["isaac"][-1, :3] - H["isaac"][0, :3]))
    results["T5_open_loop"] = {
        "duration_s": 1.5, "travel_m": travel,
        "pos_err_final_semi_implicit_m": float(e_si[-1]), "pos_err_final_rk4_m": float(e_rk[-1]),
        "att_err_max_semi_implicit_deg": float(a_si.max()), "att_err_max_rk4_deg": float(a_rk.max()),
        "pass": bool(e_si[-1] < 0.01 and a_si.max() < 0.5),
    }
    env.close()

    # ---------------- T6 closed loop velocity mode ----------------
    env = make_env("velocity")
    env.reset()
    set_state(env, start)
    cmd = torch.tensor([[0.5, 0.0, 0.3, 0.0]], device=dev).expand(n, -1)  # forward 1.5 m/s, up 0.6 m/s
    from quadsim.core.controllers import AttitudeController, PositionController, RateController, thrust_vector_to_setpoints
    m = QuadrotorModel(p, 1, dev)
    m.reset()
    r = ReferenceQuadrotorSim(m, dt, method="rk4", substeps=10)
    r.state.pos[:] = start[:1]
    c = env.cfg.control
    pc, ac, rc = PositionController(p, 1, dt, c.position_gains, dev), AttitudeController(c.attitude_gains, dev), \
        RateController(p, 1, dt, c.rate_gains, dev)
    yaw_sp = torch.zeros(1, device=dev)
    hi, hr = [], []
    for k in range(int(3.0 / (dt * env.cfg.decimation))):
        states = physics_step(env, cmd)
        hi += [s[0].cpu().numpy() for s in states]
        for _ in range(env.cfg.decimation):
            s = r.state
            yaw = qm.yaw_from_quat(s.quat)
            v_h = cmd[:1, :3] * env.max_vel
            vel_sp = torch.stack([torch.cos(yaw) * v_h[:, 0] - torch.sin(yaw) * v_h[:, 1],
                                  torch.sin(yaw) * v_h[:, 0] + torch.cos(yaw) * v_h[:, 1], v_h[:, 2]], -1)
            thrust_w = pc.velocity_to_thrust(vel_sp, s.lin_vel)
            coll, q_sp = thrust_vector_to_setpoints(thrust_w, s.quat, yaw_sp)
            tau = rc(ac(s.quat, q_sp), s.ang_vel)
            r.step(env.mixer.thrust_to_omega(env.mixer.allocate(coll, tau)))
            hr.append(torch.cat([s.pos[0], s.quat[0], s.lin_vel[0], s.ang_vel[0]]).cpu().numpy())
    hi, hr = np.array(hi), np.array(hr)
    e_cl = float(np.linalg.norm(hi[-1, :3] - hr[-1, :3]))
    v_final = hi[-1, 7:10].tolist()
    results["T6_closed_loop"] = {"pos_err_final_m": e_cl, "vel_final_isaac": v_final,
                                 "vel_cmd": (cmd[0, :3] * env.max_vel).tolist(), "pass": e_cl < 0.03}
    env.close()

    results["all_pass"] = all(v["pass"] for k, v in results.items() if isinstance(v, dict) and "pass" in v)
    out = os.path.join(args.out_dir, f"m2_dynamics_{args.preset}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("[M2] " + json.dumps(results, indent=2), flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        tt = np.arange(len(H["isaac"])) * dt
        for i, lab in enumerate("xyz"):
            ax[0].plot(tt, H["isaac"][:, i], label=f"PhysX {lab}")
            ax[0].plot(tt, H["rk4"][:, i], "--", color="k", lw=0.8)
        ax[0].set_title("T5 open-loop position (dashed: RK4 ref.)")
        ax[0].set_xlabel("t [s]"); ax[0].legend(fontsize=7)  # noqa: E702
        ax[1].semilogy(tt, e_si + 1e-9, label="vs semi-implicit ref.")
        ax[1].semilogy(tt, e_rk + 1e-9, label="vs RK4 ref.")
        ax[1].set_title("T5 position error [m]"); ax[1].legend(fontsize=7); ax[1].set_xlabel("t [s]")  # noqa: E702
        t3 = np.arange(len(meas_acc)) * dt
        ax[2].plot(t3, meas_acc, label="PhysX (finite diff.)")
        ax[2].plot(t3, pred_acc, "--", label="motor model prediction")
        ax[2].set_title("T3 thrust step: vertical accel. [m/s²]"); ax[2].legend(fontsize=7)  # noqa: E702
        ax[2].set_xlabel("t [s]")
        fig.tight_layout()
        fig.savefig(os.path.join(args.fig_dir, f"m2_dynamics_{args.preset}.pdf"))
        fig.savefig(os.path.join(args.fig_dir, f"m2_dynamics_{args.preset}.png"), dpi=120)
    except Exception as e:  # pragma: no cover
        print("figure failed:", e)
    return results["all_pass"]


if __name__ == "__main__":
    ok = main()
    app.close()
    raise SystemExit(0 if ok else 1)
