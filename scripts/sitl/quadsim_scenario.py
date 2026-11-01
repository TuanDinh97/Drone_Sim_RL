"""The PX4 SITL scenario (scripts/sitl/sitl_scenario.py) flown in quadsim, for the sim-vs-SITL comparison.

Every env starts exactly at the goal (level, at rest, hover action) = the hand-over; then hold ``--hold`` s,
goal +1 m x for ``--step`` s, +1 m z for ``--step`` s. Ground truth, onboard estimates, gyro and policy commands
are recorded at the physics rate (200 Hz) and evaluated per env with
:func:`quadsim.deploy.closed_loop_metrics.scenario_metrics` (the same code as the SITL runs).

Usage:
    python scripts/sitl/quadsim_scenario.py --headless --checkpoint logs/rsl_rl/quadsim_hover/<run> --tag nominal \
        env.randomization.enabled=False
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Quadsim-Hover-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--hold", type=float, default=30.0)
parser.add_argument("--step", type=float, default=15.0)
parser.add_argument("--goal_z", type=float, default=2.5, help="hover height [m]")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--tag", default="nominal")
parser.add_argument("--save_envs", type=int, default=4, help="envs whose raw time series are saved")
parser.add_argument("--out_dir", default="logs/sitl")
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.deploy.closed_loop_metrics import scenario_metrics  # noqa: E402
from quadsim.rl.overrides import apply_overrides  # noqa: E402
from quadsim.rl.policy_io import agent_cfg_for, latest_checkpoint, load_policy  # noqa: E402


def aggregate(per_env: list[dict]) -> dict:
    """Mean/std/median/min/max over envs for every scalar leaf of the metric dicts."""
    out = {}
    for k, v in per_env[0].items():
        if isinstance(v, dict):
            out[k] = aggregate([e[k] for e in per_env])
        elif isinstance(v, (int, float)):
            x = np.array([e[k] for e in per_env], dtype=np.float64)
            f = x[np.isfinite(x)]
            nan = float("nan")
            out[k] = {"mean": float(f.mean()) if len(f) else nan, "std": float(f.std()) if len(f) else 0.0,
                      "median": float(np.median(f)) if len(f) else nan, "min": float(f.min()) if len(f) else nan,
                      "max": float(f.max()) if len(f) else nan, "n_finite": int(len(f)), "n": int(len(x))}
    return out


def main():
    duration = args.hold + 2 * args.step if args.step > 0 else args.hold
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.episode_length_s = duration + 10.0
    apply_overrides(cfg, agent_cfg_for(args.task), overrides)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    dev = u.device
    n = u.num_envs
    g0 = torch.tensor([0.0, 0.0, args.goal_z], device=dev)

    def fixed_initial_state(env_ids):  # replaces HoverEnv._sample_initial_state
        k = len(env_ids)
        u.target_pos[env_ids] = g0
        u.target_yaw[env_ids] = 0.0
        quat = torch.zeros(k, 4, device=dev)
        quat[:, 0] = 1.0
        z = torch.zeros(k, 3, device=dev)
        return g0.expand(k, 3).clone(), quat, z, z.clone()

    u._sample_initial_state = fixed_initial_state
    ckpt = args.checkpoint if args.checkpoint.endswith(".pt") else latest_checkpoint(args.checkpoint)
    pol = load_policy(ckpt, env, task=args.task)

    dt = u.physics_dt
    n_phys = int(round(duration / dt))
    rec = {k: torch.zeros(n_phys, n, d, device=dev) for k, d in
           (("pos", 3), ("q", 4), ("rate", 3), ("est_pos", 3), ("est_q", 4), ("gyro", 3))}
    cmd = torch.zeros(int(round(duration / u.step_dt)), n, 4, device=dev)
    state = {"i": 0}
    orig_apply = u._apply_action

    def recording_apply():
        i = state["i"]
        if i < n_phys:
            d = u.robot.data
            rec["pos"][i] = d.root_pos_w - u.scene.env_origins
            rec["q"][i] = d.root_quat_w
            rec["rate"][i] = d.root_ang_vel_b
        orig_apply()  # samples the sensors of this physics step, then applies the (delayed) command
        if i < n_phys:
            rec["est_pos"][i] = u.odom_meas[:, :3]
            rec["est_q"][i] = u.odom_meas[:, 3:7] / u.odom_meas[:, 3:7].norm(dim=-1, keepdim=True)
            rec["gyro"][i] = u.imu_meas[:, :3]
        state["i"] = i + 1

    u._apply_action = recording_apply
    torch.manual_seed(args.seed)
    u.seed(args.seed)
    obs, _ = env.reset()
    state["i"] = 0
    crashed = torch.zeros(n, dtype=torch.bool, device=dev)
    for k in range(cmd.shape[0]):
        t = k * u.step_dt
        off = torch.zeros(3, device=dev)
        if args.step > 0 and t >= args.hold - 1e-9:
            off[0] = 1.0
        if args.step > 0 and t >= args.hold + args.step - 1e-9:
            off[2] = 1.0
        u.target_pos[:] = g0 + off
        with torch.no_grad():
            a = pol(obs).clamp(-1.0, 1.0)
        cmd[k] = a
        obs, _, term, trunc, _ = env.step(a)
        crashed |= term.bool() | trunc.bool()
    u._apply_action = orig_apply

    max_rate = torch.tensor(cfg.control.max_body_rate, device=dev)
    t_phys = np.arange(n_phys) * dt
    t_cmd = np.arange(cmd.shape[0]) * u.step_dt
    rec = {k: v.cpu().numpy().astype(np.float64) for k, v in rec.items()}
    rate_cmd = (cmd[..., 1:4] * max_rate).cpu().numpy().astype(np.float64)
    collective = cmd[..., 0].cpu().numpy().astype(np.float64)
    sched = {"goal0": g0.tolist(), "hover_window": (10.0, min(args.hold, 30.0))}
    if args.step > 0:
        sched.update(t_step_x=args.hold, t_step_z=args.hold + args.step, t_end=duration - 1e-6)
    per_env, ok_ids = [], [int(i) for i in torch.nonzero(~crashed).flatten()]
    for e in ok_ids:
        true = {"t": t_phys, "pos": rec["pos"][:, e], "q": rec["q"][:, e], "rate": rec["rate"][:, e]}
        est = {"t": t_phys, "pos": rec["est_pos"][:, e], "q": rec["est_q"][:, e], "rate": rec["gyro"][:, e]}
        c = {"t": t_cmd, "rate": rate_cmd[:, e], "collective": collective[:, e]}
        per_env.append(scenario_metrics(sched, true, est, c))
    res = {"task": args.task, "checkpoint": ckpt, "tag": args.tag, "overrides": overrides, "num_envs": n,
           "crashed_or_reset": int(crashed.sum()), "schedule": sched, "summary": aggregate(per_env) if per_env else {},
           "collective_hover_action_mean": float(cmd[..., 0].mean())}
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"quadsim_{args.tag}")
    with open(base + ".json", "w") as f:
        json.dump(res, f, indent=2)
    keep = ok_ids[: args.save_envs]
    np.savez_compressed(base + ".npz", t=t_phys, t_cmd=t_cmd, cmd=cmd[:, keep].cpu().numpy(),
                        **{k: v[:, keep] for k, v in rec.items()})
    s = res["summary"]
    brief = {k: round(s[k]["rms_3d"]["mean"], 4) for k in ("hover_true", "hover_est")} if s else {}
    print(f"[quadsim-scenario] {args.tag}: envs ok {len(ok_ids)}/{n}, {brief} -> {base}.json", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
