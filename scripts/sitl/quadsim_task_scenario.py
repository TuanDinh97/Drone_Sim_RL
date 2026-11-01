"""Tracking / waypoint / navigation scenarios of scripts/sitl/sitl_scenario.py flown in quadsim (sim vs SITL).

Every env starts at rest, level, yaw 0 at ``--start`` (= the SITL hand-over pose) and flies the *same* reference
(circle / figure-8 from :meth:`SinusoidReference.shape`, time from the hand-over) or waypoint course. Truth and
onboard estimate are recorded at the physics rate and evaluated with ``tracking_metrics`` / ``waypoint_metrics``
(the code used for the SITL runs, scripts/sitl/compare_tasks.py).

Usage:
    python scripts/sitl/quadsim_task_scenario.py --headless --task Quadsim-Track-v0 --checkpoint <run> \
        --track_shape circle --track_radius 1.0 --track_freq 0.2 --duration 20 --tag track_circle
    python scripts/sitl/quadsim_task_scenario.py --headless --task Quadsim-Waypoint-v0 --checkpoint <run> --tag wp
    # navigation: draw layouts from NavEnv (full density), then fly each layout in many envs
    python scripts/sitl/quadsim_task_scenario.py --headless --task Quadsim-Nav-Range-v0 --checkpoint <run> \
        --make_layouts 8 --layouts logs/sitl/nav_layouts.json env.curriculum=False
    python scripts/sitl/quadsim_task_scenario.py --headless --task Quadsim-Nav-Range-v0 --checkpoint <run> \
        --layouts logs/sitl/nav_layouts.json --num_envs 128 --tag nav env.curriculum=False
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--start", default="0,0,2.5", help="x,y,z ENU of the hand-over")
parser.add_argument("--track_shape", default="circle")
parser.add_argument("--track_radius", type=float, default=1.0)
parser.add_argument("--track_freq", type=float, default=0.2)
parser.add_argument("--track_z_amp", type=float, default=0.0)
parser.add_argument("--waypoints", default="2.5,1,2;-0.5,3,2.8;-3,0.5,1.5;-0.5,-3,2.2;3,-1.5,1.8")
parser.add_argument("--layouts", default="", help="navigation layouts JSON (written with --make_layouts)")
parser.add_argument("--make_layouts", type=int, default=0, help="navigation: draw this many layouts and exit")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--tag", default="task")
parser.add_argument("--save_envs", type=int, default=4)
parser.add_argument("--out_dir", default="logs/sitl")
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import quadsim.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from quadsim.deploy.closed_loop_metrics import tracking_metrics, waypoint_metrics  # noqa: E402
from quadsim.deploy.interface import SinusoidReference  # noqa: E402
from quadsim.rl.overrides import apply_overrides  # noqa: E402
from quadsim.rl.policy_io import agent_cfg_for, latest_checkpoint, load_policy  # noqa: E402


def stats(per_env: list[dict]) -> dict:
    out = {}
    for k, v in per_env[0].items():
        if isinstance(v, (bool, int, float)):
            x = np.array([float(e[k]) for e in per_env])
            f = x[np.isfinite(x)]
            out[k] = {"mean": float(f.mean()) if len(f) else float("nan"), "std": float(f.std()) if len(f) else 0.0,
                      "n_finite": int(len(f)), "n": int(len(x))}
    return out


def nav_main(env, u):
    """Navigation: layouts from NavEnv; every env flies layout ``env_id % L`` once (first episode)."""
    dev, n = u.device, u.num_envs
    if args.make_layouts:
        torch.manual_seed(args.seed)
        u.seed(args.seed)
        env.reset()
        lay = []
        for e in range(min(args.make_layouts, n)):
            p = (u.robot.data.root_pos_w[e] - u.scene.env_origins[e]).tolist()
            q = u.robot.data.root_quat_w[e]
            yaw = float(torch.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
            cyl = u.cylinders[e]
            lay.append({"start": p, "yaw": yaw, "goal": u.goal[e].tolist(),
                        "cylinders": cyl[cyl[:, 2] > 0].tolist()})
        os.makedirs(os.path.dirname(os.path.abspath(args.layouts)), exist_ok=True)
        with open(args.layouts, "w") as f:
            json.dump({"task": args.task, "seed": args.seed, "overrides": overrides, "layouts": lay}, f, indent=1)
        print(f"[quadsim-task] wrote {len(lay)} layouts ({[len(x['cylinders']) for x in lay]} obstacles) -> "
              f"{args.layouts}", flush=True)
        return
    layouts = json.load(open(args.layouts))["layouts"]
    m = u.cfg.num_obstacles

    def fixed_layout(env_ids):  # replaces NavEnv._sample_initial_state
        k = len(env_ids)
        start, goal = torch.zeros(k, 3, device=dev), torch.zeros(k, 3, device=dev)
        cyl = torch.zeros(k, m, 4, device=dev)
        cyl[..., 3] = u.cfg.obstacle_height
        yaw = torch.zeros(k, device=dev)
        for j, e in enumerate(env_ids.tolist()):
            L = layouts[e % len(layouts)]
            start[j], goal[j], yaw[j] = torch.tensor(L["start"]), torch.tensor(L["goal"]), L["yaw"]
            c = torch.tensor(L["cylinders"], dtype=torch.float32, device=dev)
            cyl[j, : len(c)] = c
        u.goal[env_ids] = goal
        u.cylinders[env_ids] = cyl
        pose = torch.zeros(k, m, 7, device=dev)
        pose[..., 0:2] = cyl[..., :2]
        pose[..., 2] = torch.where(cyl[..., 2] > 0, torch.full_like(cyl[..., 2], u.cfg.obstacle_height / 2),
                                   torch.full_like(cyl[..., 2], -10.0))
        pose[..., :3] += u.scene.env_origins[env_ids, None, :]
        pose[..., 3] = 1.0
        u.obstacles.write_object_pose_to_sim(pose, env_ids=env_ids)
        u.prev_goal_dist[env_ids] = (goal - start).norm(dim=-1)
        u.min_clearance[env_ids] = 1e3
        u.path_len[env_ids] = 0.0
        z = torch.zeros_like(yaw)
        quat = torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], -1)
        return start, quat, torch.zeros(k, 3, device=dev), torch.zeros(k, 3, device=dev)

    u._sample_initial_state = fixed_layout
    ckpt = args.checkpoint if args.checkpoint.endswith(".pt") else latest_checkpoint(args.checkpoint)
    pol = load_policy(ckpt, env, task=args.task)
    torch.manual_seed(args.seed)
    u.seed(args.seed)
    obs, _ = env.reset()
    u.episode_metric_buffer.clear()
    u.collect_episode_metrics = True
    first = {}
    k_rec = min(args.save_envs, n)
    rec = {"act": [], "vel": [], "yaw": []}  # policy-rate record of the first envs (velocity-loop identification)
    for _ in range(int(u.max_episode_length) + 2):
        with torch.no_grad():
            obs, *_ = env.step(pol(obs))
        q = u.robot.data.root_quat_w[:k_rec]
        rec["act"].append(u.actions[:k_rec].cpu().numpy())
        rec["vel"].append(u.robot.data.root_lin_vel_w[:k_rec].cpu().numpy())
        rec["yaw"].append(torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                      1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)).cpu().numpy())
        for r in u.episode_metric_buffer:
            for j, e in enumerate(r["env_id"].tolist()):
                if e not in first:
                    first[e] = {k: float(v[j]) for k, v in r.items() if k != "env_id"}
        u.episode_metric_buffer.clear()
        if len(first) == n:
            break
    per_layout = {}
    for e, mtr in first.items():
        per_layout.setdefault(e % len(layouts), []).append(mtr)
    summary = {}
    for li, rows in sorted(per_layout.items()):
        summary[li] = {k: float(np.nanmean([r[k] for r in rows])) if any(np.isfinite(r[k]) for r in rows)
                       else float("nan") for k in rows[0]}
        summary[li]["n"] = len(rows)
    allr = [r for rows in per_layout.values() for r in rows]
    overall = {k: float(np.nanmean([r[k] for r in allr])) if any(np.isfinite(r[k]) for r in allr) else float("nan")
               for k in allr[0]}
    res = {"task": args.task, "checkpoint": ckpt, "tag": args.tag, "overrides": overrides, "num_envs": n,
           "layouts_file": args.layouts, "per_layout": summary, "overall": overall}
    base = os.path.join(args.out_dir, f"quadsim_{args.tag}")
    with open(base + ".json", "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(base + ".npz", dt=u.step_dt, max_vel=np.asarray(u.cfg.control.max_velocity),
                        **{k: np.stack(v) for k, v in rec.items()})
    print(f"[quadsim-task] {args.tag}: success {overall['success']:.3f} collision {overall['collision']:.3f} "
          f"-> {base}.json", flush=True)


def main():
    start = np.array([float(x) for x in args.start.split(",")])
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.episode_length_s = args.duration + 10.0
    apply_overrides(cfg, agent_cfg_for(args.task), overrides)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    if type(u).__name__ == "NavEnv":
        nav_main(env, u)
        env.close()
        return
    dev, n = u.device, u.num_envs
    is_track = type(u).__name__ == "TrackingEnv"
    start_t = torch.tensor(start, dtype=torch.float32, device=dev)
    if is_track:
        ref = SinusoidReference.shape(args.track_shape, args.track_radius, args.track_freq, args.track_z_amp, start)
        h = u.cfg.num_harmonics
        amp, freq, phase = (np.zeros((h, 3)) for _ in range(3))
        amp[:3], freq[:3], phase[:3] = ref.amp, ref.freq, ref.phase  # shape() uses 3 harmonic slots
        to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    else:
        wps = np.array([[float(x) for x in w.split(",")] for w in args.waypoints.split(";")])
        assert len(wps) == u.cfg.num_waypoints, "course length must equal the trained number of waypoints"

    def fixed_initial_state(env_ids):  # replaces the task's _sample_initial_state
        k = len(env_ids)
        if is_track:
            u.amp[env_ids], u.freq[env_ids], u.phase[env_ids] = to_t(amp), to_t(freq), to_t(phase)
            u.center[env_ids] = to_t(ref.center)
        else:
            u.waypoints[env_ids] = torch.tensor(wps, dtype=torch.float32, device=dev)
            u.wp_idx[env_ids] = 0
            u.done_all[env_ids] = False
            u.time_to_finish[env_ids] = 0.0
            u.prev_dist[env_ids] = (u.waypoints[env_ids, 0] - start_t).norm(dim=-1)
        quat = torch.zeros(k, 4, device=dev)
        quat[:, 0] = 1.0
        z = torch.zeros(k, 3, device=dev)
        return start_t.expand(k, 3).clone(), quat, z, z.clone()

    u._sample_initial_state = fixed_initial_state
    ckpt = args.checkpoint if args.checkpoint.endswith(".pt") else latest_checkpoint(args.checkpoint)
    pol = load_policy(ckpt, env, task=args.task)
    n_phys = int(round(args.duration / u.physics_dt))
    rec = {k: torch.zeros(n_phys, n, 3, device=dev) for k in ("pos", "est_pos")}
    state = {"i": 0}
    orig_apply = u._apply_action

    def recording_apply():
        i = state["i"]
        if i < n_phys:
            rec["pos"][i] = u.robot.data.root_pos_w - u.scene.env_origins
        orig_apply()
        if i < n_phys:
            rec["est_pos"][i] = u.odom_meas[:, :3]
        state["i"] = i + 1

    u._apply_action = recording_apply
    torch.manual_seed(args.seed)
    u.seed(args.seed)
    obs, _ = env.reset()
    state["i"] = 0
    ended = torch.zeros(n, dtype=torch.bool, device=dev)  # crashed / course finished (env reset afterwards)
    crashed = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(int(round(args.duration / u.step_dt))):
        with torch.no_grad():
            obs, _, term, trunc, _ = env.step(pol(obs).clamp(-1.0, 1.0))
        crashed |= term.bool() & ~ended
        ended |= term.bool() | trunc.bool()
    u._apply_action = orig_apply
    t = np.arange(n_phys) * u.physics_dt
    pos, est = rec["pos"].cpu().numpy().astype(np.float64), rec["est_pos"].cpu().numpy().astype(np.float64)
    per_env = []
    for e in range(n):
        if bool(crashed[e]):
            per_env.append({"crashed": True})
            continue
        if is_track:
            r = ref(t)[0]
            m = {"crashed": False}
            for src, p in (("true", pos[:, e]), ("est", est[:, e])):
                for k, v in tracking_metrics(t, p, r, 2.0, args.duration - 1e-6).items():
                    m[f"{src}_{k}"] = v
        else:
            # the env advances its waypoints on the true position and resets the finished env: replay on the
            # trajectory up to the reset (the same logic is replayed on the SITL estimate / truth)
            m = {"crashed": False}
            for src, p in (("true", pos[:, e]), ("est", est[:, e])):
                w = waypoint_metrics(t, p, wps, u.cfg.reach_radius)
                m.update({f"{src}_{k}": v for k, v in w.items() if k != "reach_times"})
        per_env.append(m)
    ok = [m for m in per_env if not m["crashed"]]
    res = {"task": args.task, "checkpoint": ckpt, "tag": args.tag, "overrides": overrides, "num_envs": n,
           "crash_rate": float(crashed.float().mean()), "start": start.tolist(), "duration": args.duration,
           "reference": {"shape": args.track_shape, "radius": args.track_radius, "freq_hz": args.track_freq,
                         "z_amp": args.track_z_amp} if is_track else None,
           "waypoints": None if is_track else wps.tolist(), "summary": stats(ok) if ok else {}}
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"quadsim_{args.tag}")
    with open(base + ".json", "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(base + ".npz", t=t, pos=pos[:, : args.save_envs], est_pos=est[:, : args.save_envs])
    key = "true_rmse" if is_track else "true_time_to_finish"
    print(f"[quadsim-task] {args.tag}: crash {res['crash_rate']:.3f}, {key} "
          f"{res['summary'].get(key, {}).get('mean', float('nan')):.4f} -> {base}.json", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
