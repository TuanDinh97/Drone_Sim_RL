"""Milestone 3: rendered depth (Isaac Sim RTX TiledCamera) vs analytic ray-cast depth.

Validates camera intrinsics (HFOV, square pixels), extrinsics (body offset + pitch), frame conventions,
obstacle placement and the analytic ray caster used for fast depth-based training.

Acceptance (noise disabled, pixels where either source sees something within 10 m):
  * median |d_render - d_raycast| < 1 cm
  * 90th percentile < 5 cm   (residuals concentrate on depth discontinuities / anti-aliasing)
  * hit/miss agreement (both < 10 m or both >= 10 m) > 99 %

Usage: python scripts/validation/m3_depth_consistency.py --headless --enable_cameras
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--poses", type=int, default=4)
parser.add_argument("--pitch", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from quadsim.core import math as qm  # noqa: E402
from quadsim.tasks import NavDepthRenderEnvCfg, NavEnv  # noqa: E402


def main():
    cfg = NavDepthRenderEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sensors.noise_enabled = False
    cfg.randomization.enabled = False
    cfg.vision.depth_noise.enabled = False
    cfg.vision.pitch_deg = args.pitch
    cfg.vision.width, cfg.vision.height = 96, 72
    cfg.active_obstacles_range = (24, 24)
    cfg.min_obstacle_gap = 0.4
    cfg.sim.device = args.device
    env = NavEnv(cfg)
    env.reset()
    n, dev = env.num_envs, env.device
    max_r = 10.0
    errs, agree, examples, self_occ = [], [], None, []
    gen = torch.Generator(device="cpu").manual_seed(0)
    for k in range(args.poses):
        pos = torch.zeros(n, 3)
        pos[:, 0] = torch.rand(n, generator=gen) * 3.0 - 1.0
        pos[:, 1] = (torch.rand(n, generator=gen) * 2 - 1) * 2.0
        pos[:, 2] = 0.7 + torch.rand(n, generator=gen) * 1.5
        yaw = (torch.rand(n, generator=gen) * 2 - 1) * 0.8
        roll = (torch.rand(n, generator=gen) * 2 - 1) * 0.2
        pitch = (torch.rand(n, generator=gen) * 2 - 1) * 0.2
        quat = qm.quat_from_euler_xyz(roll, pitch, yaw)
        root = torch.cat([pos.to(dev) + env.scene.env_origins, quat.to(dev), torch.zeros(n, 6, device=dev)], -1)
        env.robot.write_root_state_to_sim(root)
        env.robot.permanent_wrench_composer.reset()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        for _ in range(3):
            env.sim.render()
        env.camera.update(env.physics_dt, force_recompute=True)
        d_r = env.camera.data.output["distance_to_image_plane"][..., 0].clone()
        d_r = torch.where(torch.isfinite(d_r), d_r, torch.full_like(d_r, 1e3))
        d_a = env.raycast_depth()
        # pixels showing the vehicle itself (rotor discs are < 0.4 m from the camera): the analytic caster
        # does not model self-geometry, so they are reported separately and excluded from the accuracy metrics
        own = (d_r < 0.4) & (d_a > 0.45)
        self_occ.append(own.float().flatten())
        hit_r, hit_a = (d_r < max_r) & ~own, (d_a < max_r) & ~own
        agree.append((hit_r == hit_a)[~own].float().flatten())
        both = hit_r | hit_a
        e = (d_r.clamp(max=max_r) - d_a.clamp(max=max_r)).abs()[both]
        errs.append(e.flatten())
        if examples is None:
            examples = (d_r[:4].cpu().numpy(), d_a[:4].cpu().numpy())
    e = torch.cat(errs).cpu().numpy()
    ag = torch.cat(agree).mean().item()
    res = {
        "num_pixels_compared": int(e.size),
        "median_abs_err_m": float(np.median(e)),
        "mean_abs_err_m": float(np.mean(e)),
        "p90_abs_err_m": float(np.percentile(e, 90)),
        "p99_abs_err_m": float(np.percentile(e, 99)),
        "frac_err_below_2cm": float(np.mean(e < 0.02)),
        "hit_miss_agreement": ag,
        "self_occluded_pixel_frac": torch.cat(self_occ).mean().item(),
        "cam_offset_m": list(cfg.vision.offset_pos),
        "image": [cfg.vision.width, cfg.vision.height], "hfov_deg": cfg.vision.hfov_deg, "cam_pitch_deg": args.pitch,
    }
    res["pass"] = bool(res["median_abs_err_m"] < 0.01 and res["p90_abs_err_m"] < 0.05 and ag > 0.99)
    os.makedirs("logs/validation", exist_ok=True)
    with open(f"logs/validation/m3_depth_consistency_pitch{int(args.pitch)}.json", "w") as f:
        json.dump(res, f, indent=2)
    print("[M3-DEPTH] " + json.dumps(res, indent=2), flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dr, da = examples
        fig, ax = plt.subplots(3, 4, figsize=(12, 6.4))
        for i in range(4):
            ax[0, i].imshow(np.clip(dr[i], 0, max_r), cmap="viridis_r", vmin=0, vmax=max_r)
            ax[1, i].imshow(np.clip(da[i], 0, max_r), cmap="viridis_r", vmin=0, vmax=max_r)
            im = ax[2, i].imshow(np.abs(np.clip(dr[i], 0, max_r) - np.clip(da[i], 0, max_r)), cmap="magma", vmin=0, vmax=0.1)
            for a in ax[:, i]:
                a.set_xticks([]); a.set_yticks([])  # noqa: E702
        ax[0, 0].set_ylabel("RTX rendered"); ax[1, 0].set_ylabel("analytic ray-cast"); ax[2, 0].set_ylabel("|diff| (0-10 cm)")  # noqa: E702
        fig.colorbar(im, ax=ax[2, :].tolist(), shrink=0.8)
        os.makedirs("docs/figures", exist_ok=True)
        fig.savefig(f"docs/figures/m3_depth_consistency_pitch{int(args.pitch)}.png", dpi=110, bbox_inches="tight")
        fig.savefig(f"docs/figures/m3_depth_consistency_pitch{int(args.pitch)}.pdf", bbox_inches="tight")
    except Exception as ex:  # pragma: no cover
        print("figure failed", ex)
    env.close()
    return res["pass"]


if __name__ == "__main__":
    main()
    app.close()
