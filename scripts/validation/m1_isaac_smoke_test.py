"""Milestone 1: verify that Isaac Sim + Isaac Lab start, step physics on the GPU and that
PhysX free-fall matches the analytic solution.

Usage:
    python scripts/validation/m1_isaac_smoke_test.py --headless [--num_envs 64]

Acceptance criteria (exit code 0 iff all pass):
  * SimulationApp boots and the PhysX GPU pipeline steps without error.
  * Free fall of N rigid bodies for 0.5 s: |z - (z0 - g t^2/2)| < 1e-3 m (dt = 1/400 s;
    symplectic Euler has an O(dt) bias of g*t*dt/2 = 6.1e-3 m which is compensated analytically).
"""

import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--out", type=str, default="logs/validation/m1_smoke_test.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
t_boot = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
boot_s = time.time() - t_boot

import os  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402


def main():
    dt = 1.0 / 400.0
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, gravity=(0.0, 0.0, -9.81)))
    for i in range(args.num_envs):
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=(2.0 * (i % 8), 2.0 * (i // 8), 0.0))
    cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.0, angular_damping=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0)),
    )
    cube = RigidObject(cfg)
    sim.reset()
    # sim.reset() runs warm-up physics steps: re-initialise the state explicitly before measuring
    state = cube.data.root_state_w.clone()
    state[:, 2] = 10.0  # keep each body's x/y, reset height
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=state.device)
    state[:, 7:] = 0.0  # zero linear & angular velocity
    cube.write_root_state_to_sim(state)
    cube.update(0.0)
    z0 = cube.data.root_pos_w[:, 2].clone()
    steps = int(0.5 / dt)
    t0 = time.time()
    for _ in range(steps):
        cube.write_data_to_sim()
        sim.step(render=False)
        cube.update(dt)
    wall = time.time() - t0
    t = steps * dt
    g = 9.81
    z_analytic = z0 - 0.5 * g * t**2
    # symplectic Euler: z_N = z0 - g dt^2 N(N+1)/2 = analytic - g t dt / 2
    z_symplectic = z0 - g * dt**2 * steps * (steps + 1) / 2
    z = cube.data.root_pos_w[:, 2]
    err_analytic = (z - z_analytic).abs().max().item()
    err_symplectic = (z - z_symplectic).abs().max().item()
    result = {
        "device": args.device,
        "num_bodies": args.num_envs,
        "boot_seconds": boot_s,
        "physics_steps": steps,
        "wall_seconds": wall,
        "steps_per_second": steps / wall,
        "z_err_vs_analytic_m": err_analytic,
        "z_err_vs_symplectic_euler_m": err_symplectic,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "max_gpu_mem_alloc_MB": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0,
    }
    result["passed"] = bool(err_symplectic < 1e-3 and err_analytic < 1e-2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("[M1] " + json.dumps(result, indent=2))
    return result["passed"]


if __name__ == "__main__":
    ok = main()
    # disable Isaac Lab's "keep rendering after STOP" callback, otherwise close() blocks forever
    sim_utils.SimulationContext.instance()._disable_app_control_on_stop_handle = True
    simulation_app.close()
    raise SystemExit(0 if ok else 1)
