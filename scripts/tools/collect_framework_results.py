"""Collect the framework-milestone results from logs/ into curated JSON files in docs/results/framework/.

Usage: python scripts/tools/collect_framework_results.py
Every entry records its source file; missing sources are listed under "missing" (nothing is invented).
"""

import glob
import json
import os
import re

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # paths are repo-relative
OUT = "docs/results/framework"
os.makedirs(OUT, exist_ok=True)
missing = []


def ev(path, controller=None):
    """Means (+ CI, n) of one controller from an evaluate.py JSON."""
    if not os.path.exists(path):
        missing.append(path)
        return None
    d = json.load(open(path))
    c = d["controllers"][controller] if controller else next(iter(d["controllers"].values()))
    out = {k: {"mean": v["mean"], "ci95": v.get("ci95"), "n": v.get("n")} for k, v in c.items()
           if isinstance(v, dict) and "mean" in v}
    out["_source"] = path
    out["_overrides"] = d.get("overrides", [])
    return out


def wall(log):
    if not os.path.exists(log):
        return None
    m = re.findall(r"finished in ([0-9.]+) s", open(log).read())
    return float(m[-1]) if m else None


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2)


# ---- algorithms on hover (benchmark suite) + off-policy training cost
s = "logs/benchmark_suites/hover/algos/summary.json"
dump("bench_hover_algos.json", json.load(open(s)) if os.path.exists(s) else missing.append(s))
offp = {}
for algo, tag, log in (("sac", "sac_s42", "logs/train_fw_sac.log"), ("td3", "td3_s42", "logs/train_fw_td3.log"),
                       ("td3b", "td3b_s42", "logs/train_fw_td3b.log"), ("ars", "ars_s42", "logs/train_fw_ars.log")):
    e = ev(f"logs/eval/algos/Quadsim-Hover-v0_{tag}.json")
    if e:
        offp[algo] = {"eval": e, "train_wall_s": wall(log)}
offp["ppo_reference"] = {"eval": ev("logs/eval/reverify/Quadsim-Hover-v0_regress_final.json", "policy"),
                         "train_wall_s": wall("logs/train_hover_ctbr_s42.log"), "note": "600 iterations, 4096 envs"}
dump("offpolicy_hover.json", offp)

# ---- vehicles and control interfaces
veh = {"hexa_ctbr_ppo": ev("logs/eval/vehicles/Quadsim-Hover-Hexa-v0_hexa_ctbr_s42.json", "policy"),
       "hexa_ctbr_cascade": ev("logs/eval/vehicles/Quadsim-Hover-Hexa-v0_hexa_ctbr_s42.json", "cascade"),
       "hexa_motor_ppo": ev("logs/eval/vehicles/Quadsim-Hover-Hexa-Motor-v0_hexa_motor_s42.json"),
       "accel_interface_ppo": ev("logs/eval/vehicles/Quadsim-Hover-v0_accel_s42.json"),
       "landing_ppo_reward_v1": ev("logs/eval/vehicles/Quadsim-Example-Land-v0_land_s42.json", "policy"),
       "landing_ppo": ev("logs/eval/vehicles/Quadsim-Example-Land-v0_land_v2_s42.json", "policy"),
       "landing_cascade": ev("logs/eval/vehicles/Quadsim-Example-Land-v0_land_v2_s42.json", "cascade")}
dump("vehicles_interfaces.json", veh)

# ---- safe RL
dump("safe_rl.json", {
    "task": "Quadsim-Waypoint-SpeedCost-v0 (speed limit 2 m/s, budget 0.1 s/episode)",
    "unconstrained_ppo": ev("logs/eval/safe/Quadsim-Waypoint-SpeedCost-v0_unconstrained_s42.json", "policy"),
    "unconstrained_cascade": ev("logs/eval/safe/Quadsim-Waypoint-SpeedCost-v0_unconstrained_s42.json", "cascade"),
    "ppo_lagrangian": ev("logs/eval/safe/Quadsim-Waypoint-SpeedCost-v0_safe_lagrangian_s42.json"),
    "train_wall_s": wall("logs/train_fw_safe.log")})

# ---- racing, swarm (benchmark suites; the curated tag, not e.g. the multi-seed "seeds" tag)
for suite in ("race", "swarm"):
    p = f"logs/benchmark_suites/{suite}/final/summary.json"
    dump(f"bench_{suite}.json", json.load(open(p)) if os.path.exists(p) else missing.append(p))

# ---- imitation learning
il = {"hover_distillation": {
    "expert_ppo": ev("logs/eval/reverify/Quadsim-Hover-v0_regress_final.json", "policy"),
    "bc": ev("logs/eval/il/Quadsim-Hover-v0_hover_bc.json"),
    "bc_no_last_action": ev("logs/eval/il/Quadsim-Hover-v0_hover_bc_nolast.json")}}
sym = {}
for tag in ("expert", "langnav_sym_bc", "langnav_sym_bc_nolang", "langnav_sym_bc_nolast", "langnav_sym_bc_v2_unkavg",
            "langnav_sym_bc_v2", "langnav_sym_dagger1", "langnav_sym_dagger2", "langnav_sym_dagger2_clip"):
    for task in ("Quadsim-LangNav-v0", "Quadsim-LangNav-Heldout-v0"):
        p = f"logs/eval/il/{task}_{tag}.json"
        if os.path.exists(p):
            sym.setdefault(tag, {})[task] = ev(p)
il["langnav_symbolic"] = sym
rgb = {}
for tag in ("expert", "langnav_rgb_bc", "langnav_rgb_dagger1", "langnav_rgb_dagger1_clip", "vla_color_servo",
            "langnav_rgb_bc_v2data", "langnav_rgb_dagger1_v2data"):
    for task in ("Quadsim-LangNav-RGB-v0", "Quadsim-LangNav-RGB-Heldout-v0"):
        p = f"logs/eval/il/{task}_{tag}.json"
        if os.path.exists(p):
            rgb.setdefault(tag, {})[task] = ev(p)
il["langnav_rgb"] = rgb
# SmolVLA (LeRobot) fine-tuned on the re-collected RGB demonstrations, evaluated through the policy server
vla = {}
for tag in ("smolvla_v2data_h10", "smolvla_v2data_h50", "smolvla_v2data_h1", "smolvla_v2data_5k_h10",
            "smolvla_v2data_noprompt_h10"):
    for task in ("Quadsim-LangNav-RGB-v0", "Quadsim-LangNav-RGB-Heldout-v0"):
        p = f"logs/eval/vla/{task}_{tag}.json"
        if os.path.exists(p):
            vla.setdefault(tag, {})[task] = ev(p)
if vla:
    srv = "logs/eval/vla/smolvla_v2data_server.json"
    vla["server_metadata"] = json.load(open(srv)) if os.path.exists(srv) else None
    lines = open("logs/phase_f.log").read().splitlines() if os.path.exists("logs/phase_f.log") else []
    vla["training"] = {"probe": next((ln for ln in lines if "probe:" in ln), None),
                       "final": next((ln for ln in reversed(lines) if "trained:" in ln), None),
                       "log": "logs/lerobot/train_smolvla_langnav_v2.log"}
    il["langnav_rgb_vla"] = vla
il["datasets"] = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob("datasets/il/*.h5.json"))}
il["lerobot_exports"] = {p.split("/")[-2]: json.load(open(p))
                         for p in sorted(glob.glob("datasets/lerobot/*/quadsim_export.json"))}
dump("imitation.json", il)

# ---- examples, regression
p = "logs/validation/examples_check.json"
dump("examples_check.json", json.load(open(p)) if os.path.exists(p) else missing.append(p))
reg = {}
for name in ("regress_before", "regress_after_f1", "regress_final"):
    files = sorted(glob.glob(f"logs/eval/{name}/*.json"))
    if files:
        reg[name] = {os.path.basename(f): json.load(open(f))["controllers"] for f in files}
if "regress_before" in reg:
    ref = reg["regress_before"]
    for name, cur in list(reg.items()):
        if name == "regress_before":
            continue
        diffs = [abs(v["mean"] - cur[f][c][k]["mean"]) for f in ref if f in cur for c in ref[f]
                 for k, v in ref[f][c].items() if isinstance(v, dict) and "mean" in v and v["mean"] == v["mean"]]
        reg[f"{name}_vs_before"] = {"metrics": len(diffs), "max_abs_diff": max(diffs) if diffs else None}
dump("regression.json", {k: v for k, v in reg.items() if k.endswith("_vs_before")})
dump("_missing.json", missing)
print(f"[collect] wrote {OUT}/ ; missing: {missing}")
