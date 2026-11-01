"""Rebuild the curated core evaluation files docs/results/eval_*.json (sessions 1-3) from logs/eval/.

Usage: python scripts/tools/collect_core_results.py
Every curated file is a copy, a controller subset or a merge of evaluate.py outputs (see SOURCES). Entries that can no
longer be re-run (code that no longer exists) are kept from the previous curated file and marked as such.
"""

import json
import os

R, L = "docs/results", "logs/eval"
H = "Quadsim-Hover-v0"
# curated file -> list of (source file, {source controller: curated controller}) and optional extra fields
SOURCES = {
    "eval_hover_nominal.json": [(f"{H}_nominal.json", None)],
    "eval_hover_wind_push.json": [(f"{H}_wind_push.json", None)],
    "eval_track_nominal.json": [("Quadsim-Track-v0_nominal.json", None)],
    "eval_track_moderate.json": [("Quadsim-Track-v0_moderate.json", None)],
    "eval_track_geometric_v2.json": [("Quadsim-Track-v0_geometric_v2.json", None)],
    "eval_waypoint_nominal.json": [("Quadsim-Waypoint-v0_nominal.json", None)],
    "eval_nav_1_pf_paired.json": [("Quadsim-Nav-Range-v0_range_v2.json", {"potential_field": "potential_field"})],
    "eval_nav_2_range_v1.json": [("Quadsim-Nav-Range-v0_range_v1.json", None)],
    "eval_nav_3_range_v2.json": [("Quadsim-Nav-Range-v0_range_v2.json", {"policy": "policy"})],
    "eval_nav_4a_depth_v2_raycast.json": [("Quadsim-Nav-Depth-v0_depth_v2_raycast.json", None)],
    "eval_nav_4b_depth_v2_rendered.json": [("Quadsim-Nav-Depth-Render-v0_depth_v2_rendered.json", None)],
    "eval_nav_5_pf_independent_run.json": [("Quadsim-Nav-Range-v0_pf_full.json", None)],
}
for i, c in enumerate(("ideal", "randomized", "harsh"), 1):
    SOURCES[f"eval_dr_ablation_{i}_{c}.json"] = [(f"{H}_ablation_{c}_DR.json", {"policy": "policy_DR"}),
                                               (f"{H}_ablation_{c}_noDR.json", {"policy": "policy_noDR"})]
KEEP = {"eval_track_nominal.json": ["geometric_v1"]}  # first SE(3) design: its code was replaced (PROGRESS M4)
KEEP_NOTE = ("measured with the earlier evaluator (before the per-environment episode quota) - the controller's "
             "code no longer exists, so it was not re-measured")


def main():
    missing = []
    for name, parts in SOURCES.items():
        old = json.load(open(os.path.join(R, name))) if os.path.exists(os.path.join(R, name)) else {}
        out, ckpts = None, {}
        for src, rename in parts:
            p = os.path.join(L, src)
            if not os.path.exists(p):
                missing.append(p)
                continue
            d = json.load(open(p))
            ctrls = {(rename or {}).get(c, c): v for c, v in d["controllers"].items() if rename is None or c in rename}
            if out is None:
                out = {k: v for k, v in d.items() if k != "controllers"}
                out["controllers"] = {}
            out["controllers"].update(ctrls)
            for c in ctrls:
                ckpts[c] = d.get("checkpoint")
            out.setdefault("sources", []).append(p)
        if out is None:
            continue
        if len(parts) > 1:
            out["checkpoint"] = ckpts
        for k in ("label", "note"):
            if k in old:
                out[k] = old[k]
        for c in KEEP.get(name, []):
            if c in old.get("controllers", {}):
                out["controllers"][c] = old["controllers"][c]
                out.setdefault("not_re_measured", {})[c] = KEEP_NOTE
        with open(os.path.join(R, name), "w") as f:
            json.dump(out, f, indent=2)
    print(f"[core] rebuilt {len(SOURCES)} curated files; missing sources: {missing}")


if __name__ == "__main__":
    main()
