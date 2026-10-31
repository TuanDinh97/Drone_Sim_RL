"""Minimal ``env.a.b=value`` / ``agent.a.b=value`` command-line overrides for configclass objects."""

from __future__ import annotations

import ast


def _parse(v: str):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def apply_overrides(env_cfg, agent_cfg, overrides: list[str]):
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"bad override '{item}' (expected env.x.y=value or agent.x.y=value)")
        key, val = item.split("=", 1)
        root, *path = key.split(".")
        obj = {"env": env_cfg, "agent": agent_cfg}[root]
        for p in path[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, path[-1]):
            raise AttributeError(f"override '{key}': no field '{path[-1]}'")
        setattr(obj, path[-1], _parse(val))
        print(f"[override] {key} = {getattr(obj, path[-1])!r}")
