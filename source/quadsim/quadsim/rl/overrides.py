"""Minimal ``env.a.b=value`` / ``agent.a.b=value`` command-line overrides for configclass objects."""

from __future__ import annotations

import ast


def is_override(item: str) -> bool:
    return "=" in item and item.split("=", 1)[0].split(".", 1)[0] in ("env", "agent")


def split_overrides(args, overrides: list[str], argv: list[str], list_options=("plugins", "controllers")) -> list[str]:
    """Overrides in command-line order, including those that argparse attached to a list option.

    ``nargs="*"``/``"+"`` options swallow the positional ``env.x=v`` items that follow them
    (``--plugins my.py env.control.mode=x``); move such items back to the override list (in place on ``args``).
    """
    found = set(overrides)
    for name in list_options:
        values = getattr(args, name, None)
        if values:
            found.update(v for v in values if is_override(v))
            setattr(args, name, [v for v in values if not is_override(v)])
    return [a for a in argv if a in found]


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
        print(f"[override] {key} = {getattr(obj, path[-1])!r}", flush=True)
