"""Name -> component registries: the framework's extension points.

Every pluggable component type has one registry. Built-in components register themselves on import; research code
registers its own components the same way and selects them by name from a configuration::

    from quadsim.registry import ACTION_INTERFACES

    @ACTION_INTERFACES.register("my_interface")
    class MyInterface(ActionInterface): ...

    cfg.control.mode = "my_interface"

Registries (``quadsim.registry``):
    VEHICLES            vehicle factories ``() -> QuadrotorParams`` (``cfg.dynamics.preset``)
    ACTION_INTERFACES   policy action -> rotor thrust command (``cfg.control.mode``)
    ALGORITHMS          training back-ends (``scripts/rl/train.py --algo``)
    BASELINES           classical controllers acting through the policy action interface (``evaluate.py``)
    BENCHMARKS          benchmark suites (``scripts/benchmark/run.py``)

Plug-in modules (anything that registers components) are imported with ``--plugins pkg.module`` on the command line
of every quadsim script, or :func:`import_plugins`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from typing import Any


class Registry:
    """A named mapping with decorator-style registration and helpful lookup errors."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str, obj: Any = None, *, overwrite: bool = False) -> Callable | Any:
        """``REG.register("name", obj)`` or ``@REG.register("name")``."""

        def _add(o):
            if name in self._items and not overwrite and self._items[name] is not o:
                raise KeyError(f"{self.kind} '{name}' is already registered (pass overwrite=True to replace it)")
            self._items[name] = o
            return o

        return _add if obj is None else _add(obj)

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} '{name}'. Registered: {sorted(self._items)}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry({self.kind}: {self.names()})"


VEHICLES = Registry("vehicle")
ACTION_INTERFACES = Registry("action interface")
ALGORITHMS = Registry("algorithm")
BASELINES = Registry("baseline controller")
BENCHMARKS = Registry("benchmark suite")


def import_plugins(modules: list[str] | tuple[str, ...] | None) -> list[str]:
    """Import user modules that register components (``pkg.module`` or a ``path/to/file.py``)."""
    loaded = []
    for m in modules or ():
        if m.endswith(".py"):
            import importlib.util
            import os
            import sys

            name = "quadsim_plugin_" + os.path.splitext(os.path.basename(m))[0]
            spec = importlib.util.spec_from_file_location(name, m)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        else:
            importlib.import_module(m)
        loaded.append(m)
    return loaded
