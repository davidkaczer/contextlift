"""Defense registry.

Every module in this package is imported automatically, so a defense
registers itself just by carrying the @register_defense decorator — no
factory edits needed. Canonical names (the single source of truth is each
class's `name` attribute; the README table mirrors it):

    none · interleaving
"""

from __future__ import annotations

import importlib
import pkgutil

from emabench.defenses.base import Defense, register_defense, registered_defenses

# Import every sibling module so @register_defense decorators run.
for _mod in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_mod.name}")


def make_defense(cfg) -> Defense:
    name = cfg.defense.name.lower()
    registry = registered_defenses()
    cls = registry.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown defense.name: '{cfg.defense.name}'. "
            f"Available: {sorted(registry)}"
        )
    return cls(cfg)
