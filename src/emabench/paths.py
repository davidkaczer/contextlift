"""
Maybe merge it to an utils.py later
"""

from __future__ import annotations

from pathlib import Path


def find_project_root(config_path: Path) -> Path:
    """Resolve emabench project root for a given config path."""
    # Standard case: emabench/configs/<config>.yaml
    if config_path.parent.name == "configs":
        return config_path.parent.parent

    # Seed-sweep case: runs/<prefix>/seedX/config.yaml
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "data").exists() and (parent / "evaluation").exists():
            return parent

    # Fallback: use config's parent
    return config_path.parent
