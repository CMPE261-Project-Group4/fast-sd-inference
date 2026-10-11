"""Load experiment files: a base YAML plus a list of per-run overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of ``base`` with ``override`` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_experiment(path: str | Path, root: str | Path = ".") -> list[dict[str, Any]]:
    """Expand an experiment file into a list of fully-resolved run configs."""
    exp = load_yaml(path)
    base = load_yaml(Path(root) / exp["base"])
    runs = []
    for override in exp["runs"]:
        if "name" not in override:
            raise ValueError(f"Every run in {path} needs a 'name'")
        runs.append(deep_merge(base, override))
    names = [r["name"] for r in runs]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate run names in {path}: {names}")
    return runs
