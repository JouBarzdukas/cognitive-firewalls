"""YAML experiment configs. Every config inherits from another via `base:`; `main` is the root.

Overrides may only set keys that already exist in the base, so typos fail loudly.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _merge(base: dict, override: dict, where: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key not in base:
            raise KeyError(f"unknown config key '{where}{key}'")
        if isinstance(value, dict) and isinstance(base[key], dict):
            out[key] = _merge(base[key], value, f"{where}{key}.")
        else:
            out[key] = value
    return out


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        options = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no config '{name}'; options: {options}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("base", None)
    cfg = _merge(load_config(parent), raw) if parent else raw
    cfg["name"] = name
    return cfg
