"""Load and expose settings.yaml as a typed config object."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class Settings:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            val = self._data[name]
            return Settings(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(f"No config key: {name!r}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __repr__(self) -> str:
        return f"Settings({self._data!r})"


def load_settings(path: Path | str | None = None, overrides: dict | None = None) -> Settings:
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        data = yaml.safe_load(f)
    if overrides:
        data = _deep_merge(data, overrides)
    return Settings(data)


# Module-level singleton — import this everywhere
settings = load_settings()
