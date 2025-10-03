"""Configuration loading utilities supporting YAML/TOML and env overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping

import yaml

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for Python <3.11
    import tomli as tomllib  # type: ignore

from .logging_conf import get_logger

LOGGER = get_logger(__name__)


def _read_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text()) or {}
    if suffix == ".toml":
        return tomllib.loads(path.read_text())
    if suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported config format: {path}")


def _merge_dicts(base: MutableMapping[str, Any], override: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, MutableMapping) and isinstance(base.get(key), MutableMapping):
            _merge_dicts(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def _apply_env_overrides(config: Dict[str, Any], prefix: str | None = None) -> None:
    def recurse(path: Iterable[str], node: Dict[str, Any]) -> None:
        for key, value in list(node.items()):
            env_key = "_".join([*(p.upper() for p in path), key.upper()]) if path else key.upper()
            env_value = os.getenv(env_key)
            if env_value is not None:
                LOGGER.debug("Overriding config key %s with env var %s", key, env_key)
                node[key] = _coerce_type(env_value, value)
            if isinstance(node[key], dict):
                recurse([*path, key], node[key])

    recurse(list(prefix or ""), config)


def _coerce_type(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        try:
            return int(value)
        except ValueError:
            return current
    if isinstance(current, float):
        try:
            return float(value)
        except ValueError:
            return current
    if isinstance(current, (list, tuple)):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load a configuration file supporting nested includes and env overrides."""

    path = Path(path)
    config = _read_file(path)

    includes = config.pop("include", []) or []
    if not isinstance(includes, list):
        raise ValueError("include must be a list of config file paths")

    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_path = (path.parent / inc).resolve()
        merged = _merge_dicts(merged, load_config(inc_path))

    config = _merge_dicts(merged, config)

    if overrides:
        config = _merge_dicts(config, overrides)

    _apply_env_overrides(config)

    return config


def resolve_path(base: str | Path, *segments: str) -> Path:
    """Resolve a path relative to a base directory."""

    base_path = Path(base).expanduser().resolve()
    return base_path.joinpath(*segments).resolve()
