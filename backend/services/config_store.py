"""
Persistent config store — reads/writes backend/config.json.

Only the trim section is used today; the file is designed to grow.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(os.getenv("CONFIG_FILE", Path(__file__).parent.parent / "config.json"))

_DEFAULTS: dict[str, Any] = {
    "trim": {
        "roll":     0,
        "pitch":    0,
        "yaw":      0,
        "throttle": 0,
    }
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return dict(_DEFAULTS)
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _deep_merge(_DEFAULTS, data)
    except Exception as exc:
        logger.warning("[config] Failed to load %s: %s — using defaults", _CONFIG_PATH, exc)
        return dict(_DEFAULTS)


def save(data: dict[str, Any]) -> None:
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("[config] Saved to %s", _CONFIG_PATH)
    except Exception as exc:
        logger.error("[config] Failed to save %s: %s", _CONFIG_PATH, exc)


# ── Singleton in-memory config ────────────────────────────────────────────────

_config: dict[str, Any] = {}


def init() -> None:
    global _config
    _config = load()
    logger.info("[config] Loaded: %s", _config)


def get(section: str) -> Any:
    return _config.get(section, _DEFAULTS.get(section))


def update_and_save(section: str, values: dict) -> None:
    global _config
    _config[section] = {**_config.get(section, {}), **values}
    save(_config)
