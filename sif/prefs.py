"""Small persistent preferences file, for the handful of choices worth keeping.

Deliberately tiny: a JSON file in the user's configuration directory holding
things like "do not offer version 2.1.0 again". Anything that matters for safety
belongs in the log or the database, not here, so a corrupt or missing file is
never an error - it simply reads as empty.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict

__all__ = ["config_directory", "load", "save", "get", "set_value"]

LOGGER = logging.getLogger(__name__)
FILE_NAME = "settings.json"
APP_DIRECTORY = "SIF Insight Console"


def config_directory() -> str:
    """Per-user configuration directory, following each platform's convention."""
    if sys.platform.startswith("win"):
        root = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        root = os.path.expanduser("~/Library/Application Support")
    else:
        root = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(root, APP_DIRECTORY)


def _path() -> str:
    return os.path.join(config_directory(), FILE_NAME)


def load() -> Dict[str, Any]:
    """Read the preferences; an unreadable file reads as empty."""
    try:
        with open(_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 - corrupt file must not stop start-up
        LOGGER.warning("Ignoring unreadable preferences (%s)", exc)
        return {}


def save(values: Dict[str, Any]) -> bool:
    """Write the preferences; returns False when the location is not writable."""
    try:
        os.makedirs(config_directory(), exist_ok=True)
        with open(_path(), "w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2)
        return True
    except Exception as exc:  # noqa: BLE001 - a read-only home is not fatal
        LOGGER.warning("Could not save preferences (%s)", exc)
        return False


def get(key: str, default: Any = None) -> Any:
    """One preference."""
    return load().get(key, default)


def set_value(key: str, value: Any) -> bool:
    """Update one preference in place."""
    values = load()
    values[key] = value
    return save(values)
