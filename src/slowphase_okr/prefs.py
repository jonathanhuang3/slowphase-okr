"""Persisted app preferences (personal markings folder, etc.)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREFS_FILENAME = ".slowphase_okr_prefs.json"

# Defaults match a typical 27" 16:9 at ~1 m (same as eyelink_asc defaults).
DEFAULT_MONITOR_WIDTH_CM = 60.0
DEFAULT_MONITOR_HEIGHT_CM = 33.5
DEFAULT_VIEWING_DISTANCE_CM = 100.0


def prefs_path() -> Path:
    return Path.home() / PREFS_FILENAME


def load_prefs() -> dict[str, Any]:
    path = prefs_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_prefs(prefs: dict[str, Any]) -> None:
    path = prefs_path()
    path.write_text(json.dumps(prefs, indent=2))


def get_annotations_dir(prefs: dict[str, Any] | None = None) -> Path | None:
    data = prefs if prefs is not None else load_prefs()
    raw = data.get("annotations_dir")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return Path(raw).expanduser()


def set_annotations_dir(directory: str | Path) -> Path:
    path = Path(directory).expanduser().resolve()
    prefs = load_prefs()
    prefs["annotations_dir"] = str(path)
    save_prefs(prefs)
    return path


def get_monitor_geometry_prefs(
    prefs: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return monitor width/height/distance (cm) from prefs or lab defaults."""
    data = prefs if prefs is not None else load_prefs()

    def _float(key: str, default: float) -> float:
        raw = data.get(key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    return {
        "width_cm": _float("monitor_width_cm", DEFAULT_MONITOR_WIDTH_CM),
        "height_cm": _float("monitor_height_cm", DEFAULT_MONITOR_HEIGHT_CM),
        "distance_cm": _float("viewing_distance_cm", DEFAULT_VIEWING_DISTANCE_CM),
    }


def set_monitor_geometry_prefs(
    *,
    width_cm: float,
    height_cm: float,
    distance_cm: float,
) -> None:
    if width_cm <= 0 or height_cm <= 0 or distance_cm <= 0:
        raise ValueError("Monitor width, height, and distance must be > 0.")
    prefs = load_prefs()
    prefs["monitor_width_cm"] = float(width_cm)
    prefs["monitor_height_cm"] = float(height_cm)
    prefs["viewing_distance_cm"] = float(distance_cm)
    save_prefs(prefs)