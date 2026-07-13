"""Persisted app preferences (personal markings folder, etc.)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREFS_FILENAME = ".slowphase_okr_prefs.json"


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
