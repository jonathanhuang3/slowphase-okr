"""JSON autosave for in-progress trial annotations."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from slowphase_okr.fit import SegmentFit

AUTOSAVE_FILENAME_SUFFIX = "_slowphase_okr_autosave.json"


def autosave_path(trial_dir: str | Path, trial_id: str) -> Path:
    """Default autosave location beside the trial gaze files."""
    return Path(trial_dir) / f"{trial_id}{AUTOSAVE_FILENAME_SUFFIX}"


def segment_to_dict(segment: SegmentFit) -> dict[str, Any]:
    return asdict(segment)


def segment_from_dict(data: dict[str, Any]) -> SegmentFit:
    return SegmentFit(**data)


def save_autosave(
    path: str | Path,
    *,
    trial_id: str,
    gaze_source: str,
    time_source: str,
    stimulus_velocity: float,
    segments: list[SegmentFit],
    software_version: str,
    signal_mode: str = "elevation",
) -> Path:
    """Write annotation state to JSON."""
    path = Path(path)
    payload = {
        "trial_id": trial_id,
        "gaze_source": gaze_source,
        "time_source": time_source,
        "stimulus_velocity": stimulus_velocity,
        "signal_mode": signal_mode,
        "software_version": software_version,
        "segments": [segment_to_dict(s) for s in segments],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_autosave(path: str | Path) -> dict[str, Any] | None:
    """Load autosave payload, or None if missing / invalid."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def segments_from_autosave(data: dict[str, Any]) -> list[SegmentFit]:
    raw = data.get("segments", [])
    if not isinstance(raw, list):
        return []
    return [segment_from_dict(item) for item in raw]


def autosave_matches_trial(
    data: dict[str, Any],
    gaze_source: str,
    time_source: str,
) -> bool:
    """True when autosave refers to the same gaze/time files."""
    return (
        str(data.get("gaze_source", "")) == str(gaze_source)
        and str(data.get("time_source", "")) == str(time_source)
    )
