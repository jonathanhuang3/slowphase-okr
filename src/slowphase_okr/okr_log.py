"""Parse Unity OKR condition logs for stimulus timing markers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OkrLogBlockMarker:
    event_index: int
    event_type: str
    block_index: int | None
    start_time: float
    contrast_level: float | None
    direction: str | None
    label: str


@dataclass(frozen=True)
class OkrLogFixationMarker:
    event_index: int
    event_type: str
    block_index: int | None
    start_time: float


@dataclass
class OkrLog:
    source_path: str
    block_markers: list[OkrLogBlockMarker]
    fixation_markers: list[OkrLogFixationMarker]
    stimulus_eye_patch: str | None = None


def _is_fixation_event(event_type: str) -> bool:
    return "fixation" in event_type.lower()


def _parse_optional_int(text: str) -> int | None:
    text = text.strip()
    if not text or text.upper() == "NA":
        return None
    return int(text)


def _parse_optional_float(text: str) -> float | None:
    text = text.strip()
    if not text or text.upper() == "NA":
        return None
    return float(text)


def _block_label(
    block_index: int | None,
    direction: str | None,
    event_type: str,
) -> str:
    if block_index is not None:
        prefix = f"B{block_index}"
    else:
        prefix = event_type.removesuffix("Block") if event_type.endswith("Block") else event_type
    if direction in ("Up", "Down"):
        arrow = "↑" if direction == "Up" else "↓"
        return f"{prefix}{arrow}"
    return prefix


def load_okr_log(path: str | Path) -> OkrLog:
    """Load OKR_Log_*.txt (tab-separated Unity stimulus event log)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    header: list[str] | None = None
    rows: list[dict[str, str]] = []

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if header is None:
            header = parts
            continue
        if len(parts) < len(header):
            parts.extend([""] * (len(header) - len(parts)))
        rows.append(dict(zip(header, parts)))

    if header is None:
        raise ValueError(f"No event table found in {path}")

    required = {"eventIndex", "eventType", "contrastBlockIndex", "startTime"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"OKR log missing columns: {', '.join(sorted(missing))}")

    block_markers: list[OkrLogBlockMarker] = []
    fixation_markers: list[OkrLogFixationMarker] = []
    stimulus_eye_patch: str | None = None

    for row in rows:
        eye_patch = row.get("eyePatch", "").strip()
        if eye_patch and eye_patch.upper() != "NA" and stimulus_eye_patch is None:
            stimulus_eye_patch = eye_patch

        event_type = row["eventType"].strip()
        event_index = int(row["eventIndex"])
        block_index = _parse_optional_int(row.get("contrastBlockIndex", ""))
        start_time = float(row["startTime"])
        direction = row.get("direction", "").strip() or None
        if direction == "NA":
            direction = None
        contrast_level = _parse_optional_float(row.get("contrastLevel", ""))

        if _is_fixation_event(event_type):
            fixation_markers.append(
                OkrLogFixationMarker(
                    event_index=event_index,
                    event_type=event_type,
                    block_index=block_index,
                    start_time=start_time,
                )
            )
        else:
            block_markers.append(
                OkrLogBlockMarker(
                    event_index=event_index,
                    event_type=event_type,
                    block_index=block_index,
                    start_time=start_time,
                    contrast_level=contrast_level,
                    direction=direction,
                    label=_block_label(block_index, direction, event_type),
                )
            )

    if not block_markers and not fixation_markers:
        raise ValueError(f"No contrast blocks or fixation events found in {path}")

    return OkrLog(
        source_path=str(path.resolve()),
        block_markers=block_markers,
        fixation_markers=fixation_markers,
        stimulus_eye_patch=stimulus_eye_patch,
    )
