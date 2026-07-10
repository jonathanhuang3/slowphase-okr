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
    end_time: float | None
    contrast_level: float | None
    direction: str | None
    dot_color: str | None
    use_persistent_dots: bool | None
    is_anchor100: bool | None
    threshold_multiplier: float | None
    label: str


@dataclass(frozen=True)
class OkrLogFixationMarker:
    event_index: int
    event_type: str
    block_index: int | None
    start_time: float
    end_time: float | None


@dataclass
class OkrLog:
    source_path: str
    block_markers: list[OkrLogBlockMarker]
    fixation_markers: list[OkrLogFixationMarker]
    stimulus_eye_patch: str | None = None
    stimulus_name: str | None = None


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


def _parse_optional_bool01(text: str) -> bool | None:
    text = text.strip()
    if not text or text.upper() == "NA":
        return None
    if text in ("0", "1"):
        return text == "1"
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    return None


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


def _parse_header_metadata(lines: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        body = stripped[1:].strip()
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


def _session_tags_from_stimulus_name(stimulus_name: str | None) -> list[str]:
    """Extract Increment/Decrement, White/Black, etc. from StimulusName."""
    if not stimulus_name:
        return []
    upper = stimulus_name.upper()
    tags: list[str] = []
    if "INCREMENT" in upper:
        tags.append("Increment")
    elif "DECREMENT" in upper:
        tags.append("Decrement")
    if "WHITE" in upper:
        tags.append("White dots")
    elif "BLACK" in upper:
        tags.append("Black dots")
    return tags


def _flicker_label(marker: OkrLogBlockMarker) -> str | None:
    event_lower = marker.event_type.lower()
    if "persistent" in event_lower:
        return "Persistent (non-flicker)"
    if "flicker" in event_lower:
        return "Flicker"
    if marker.use_persistent_dots is True:
        return "Persistent (non-flicker)"
    if marker.use_persistent_dots is False:
        return "Flicker"
    return None


def format_block_condition(marker: OkrLogBlockMarker) -> str:
    """Human-readable condition string for one stimulus block."""
    parts: list[str] = []
    if marker.block_index is not None:
        parts.append(f"B{marker.block_index}")
    else:
        parts.append(marker.event_type)

    if marker.contrast_level is not None:
        if marker.is_anchor100:
            parts.append(f"contrast {marker.contrast_level:g} (anchor 100%)")
        elif marker.threshold_multiplier is not None:
            parts.append(
                f"contrast {marker.contrast_level:g} ({marker.threshold_multiplier:g}×T)"
            )
        else:
            parts.append(f"contrast {marker.contrast_level:g}")
    elif marker.is_anchor100:
        parts.append("anchor 100%")

    if marker.direction in ("Up", "Down"):
        parts.append(marker.direction)

    flicker = _flicker_label(marker)
    if flicker:
        parts.append(flicker)

    if marker.dot_color and marker.dot_color.upper() != "NA":
        parts.append(f"{marker.dot_color} dots")

    return " · ".join(parts)


def format_fixation_condition(marker: OkrLogFixationMarker) -> str:
    if marker.event_type == "InitialFixation":
        return "Initial fixation"
    if marker.event_type == "FixationITI":
        if marker.block_index is not None:
            return f"Fixation ITI (after B{marker.block_index})"
        return "Fixation ITI"
    return marker.event_type


def condition_at_time(okr_log: OkrLog, t: float) -> str:
    """Describe the stimulus/fixation condition covering time ``t``."""
    session = _session_tags_from_stimulus_name(okr_log.stimulus_name)
    session_prefix = f"Session: {' · '.join(session)} | " if session else ""

    for marker in okr_log.block_markers:
        end = marker.end_time if marker.end_time is not None else float("inf")
        if marker.start_time <= t <= end:
            return session_prefix + format_block_condition(marker)

    for marker in okr_log.fixation_markers:
        end = marker.end_time if marker.end_time is not None else float("inf")
        if marker.start_time <= t <= end:
            return session_prefix + format_fixation_condition(marker)

    # Nearest preceding event if between logged intervals
    preceding_blocks = [m for m in okr_log.block_markers if m.start_time <= t]
    preceding_fix = [m for m in okr_log.fixation_markers if m.start_time <= t]
    candidates: list[tuple[float, str]] = []
    if preceding_blocks:
        m = max(preceding_blocks, key=lambda x: x.start_time)
        candidates.append((m.start_time, format_block_condition(m) + " (ended)"))
    if preceding_fix:
        m = max(preceding_fix, key=lambda x: x.start_time)
        candidates.append((m.start_time, format_fixation_condition(m) + " (ended)"))
    if candidates:
        _start, text = max(candidates, key=lambda item: item[0])
        return session_prefix + text

    if session:
        return session_prefix + "No block/fixation at this time"
    return "No OKR condition at this time"


def load_okr_log(path: str | Path) -> OkrLog:
    """Load OKR_Log_*.txt (tab-separated Unity stimulus event log)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    raw_lines = path.read_text().splitlines()
    meta = _parse_header_metadata(raw_lines)
    stimulus_name = meta.get("StimulusName") or None
    header_eye_patch = meta.get("StimulusEyePatch") or None

    header: list[str] | None = None
    rows: list[dict[str, str]] = []

    for line in raw_lines:
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
    stimulus_eye_patch: str | None = header_eye_patch

    for row in rows:
        eye_patch = row.get("eyePatch", "").strip()
        if eye_patch and eye_patch.upper() != "NA" and stimulus_eye_patch is None:
            stimulus_eye_patch = eye_patch

        event_type = row["eventType"].strip()
        event_index = int(row["eventIndex"])
        block_index = _parse_optional_int(row.get("contrastBlockIndex", ""))
        start_time = float(row["startTime"])
        end_time = _parse_optional_float(row.get("endTime", ""))
        direction = row.get("direction", "").strip() or None
        if direction == "NA":
            direction = None
        contrast_level = _parse_optional_float(row.get("contrastLevel", ""))
        dot_color = row.get("dotColor", "").strip() or None
        if dot_color == "NA":
            dot_color = None
        use_persistent_dots = _parse_optional_bool01(row.get("usePersistentDots", ""))
        is_anchor100 = _parse_optional_bool01(row.get("isAnchor100", ""))
        threshold_multiplier = _parse_optional_float(row.get("thresholdMultiplier", ""))

        if _is_fixation_event(event_type):
            fixation_markers.append(
                OkrLogFixationMarker(
                    event_index=event_index,
                    event_type=event_type,
                    block_index=block_index,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
        else:
            block_markers.append(
                OkrLogBlockMarker(
                    event_index=event_index,
                    event_type=event_type,
                    block_index=block_index,
                    start_time=start_time,
                    end_time=end_time,
                    contrast_level=contrast_level,
                    direction=direction,
                    dot_color=dot_color,
                    use_persistent_dots=use_persistent_dots,
                    is_anchor100=is_anchor100,
                    threshold_multiplier=threshold_multiplier,
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
        stimulus_name=stimulus_name,
    )
