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
    dot_start_size: float | None = None
    emission_rate: float | None = None


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
    if direction in ("Left", "Right"):
        arrow = "←" if direction == "Left" else "→"
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


def resolve_viewing_eye(
    *,
    eye_patch: str | None = None,
    stimulus_name: str | None = None,
) -> str | None:
    """Which eye was shown the dots (unpatched / viewing), or None if unknown.

    Prefers StimulusEyePatch / eyePatch (patched side → opposite viewing eye).
    Falls back to LE/RE tokens in StimulusName.
    """
    if eye_patch:
        patch = eye_patch.strip().lower()
        if patch == "left":
            return "right"
        if patch == "right":
            return "left"

    if stimulus_name:
        normalized = f" {stimulus_name.replace('_', ' ').upper()} "
        if " LE " in normalized:
            return "left"
        if " RE " in normalized:
            return "right"
        # Older stimulus names used bare Left/Right (e.g. "... Left Dot Size Testing")
        if " LEFT " in normalized:
            return "left"
        if " RIGHT " in normalized:
            return "right"
    return None


def format_viewing_eye_label(eye: str | None) -> str | None:
    if not eye:
        return None
    return f"Dots → {eye.capitalize()} eye"


def _session_tags(okr_log: OkrLog) -> list[str]:
    """Session-level tags for the condition readout (eye, Increment/Decrement, etc.)."""
    tags: list[str] = []
    viewing = resolve_viewing_eye(
        eye_patch=okr_log.stimulus_eye_patch,
        stimulus_name=okr_log.stimulus_name,
    )
    eye_label = format_viewing_eye_label(viewing)
    if eye_label:
        tags.append(eye_label)

    if not okr_log.stimulus_name:
        return tags
    upper = okr_log.stimulus_name.upper()
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

    # Legacy TrialIndex logs vary size/rate at fixed contrast — lead with those.
    legacy_trial = marker.event_type == "TrialBlock"
    if legacy_trial:
        if marker.dot_start_size is not None:
            parts.append(f"size {marker.dot_start_size:g}")
        if marker.emission_rate is not None:
            parts.append(f"rate {marker.emission_rate:g}")

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

    if marker.direction in ("Up", "Down", "Left", "Right"):
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


def block_at_time(okr_log: OkrLog, t: float) -> OkrLogBlockMarker | None:
    """Return the stimulus block covering time ``t``, if any."""
    for marker in okr_log.block_markers:
        end = marker.end_time if marker.end_time is not None else float("inf")
        if marker.start_time <= t <= end:
            return marker
    return None


def segment_condition_fields(
    okr_log: OkrLog | None,
    t_start: float,
    t_end: float,
) -> dict[str, object]:
    """Condition metadata for a segment using its midpoint time."""
    fields: dict[str, object] = {
        "block_label": "Outside blocks",
        "block_index": None,
        "event_type": None,
        "direction": None,
        "contrast_level": None,
        "threshold_multiplier": None,
        "is_anchor100": None,
        "flicker_mode": None,
        "dot_color": None,
        "dot_start_size": None,
        "emission_rate": None,
        "eye_patch": None,
        "viewing_eye": None,
        "condition": "Outside blocks",
        "session_tags": "",
    }
    if okr_log is None:
        fields["block_label"] = "No OKR log"
        fields["condition"] = "No OKR log"
        return fields

    viewing = resolve_viewing_eye(
        eye_patch=okr_log.stimulus_eye_patch,
        stimulus_name=okr_log.stimulus_name,
    )
    session = _session_tags(okr_log)
    fields["session_tags"] = " · ".join(session)
    fields["eye_patch"] = okr_log.stimulus_eye_patch
    fields["viewing_eye"] = viewing
    t_mid = (t_start + t_end) / 2.0
    marker = block_at_time(okr_log, t_mid)
    if marker is None:
        return fields

    flicker = _flicker_label(marker)
    fields.update(
        {
            "block_label": marker.label,
            "block_index": marker.block_index,
            "event_type": marker.event_type,
            "direction": marker.direction,
            "contrast_level": marker.contrast_level,
            "threshold_multiplier": marker.threshold_multiplier,
            "is_anchor100": marker.is_anchor100,
            "flicker_mode": flicker,
            "dot_color": marker.dot_color,
            "dot_start_size": marker.dot_start_size,
            "emission_rate": marker.emission_rate,
            "condition": format_block_condition(marker),
        }
    )
    return fields


def condition_at_time(okr_log: OkrLog, t: float) -> str:
    """Describe the stimulus/fixation condition covering time ``t``."""
    session = _session_tags(okr_log)
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


def _read_okr_table(raw_lines: list[str]) -> tuple[list[str], list[dict[str, str]]]:
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
        raise ValueError("No event table found")
    return header, rows


def _direction_or_none(text: str | None) -> str | None:
    if not text:
        return None
    direction = text.strip()
    if not direction or direction.upper() == "NA":
        return None
    return direction


def _dot_color_or_none(text: str | None) -> str | None:
    if not text:
        return None
    color = text.strip()
    if not color or color.upper() == "NA":
        return None
    return color


def _load_legacy_trial_okr_log(
    path: Path,
    *,
    stimulus_name: str | None,
    header_eye_patch: str | None,
    header: list[str],
    rows: list[dict[str, str]],
) -> OkrLog:
    """Parse older TrialIndex logs (dot-size / emission-rate schedules).

    Columns look like:
    TrialIndex, dotColor, direction, contrastLevel, dotStartSize, emissionRate,
    startTime, endTime
    """
    required = {"TrialIndex", "startTime"}
    missing = required - set(header)
    if missing:
        raise ValueError(
            f"Legacy OKR log missing columns: {', '.join(sorted(missing))}"
        )

    block_markers: list[OkrLogBlockMarker] = []
    for row in rows:
        trial_index = int(row["TrialIndex"])
        block_index = trial_index - 1
        direction = _direction_or_none(row.get("direction"))
        event_type = "TrialBlock"
        block_markers.append(
            OkrLogBlockMarker(
                event_index=trial_index,
                event_type=event_type,
                block_index=block_index,
                start_time=float(row["startTime"]),
                end_time=_parse_optional_float(row.get("endTime", "")),
                contrast_level=_parse_optional_float(row.get("contrastLevel", "")),
                direction=direction,
                dot_color=_dot_color_or_none(row.get("dotColor")),
                use_persistent_dots=None,
                is_anchor100=None,
                threshold_multiplier=None,
                label=_block_label(block_index, direction, event_type),
                dot_start_size=_parse_optional_float(row.get("dotStartSize", "")),
                emission_rate=_parse_optional_float(row.get("emissionRate", "")),
            )
        )

    if not block_markers:
        raise ValueError(f"No trial blocks found in {path}")

    return OkrLog(
        source_path=str(path.resolve()),
        block_markers=block_markers,
        fixation_markers=[],
        stimulus_eye_patch=header_eye_patch,
        stimulus_name=stimulus_name,
    )


def load_okr_log(path: str | Path) -> OkrLog:
    """Load OKR_Log_*.txt (tab-separated Unity stimulus event log).

    Supports:
    - Current event logs (``eventIndex`` / ``eventType`` / fixation + contrast blocks)
    - Older trial logs (``TrialIndex`` rows with ``dotStartSize`` / ``emissionRate``)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    raw_lines = path.read_text().splitlines()
    meta = _parse_header_metadata(raw_lines)
    stimulus_name = meta.get("StimulusName") or None
    header_eye_patch = meta.get("StimulusEyePatch") or None

    try:
        header, rows = _read_okr_table(raw_lines)
    except ValueError as exc:
        raise ValueError(f"No event table found in {path}") from exc

    # Older format: one row per trial, no eventType column
    if "TrialIndex" in header and "eventType" not in header:
        return _load_legacy_trial_okr_log(
            path,
            stimulus_name=stimulus_name,
            header_eye_patch=header_eye_patch,
            header=header,
            rows=rows,
        )

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
        direction = _direction_or_none(row.get("direction"))
        contrast_level = _parse_optional_float(row.get("contrastLevel", ""))
        dot_color = _dot_color_or_none(row.get("dotColor"))
        use_persistent_dots = _parse_optional_bool01(row.get("usePersistentDots", ""))
        is_anchor100 = _parse_optional_bool01(row.get("isAnchor100", ""))
        threshold_multiplier = _parse_optional_float(row.get("thresholdMultiplier", ""))
        dot_start_size = _parse_optional_float(row.get("dotStartSize", ""))
        emission_rate = _parse_optional_float(row.get("emissionRate", ""))

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
                    dot_start_size=dot_start_size,
                    emission_rate=emission_rate,
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
