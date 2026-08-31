"""Load EyeLink ASC (ASCII EDF export) gaze trials for manual OKR marking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slowphase_okr.gaze import (
    EyeInHeadTrace,
    GazeTrial,
    heading_trace_from_euler_deg,
)

_VALIDATE_RE = re.compile(
    r"VALIDATE LR POINT \d+\s+(?:LEFT|RIGHT)\s+at [\d.]+\,[\d.]+\s+"
    r"OFFSET ([\d.]+) deg\.\s+([-\d.]+)\,([-\d.]+) pix",
    re.IGNORECASE,
)
_DISPLAY_COORDS_RE = re.compile(
    r"(?:DISPLAY_COORDS|GAZE_COORDS)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
)


@dataclass(frozen=True)
class _AscColumnMap:
    """Float-column indices after parsing one EyeLink sample line."""

    binocular: bool
    gaze_lx: int
    gaze_ly: int
    gaze_rx: int | None
    gaze_ry: int | None
    hpose_roll: int | None
    hpose_pitch: int | None
    hpose_yaw: int | None
    eih_ly: int | None
    eih_ry: int | None


@dataclass(frozen=True)
class _AscMeta:
    display_left: float
    display_top: float
    display_right: float
    display_bottom: float
    center_x: float
    center_y: float
    px_per_deg_x: float
    px_per_deg_y: float
    trim_start_ms: int | None
    trim_end_ms: int | None
    column_map: _AscColumnMap


def _column_map_from_samples_line(line: str) -> _AscColumnMap:
    """Map parsed float columns from the SAMPLES header line."""
    tokens = {t.strip().upper() for t in line.split("\t")}
    has_right = "RIGHT" in tokens
    has_hpose = "HPOSE" in tokens
    if has_right:
        return _AscColumnMap(
            binocular=True,
            gaze_lx=0,
            gaze_ly=1,
            gaze_rx=3,
            gaze_ry=4,
            hpose_roll=6 if has_hpose else None,
            hpose_pitch=7 if has_hpose else None,
            hpose_yaw=8 if has_hpose else None,
            eih_ly=17 if has_hpose else None,
            eih_ry=18 if has_hpose else None,
        )
    return _AscColumnMap(
        binocular=False,
        gaze_lx=0,
        gaze_ly=1,
        gaze_rx=None,
        gaze_ry=None,
        hpose_roll=3 if has_hpose else None,
        hpose_pitch=4 if has_hpose else None,
        hpose_yaw=5 if has_hpose else None,
        eih_ly=12 if has_hpose else None,
        eih_ry=None,
    )


def is_eyelink_asc(path: str | Path) -> bool:
    """True if ``path`` looks like an EyeLink ASC export."""
    path = Path(path)
    if path.suffix.lower() != ".asc":
        return False
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(80):
                line = handle.readline()
                if not line:
                    break
                upper = line.upper()
                if "EYELINK" in upper and line.startswith("**"):
                    return True
                if line.startswith("SAMPLES\t") and "GAZE" in line:
                    return True
    except OSError:
        return False
    return False


def _parse_float_token(token: str) -> float | None:
    token = token.strip()
    if not token or token == "." or set(token) <= {"."}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _parse_sample_floats(parts: list[str]) -> list[float | None]:
    floats: list[float | None] = []
    for token in parts[1:]:
        if len(token) >= 3 and set(token.strip()) <= {"."}:
            continue
        floats.append(_parse_float_token(token))
    return floats


def _parse_meta(lines: list[str]) -> _AscMeta:
    left = top = 0.0
    right = bottom = 1919.0
    px_x: list[float] = []
    px_y: list[float] = []
    trim_start: int | None = None
    trim_end: int | None = None
    column_map = _AscColumnMap(
        binocular=True,
        gaze_lx=0,
        gaze_ly=1,
        gaze_rx=3,
        gaze_ry=4,
        hpose_roll=6,
        hpose_pitch=7,
        hpose_yaw=8,
        eih_ly=17,
        eih_ry=18,
    )

    for line in lines:
        if line.startswith("SAMPLES\t"):
            column_map = _column_map_from_samples_line(line)
        m = _DISPLAY_COORDS_RE.search(line)
        if m:
            left, top, right, bottom = (float(m.group(i)) for i in range(1, 5))
        vm = _VALIDATE_RE.search(line)
        if vm:
            off = float(vm.group(1))
            pix_x = abs(float(vm.group(2)))
            pix_y = abs(float(vm.group(3)))
            if off >= 0.2:
                if pix_x > 0:
                    px_x.append(pix_x / off)
                if pix_y > 0:
                    px_y.append(pix_y / off)
        if "SYNCTIME" in line and line.startswith("MSG"):
            try:
                trim_start = int(line.split("\t", 2)[1].split()[0])
            except (IndexError, ValueError):
                pass
        if "TRIAL_RESULT" in line and line.startswith("MSG"):
            try:
                trim_end = int(line.split("\t", 2)[1].split()[0])
            except (IndexError, ValueError):
                pass

    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    px_per_deg_x = float(np.median(px_x)) if px_x else 40.0
    px_per_deg_y = float(np.median(px_y)) if px_y else 40.0
    return _AscMeta(
        display_left=left,
        display_top=top,
        display_right=right,
        display_bottom=bottom,
        center_x=center_x,
        center_y=center_y,
        px_per_deg_x=px_per_deg_x,
        px_per_deg_y=px_per_deg_y,
        trim_start_ms=trim_start,
        trim_end_ms=trim_end,
        column_map=column_map,
    )


def _px_to_elev_az(
    x_px: np.ndarray,
    y_px: np.ndarray,
    meta: _AscMeta,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth = (x_px - meta.center_x) / meta.px_per_deg_x
    elevation = (meta.center_y - y_px) / meta.px_per_deg_y
    return elevation.astype(float), azimuth.astype(float)


def _binocular_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.full(len(left), np.nan, dtype=float)
    both = np.isfinite(left) & np.isfinite(right)
    out[both] = 0.5 * (left[both] + right[both])
    left_only = np.isfinite(left) & ~np.isfinite(right)
    right_only = np.isfinite(right) & ~np.isfinite(left)
    out[left_only] = left[left_only]
    out[right_only] = right[right_only]
    return out


def _col(rows: list[tuple[int, list[float | None]]], idx: int) -> np.ndarray:
    out = np.full(len(rows), np.nan, dtype=float)
    for i, (_, floats) in enumerate(rows):
        if idx < len(floats) and floats[idx] is not None:
            out[i] = floats[idx]
    return out


def _mask_hpose_angles(
    roll: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    *,
    max_abs_deg: float = 45.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop HPOSE Euler samples outside a plausible tower range."""
    roll = roll.astype(float, copy=True)
    pitch = pitch.astype(float, copy=True)
    yaw = yaw.astype(float, copy=True)
    bad = (
        ~np.isfinite(roll)
        | ~np.isfinite(pitch)
        | ~np.isfinite(yaw)
        | (np.abs(roll) > max_abs_deg)
        | (np.abs(pitch) > max_abs_deg)
        | (np.abs(yaw) > max_abs_deg)
    )
    roll[bad] = np.nan
    pitch[bad] = np.nan
    yaw[bad] = np.nan
    return roll, pitch, yaw


def _nan_array_like(rows: list[tuple[int, list[float | None]]]) -> np.ndarray:
    return np.full(len(rows), np.nan, dtype=float)


def load_eyelink_asc_trial(
    gaze_path: str | Path,
    trial_id: str = "",
) -> GazeTrial:
    """Load EyeLink ASC samples (binocular or monocular) as elevation/azimuth deg."""
    gaze_path = Path(gaze_path)
    if not gaze_path.is_file():
        raise FileNotFoundError(gaze_path)

    text = gaze_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    meta = _parse_meta(lines)
    cols = meta.column_map

    in_samples = False
    raw_rows: list[tuple[int, list[float | None]]] = []
    for line in lines:
        if line.startswith("SAMPLES\t"):
            in_samples = True
            continue
        if line.startswith("END\t"):
            break
        if not in_samples:
            continue
        if not line or line[0].isalpha():
            continue
        parts = line.rstrip("\n").split("\t")
        if not parts[0].strip().isdigit():
            continue
        ts = int(parts[0])
        floats = _parse_sample_floats(parts)
        if len(floats) < 3:
            continue
        raw_rows.append((ts, floats))

    if not raw_rows:
        raise ValueError(f"No gaze samples found in {gaze_path.name}")

    t_start = meta.trim_start_ms
    t_end = meta.trim_end_ms
    if t_start is not None:
        raw_rows = [(t, f) for t, f in raw_rows if t >= t_start]
    if t_end is not None:
        raw_rows = [(t, f) for t, f in raw_rows if t <= t_end]
    if not raw_rows:
        raise ValueError(
            f"No gaze samples remain after trial trim in {gaze_path.name} "
            f"(SYNCTIME={t_start}, TRIAL_RESULT={t_end})."
        )

    t0_ms = raw_rows[0][0]
    times = np.array([(t - t0_ms) / 1000.0 for t, _ in raw_rows], dtype=float)

    lx = _col(raw_rows, cols.gaze_lx)
    ly = _col(raw_rows, cols.gaze_ly)
    if cols.gaze_rx is not None and cols.gaze_ry is not None:
        rx = _col(raw_rows, cols.gaze_rx)
        ry = _col(raw_rows, cols.gaze_ry)
    else:
        rx = _nan_array_like(raw_rows)
        ry = _nan_array_like(raw_rows)

    margin = 500.0
    x_min = meta.display_left - margin
    x_max = meta.display_right + margin
    y_min = meta.display_top - margin
    y_max = meta.display_bottom + margin

    def _mask_oob(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bad = (
            ~np.isfinite(x)
            | ~np.isfinite(y)
            | (x < x_min)
            | (x > x_max)
            | (y < y_min)
            | (y > y_max)
        )
        x = x.astype(float, copy=True)
        y = y.astype(float, copy=True)
        x[bad] = np.nan
        y[bad] = np.nan
        return x, y

    lx, ly = _mask_oob(lx, ly)
    if cols.binocular:
        rx, ry = _mask_oob(rx, ry)

    elev_l, az_l = _px_to_elev_az(lx, ly, meta)
    if cols.binocular:
        elev_r, az_r = _px_to_elev_az(rx, ry, meta)
        elev_b = _binocular_mean(elev_l, elev_r)
        az_b = _binocular_mean(az_l, az_r)
    else:
        elev_r = _nan_array_like(raw_rows)
        az_r = _nan_array_like(raw_rows)
        elev_b = elev_l.copy()
        az_b = az_l.copy()

    heading = None
    eye_in_head = None
    n_cols = len(raw_rows[0][1])
    roll = pitch = yaw = None
    if (
        cols.hpose_roll is not None
        and cols.hpose_pitch is not None
        and cols.hpose_yaw is not None
        and n_cols > cols.hpose_yaw
    ):
        roll, pitch, yaw = _mask_hpose_angles(
            _col(raw_rows, cols.hpose_roll),
            _col(raw_rows, cols.hpose_pitch),
            _col(raw_rows, cols.hpose_yaw),
        )

    if cols.eih_ly is not None and n_cols > cols.eih_ly:
        eih_ly = _col(raw_rows, cols.eih_ly)
        bad_l = ~np.isfinite(eih_ly) | (eih_ly < y_min) | (eih_ly > y_max)
        eih_ly = eih_ly.astype(float, copy=True)
        eih_ly[bad_l] = np.nan
        eih_elev_l = (meta.center_y - eih_ly) / meta.px_per_deg_y
        if cols.eih_ry is not None and n_cols > cols.eih_ry:
            eih_ry = _col(raw_rows, cols.eih_ry)
            bad_r = ~np.isfinite(eih_ry) | (eih_ry < y_min) | (eih_ry > y_max)
            eih_ry = eih_ry.astype(float, copy=True)
            eih_ry[bad_r] = np.nan
            eih_elev_r = (meta.center_y - eih_ry) / meta.px_per_deg_y
            eih_elev_b = _binocular_mean(eih_elev_l, eih_elev_r)
        else:
            eih_elev_r = _nan_array_like(raw_rows)
            eih_elev_b = eih_elev_l.copy()
        eye_in_head = EyeInHeadTrace(
            times=times,
            elevation_left_deg=eih_elev_l.astype(float),
            elevation_right_deg=eih_elev_r.astype(float),
            elevation_deg=eih_elev_b.astype(float),
        )

    if not trial_id:
        trial_id = gaze_path.stem or gaze_path.parent.name or "trial"

    resolved = str(gaze_path.resolve())
    if roll is not None and pitch is not None and yaw is not None:
        heading = heading_trace_from_euler_deg(
            times,
            roll,
            pitch,
            yaw,
            source_rotations=resolved,
            source_time=resolved,
        )
    return GazeTrial(
        times=times,
        elevation_deg=elev_b,
        azimuth_deg=az_b,
        trial_id=trial_id,
        source_gaze=resolved,
        source_time=resolved,
        source_format="eyelink_asc",
        elevation_left_deg=elev_l,
        elevation_right_deg=elev_r if cols.binocular else None,
        azimuth_left_deg=az_l,
        azimuth_right_deg=az_r if cols.binocular else None,
        heading=heading,
        eye_in_head=eye_in_head,
    )
