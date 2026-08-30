"""Load EyeLink ASC (ASCII EDF export) gaze trials for manual OKR marking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slowphase_okr.gaze import GazeTrial

_VALIDATE_RE = re.compile(
    r"VALIDATE LR POINT \d+\s+(?:LEFT|RIGHT)\s+at [\d.]+\,[\d.]+\s+"
    r"OFFSET ([\d.]+) deg\.\s+([-\d.]+)\,([-\d.]+) pix",
    re.IGNORECASE,
)
_DISPLAY_COORDS_RE = re.compile(
    r"(?:DISPLAY_COORDS|GAZE_COORDS)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
)


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
    has_hpose: bool


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
    has_hpose = False

    for line in lines:
        if line.startswith("SAMPLES\t") and "HPOSE" in line:
            has_hpose = True
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
        has_hpose=has_hpose,
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


def load_eyelink_asc_trial(
    gaze_path: str | Path,
    trial_id: str = "",
) -> GazeTrial:
    """Load binocular EyeLink ASC samples (screen GAZE → elevation/azimuth deg)."""
    gaze_path = Path(gaze_path)
    if not gaze_path.is_file():
        raise FileNotFoundError(gaze_path)

    text = gaze_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    meta = _parse_meta(lines)

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
        if len(floats) < 5:
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

    lx_i, ly_i, rx_i, ry_i = 0, 1, 3, 4

    lx = _col(raw_rows, lx_i)
    ly = _col(raw_rows, ly_i)
    rx = _col(raw_rows, rx_i)
    ry = _col(raw_rows, ry_i)

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
    rx, ry = _mask_oob(rx, ry)

    elev_l, az_l = _px_to_elev_az(lx, ly, meta)
    elev_r, az_r = _px_to_elev_az(rx, ry, meta)
    elev_b = _binocular_mean(elev_l, elev_r)
    az_b = _binocular_mean(az_l, az_r)

    if not trial_id:
        trial_id = gaze_path.stem or gaze_path.parent.name or "trial"

    resolved = str(gaze_path.resolve())
    return GazeTrial(
        times=times,
        elevation_deg=elev_b,
        azimuth_deg=az_b,
        trial_id=trial_id,
        source_gaze=resolved,
        source_time=resolved,
        source_format="eyelink_asc",
        elevation_left_deg=elev_l,
        elevation_right_deg=elev_r,
        azimuth_left_deg=az_l,
        azimuth_right_deg=az_r,
    )
