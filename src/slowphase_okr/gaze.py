"""Load gaze traces from gaze direction + timestamp text files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PupilTrace:
    """SRanipal pupil position in normalized screen coordinates."""

    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    eye: str  # "left" or "right"
    source_position: str = ""
    source_time: str = ""


@dataclass
class GazeTrial:
    """Eye position time series for one trial."""

    times: np.ndarray  # seconds
    elevation_deg: np.ndarray  # degrees
    azimuth_deg: np.ndarray | None = None  # degrees (horizontal)
    trial_id: str = ""
    source_gaze: str = ""
    source_time: str = ""
    pupil: PupilTrace | None = None


def unity_gaze_direction(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(x, z)
    elevation = np.arcsin(np.divide(y, r, out=np.zeros_like(y), where=r != 0))
    azimuth = np.where(np.isnan(azimuth), 0.0, azimuth)
    elevation = np.where(np.isnan(elevation), 0.0, elevation)
    r = np.where(np.isnan(r), 0.0, r)
    return azimuth, elevation, r


def _parse_gaze_component(text: str) -> float:
    text = text.strip()
    if text.lower() == "nan":
        return float("nan")
    return float(text)


def _parse_rotated_gaze(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse one (x, y, z) tuple per non-empty line, preserving line order."""
    tuple_pattern = re.compile(
        r"^\(\s*([^,)]+)\s*,\s*([^,)]+)\s*,\s*([^,)]+)\s*\)\s*$",
        re.IGNORECASE,
    )
    xs_list: list[float] = []
    ys_list: list[float] = []
    zs_list: list[float] = []

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = tuple_pattern.match(stripped)
        if not match:
            raise ValueError(f"No (x, y, z) tuple found in line: {stripped!r}")
        xs_list.append(_parse_gaze_component(match.group(1)))
        ys_list.append(_parse_gaze_component(match.group(2)))
        zs_list.append(_parse_gaze_component(match.group(3)))

    if not xs_list:
        raise ValueError(f"No (x, y, z) tuples found in {path}")

    return (
        np.array(xs_list, dtype=float),
        np.array(ys_list, dtype=float),
        np.array(zs_list, dtype=float),
    )


def _parse_gaze_times(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=float)
    return np.atleast_1d(values).ravel()


def load_ush2a_trial(
    gaze_path: str | Path,
    time_path: str | Path,
    trial_id: str = "",
    padding_frames: int = 0,
) -> GazeTrial:
    """Load rotatedGaze.txt + gazeTime.txt (Unity / USH2A format)."""
    gaze_path = Path(gaze_path)
    time_path = Path(time_path)
    if not gaze_path.is_file():
        raise FileNotFoundError(gaze_path)
    if not time_path.is_file():
        raise FileNotFoundError(time_path)

    xs, ys, zs = _parse_rotated_gaze(gaze_path)
    times = _parse_gaze_times(time_path)
    if len(xs) != len(times):
        raise ValueError(
            f"Length mismatch: {len(xs)} gaze samples vs {len(times)} timestamps"
        )

    invalid = np.isnan(xs) | np.isnan(ys) | np.isnan(zs)
    azimuth, elevation, _ = unity_gaze_direction(xs, ys, zs)
    elevation_deg = np.degrees(elevation).astype(float)
    azimuth_deg = np.degrees(azimuth).astype(float)
    elevation_deg[invalid] = np.nan
    azimuth_deg[invalid] = np.nan

    failure = (~invalid) & (azimuth_deg == 0) & (elevation_deg == 0)
    padded = failure.copy()
    for offset in range(1, padding_frames + 1):
        padded[offset:] |= failure[:-offset]
        padded[:-offset] |= failure[offset:]
    elevation_deg[padded] = np.nan
    azimuth_deg[padded] = np.nan

    if not trial_id:
        trial_id = gaze_path.parent.name or gaze_path.stem

    return GazeTrial(
        times=times,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        trial_id=trial_id,
        source_gaze=str(gaze_path.resolve()),
        source_time=str(time_path.resolve()),
    )


def _parse_pupil_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse one (x, y) tuple per non-empty line."""
    tuple_pattern = re.compile(
        r"^\(\s*([^,)]+)\s*,\s*([^,)]+)\s*\)\s*$",
        re.IGNORECASE,
    )
    xs_list: list[float] = []
    ys_list: list[float] = []

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = tuple_pattern.match(stripped)
        if not match:
            raise ValueError(f"No (x, y) tuple found in line: {stripped!r}")
        xs_list.append(_parse_gaze_component(match.group(1)))
        ys_list.append(_parse_gaze_component(match.group(2)))

    if not xs_list:
        raise ValueError(f"No (x, y) tuples found in {path}")

    return np.array(xs_list, dtype=float), np.array(ys_list, dtype=float)


def discover_pupil_files(
    trial_dir: str | Path,
    viewing_eye: str = "right",
) -> tuple[Path, Path] | None:
    """Return SRanipal pupil position + time files for the viewing eye, if present."""
    trial_dir = Path(trial_dir)
    eye = viewing_eye.strip().lower()
    if eye not in {"left", "right"}:
        raise ValueError(f"viewing_eye must be 'left' or 'right', got {viewing_eye!r}")

    prefix = "sranipalRight" if eye == "right" else "sranipalLeft"
    pos_path = trial_dir / f"{prefix}PupilPositions.txt"
    if not pos_path.is_file():
        return None

    for time_name in (f"{prefix}PupilPositionTimes.txt", f"{prefix}PupilTimes.txt"):
        time_path = trial_dir / time_name
        if time_path.is_file():
            return pos_path, time_path
    return None


def infer_viewing_eye(
    *,
    eye_patch: str | None = None,
    trial_id: str = "",
) -> str:
    """Infer which eye was viewing (unpatched) from OKR log patch side or trial name."""
    if eye_patch:
        patch = eye_patch.strip().lower()
        if patch == "left":
            return "right"
        if patch == "right":
            return "left"

    normalized = f" {trial_id.replace('_', ' ').upper()} "
    if " LE " in normalized:
        return "left"
    if " RE " in normalized:
        return "right"
    return "right"


def load_sranipal_pupil(
    position_path: str | Path,
    time_path: str | Path,
    *,
    eye: str,
) -> PupilTrace:
    """Load sranipal*PupilPositions.txt + matching time file."""
    position_path = Path(position_path)
    time_path = Path(time_path)
    if not position_path.is_file():
        raise FileNotFoundError(position_path)
    if not time_path.is_file():
        raise FileNotFoundError(time_path)

    xs, ys = _parse_pupil_positions(position_path)
    times = _parse_gaze_times(time_path)
    if len(xs) != len(times):
        raise ValueError(
            f"Length mismatch: {len(xs)} pupil samples vs {len(times)} timestamps"
        )

    invalid = np.isnan(xs) | np.isnan(ys)
    xs = xs.astype(float)
    ys = ys.astype(float)
    xs[invalid] = np.nan
    ys[invalid] = np.nan

    eye_norm = eye.strip().lower()
    if eye_norm not in {"left", "right"}:
        raise ValueError(f"eye must be 'left' or 'right', got {eye!r}")

    return PupilTrace(
        times=times,
        x=xs,
        y=ys,
        eye=eye_norm,
        source_position=str(position_path.resolve()),
        source_time=str(time_path.resolve()),
    )


def attach_sranipal_pupil(
    trial: GazeTrial,
    trial_dir: str | Path,
    *,
    viewing_eye: str | None = None,
) -> PupilTrace | None:
    """Discover and attach SRanipal pupil trace to ``trial``, if files exist."""
    eye = viewing_eye or infer_viewing_eye(trial_id=trial.trial_id)
    discovered = discover_pupil_files(trial_dir, viewing_eye=eye)
    if discovered is None:
        trial.pupil = None
        return None

    pos_path, time_path = discovered
    trial.pupil = load_sranipal_pupil(pos_path, time_path, eye=eye)
    return trial.pupil


def analysis_window_mask(
    times: np.ndarray,
    t0: float,
    duration_sec: float | None = None,
    t_end: float | None = None,
) -> np.ndarray:
    """Boolean mask for samples in the analysis window.

    If ``t_end`` is given, the window is ``[t0, t_end]`` (inclusive).
    Otherwise the window is ``[t0, t0 + duration_sec]`` (default 40 s).
    """
    if t_end is not None:
        return (times >= t0) & (times <= t_end)
    if duration_sec is None:
        duration_sec = 40.0
    return (times >= t0) & (times <= t0 + duration_sec)
