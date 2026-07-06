"""Load gaze traces from USH2A tuple format or generic CSV."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class GazeTrial:
    """Eye position time series for one trial."""

    times: np.ndarray  # seconds
    elevation_deg: np.ndarray  # degrees
    trial_id: str = ""
    source_gaze: str = ""
    source_time: str = ""


def unity_gaze_direction(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match MATLAB unityGazeDirection.m."""
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(x, z)
    elevation = np.arcsin(np.divide(y, r, out=np.zeros_like(y), where=r != 0))
    azimuth = np.where(np.isnan(azimuth), 0.0, azimuth)
    elevation = np.where(np.isnan(elevation), 0.0, elevation)
    r = np.where(np.isnan(r), 0.0, r)
    return azimuth, elevation, r


def _parse_rotated_gaze(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = path.read_text()
    pattern = re.compile(r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)")
    matches = pattern.findall(text)
    if not matches:
        raise ValueError(f"No (x, y, z) tuples found in {path}")
    arr = np.array(matches, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


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

    azimuth, elevation, _ = unity_gaze_direction(xs, ys, zs)
    elevation_deg = np.degrees(elevation)
    azimuth_deg = np.degrees(azimuth)

    failure = (azimuth_deg == 0) & (elevation_deg == 0)
    padded = failure.copy()
    for offset in range(1, padding_frames + 1):
        padded[offset:] |= failure[:-offset]
        padded[:-offset] |= failure[offset:]
    elevation_deg = elevation_deg.astype(float)
    elevation_deg[padded] = np.nan

    if not trial_id:
        trial_id = gaze_path.parent.name or gaze_path.stem

    return GazeTrial(
        times=times,
        elevation_deg=elevation_deg,
        trial_id=trial_id,
        source_gaze=str(gaze_path.resolve()),
        source_time=str(time_path.resolve()),
    )


def load_csv_trial(
    csv_path: str | Path,
    time_col: str = "time",
    elevation_col: str = "elevation",
    trial_id: str = "",
) -> GazeTrial:
    """Load a CSV with time (s) and elevation (deg) columns."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if time_col not in df.columns or elevation_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{time_col}' and '{elevation_col}' columns. "
            f"Found: {list(df.columns)}"
        )
    if not trial_id:
        trial_id = csv_path.stem
    return GazeTrial(
        times=df[time_col].to_numpy(dtype=float),
        elevation_deg=df[elevation_col].to_numpy(dtype=float),
        trial_id=trial_id,
        source_gaze=str(csv_path.resolve()),
        source_time=str(csv_path.resolve()),
    )


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
