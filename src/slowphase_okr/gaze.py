"""Load gaze traces from Unity text exports or Tobii Glasses 3 JSON."""

from __future__ import annotations

import json
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
    elevation_deg: np.ndarray  # degrees (binocular / primary)
    azimuth_deg: np.ndarray | None = None  # degrees (horizontal)
    trial_id: str = ""
    source_gaze: str = ""
    source_time: str = ""
    pupil: PupilTrace | None = None
    source_format: str = "ush2a"  # "ush2a" | "tobii_glasses3"
    # Per-eye traces (Tobii Glasses 3); None for Unity/Vive exports.
    elevation_left_deg: np.ndarray | None = None
    elevation_right_deg: np.ndarray | None = None
    azimuth_left_deg: np.ndarray | None = None
    azimuth_right_deg: np.ndarray | None = None

    def has_per_eye_gaze(self) -> bool:
        return (
            self.elevation_left_deg is not None
            and self.elevation_right_deg is not None
        )


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
        source_format="ush2a",
    )


def _xyz_to_elev_az_deg(x: float, y: float, z: float) -> tuple[float, float]:
    """Elevation / azimuth (deg) from a 3D vector (Y-up, Z-forward)."""
    r = float(np.sqrt(x * x + y * y + z * z))
    if r == 0.0 or not np.isfinite(r):
        return float("nan"), float("nan")
    elev = float(np.degrees(np.arcsin(np.clip(y / r, -1.0, 1.0))))
    az = float(np.degrees(np.arctan2(x, z)))
    return elev, az


def _tobii_eye_elev_az(data: dict, eye_key: str) -> tuple[float, float]:
    eye = data.get(eye_key) or {}
    direction = eye.get("gazedirection")
    if (
        isinstance(direction, (list, tuple))
        and len(direction) >= 3
        and all(v is not None for v in direction[:3])
    ):
        return _xyz_to_elev_az_deg(
            float(direction[0]), float(direction[1]), float(direction[2])
        )
    return float("nan"), float("nan")


def _tobii_gaze3d_elev_az(data: dict) -> tuple[float, float]:
    gaze3d = data.get("gaze3d")
    if (
        isinstance(gaze3d, (list, tuple))
        and len(gaze3d) >= 3
        and all(v is not None for v in gaze3d[:3])
    ):
        return _xyz_to_elev_az_deg(
            float(gaze3d[0]), float(gaze3d[1]), float(gaze3d[2])
        )
    return float("nan"), float("nan")


def _tobii_sample_eyes(
    data: dict,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return (left, right, binocular) elevation/azimuth pairs for one sample.

    Binocular is the mean of available eyes; if neither eye has a direction,
    falls back to ``gaze3d``.
    """
    left = _tobii_eye_elev_az(data, "eyeleft")
    right = _tobii_eye_elev_az(data, "eyeright")
    elevs = [e for e, a in (left, right) if np.isfinite(e) and np.isfinite(a)]
    azs = [a for e, a in (left, right) if np.isfinite(e) and np.isfinite(a)]
    if elevs:
        binocular = (float(np.mean(elevs)), float(np.mean(azs)))
    else:
        binocular = _tobii_gaze3d_elev_az(data)
    return left, right, binocular


def is_tobii_glasses3_gazedata(path: str | Path) -> bool:
    """True if ``path`` looks like Tobii Glasses 3 NDJSON ``gazedata``."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(40):
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                if not isinstance(obj, dict):
                    return False
                if obj.get("type") == "gaze" and "timestamp" in obj:
                    return True
                # Non-gaze event lines are common; keep scanning.
                if "timestamp" in obj and "data" in obj:
                    continue
                return False
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    return False


def load_tobii_glasses3_trial(
    gaze_path: str | Path,
    trial_id: str = "",
) -> GazeTrial:
    """Load Tobii Pro Glasses 3 ``gazedata.json`` (NDJSON, one object per line).

    Timestamps are embedded (seconds). Stores left, right, and binocular
    elevation/azimuth from gaze direction (scene-camera CS: Y up).
    """
    gaze_path = Path(gaze_path)
    if not gaze_path.is_file():
        raise FileNotFoundError(gaze_path)

    times_list: list[float] = []
    elev_b_list: list[float] = []
    az_b_list: list[float] = []
    elev_l_list: list[float] = []
    az_l_list: list[float] = []
    elev_r_list: list[float] = []
    az_r_list: list[float] = []
    n_gaze = 0

    with gaze_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {gaze_path.name}: {exc}"
                ) from exc
            if not isinstance(obj, dict) or obj.get("type") != "gaze":
                continue
            n_gaze += 1
            data = obj.get("data")
            if not isinstance(data, dict) or not data:
                # Glasses 3 emits empty ``data`` when tracking is lost.
                continue
            try:
                t = float(obj["timestamp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Missing/invalid timestamp on line {line_no} of {gaze_path.name}"
                ) from exc
            left, right, binocular = _tobii_sample_eyes(data)
            times_list.append(t)
            elev_l_list.append(left[0])
            az_l_list.append(left[1])
            elev_r_list.append(right[0])
            az_r_list.append(right[1])
            elev_b_list.append(binocular[0])
            az_b_list.append(binocular[1])

    if not times_list:
        if n_gaze == 0:
            raise ValueError(
                f"No gaze samples found in {gaze_path}. "
                "Expected Tobii Glasses 3 NDJSON (type=gaze per line)."
            )
        raise ValueError(
            f"No valid gaze directions in {gaze_path} "
            f"({n_gaze} gaze rows, all empty or missing direction)."
        )

    times = np.asarray(times_list, dtype=float)
    elevation_deg = np.asarray(elev_b_list, dtype=float)
    azimuth_deg = np.asarray(az_b_list, dtype=float)
    elevation_left = np.asarray(elev_l_list, dtype=float)
    elevation_right = np.asarray(elev_r_list, dtype=float)
    azimuth_left = np.asarray(az_l_list, dtype=float)
    azimuth_right = np.asarray(az_r_list, dtype=float)

    if not trial_id:
        trial_id = gaze_path.parent.name or gaze_path.stem

    resolved = str(gaze_path.resolve())
    return GazeTrial(
        times=times,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        trial_id=trial_id,
        source_gaze=resolved,
        source_time=resolved,
        source_format="tobii_glasses3",
        elevation_left_deg=elevation_left,
        elevation_right_deg=elevation_right,
        azimuth_left_deg=azimuth_left,
        azimuth_right_deg=azimuth_right,
    )


def load_gaze_trial(
    gaze_path: str | Path,
    time_path: str | Path | None = None,
    trial_id: str = "",
    padding_frames: int = 0,
) -> GazeTrial:
    """Load a trial from Unity gaze+time files or Tobii Glasses 3 JSON."""
    gaze_path = Path(gaze_path)
    if is_tobii_glasses3_gazedata(gaze_path):
        return load_tobii_glasses3_trial(gaze_path, trial_id=trial_id)
    if time_path is None:
        raise ValueError(
            "Time file is required for Unity/Vive gaze exports "
            "(rotatedGaze.txt + gazeTime.txt)."
        )
    return load_ush2a_trial(
        gaze_path, time_path, trial_id=trial_id, padding_frames=padding_frames
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
