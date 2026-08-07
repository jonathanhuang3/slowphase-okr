"""Load gaze traces from Unity text exports or Tobii Glasses 3 JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PupilTrace:
    """SRanipal pupil position in normalized sensor-area coordinates (≈0–1)."""

    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    eye: str  # "left" or "right"
    source_position: str = ""
    source_time: str = ""


@dataclass
class PupilSensorStats:
    """Summary of how close pupil samples sit to the eye-camera sensor edge."""

    eye: str
    n_valid: int
    n_total: int
    mean_x: float
    mean_y: float
    pct_near_edge: float
    median_edge_dist: float
    margin: float
    source_position: str = ""


def pupil_edge_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance to nearest sensor border in normalized [0, 1] coordinates.

    ``0`` = on the edge; ``0.5`` = dead center. Invalid (NaN) inputs stay NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))


def summarize_pupil_sensor(
    pupil: PupilTrace,
    *,
    margin: float = 0.15,
) -> PupilSensorStats:
    """Compute mean pupil location and % of valid samples near the sensor edge."""
    if margin < 0 or margin >= 0.5:
        raise ValueError(f"margin must be in [0, 0.5), got {margin}")

    valid = np.isfinite(pupil.x) & np.isfinite(pupil.y)
    n_total = int(len(pupil.x))
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return PupilSensorStats(
            eye=pupil.eye,
            n_valid=0,
            n_total=n_total,
            mean_x=float("nan"),
            mean_y=float("nan"),
            pct_near_edge=float("nan"),
            median_edge_dist=float("nan"),
            margin=float(margin),
            source_position=pupil.source_position,
        )

    xs = pupil.x[valid]
    ys = pupil.y[valid]
    edge = pupil_edge_distance(xs, ys)
    near = edge < float(margin)
    return PupilSensorStats(
        eye=pupil.eye,
        n_valid=n_valid,
        n_total=n_total,
        mean_x=float(np.mean(xs)),
        mean_y=float(np.mean(ys)),
        pct_near_edge=float(100.0 * np.mean(near)),
        median_edge_dist=float(np.median(edge)),
        margin=float(margin),
        source_position=pupil.source_position,
    )


@dataclass
class EyeOpennessTrace:
    """SRanipal eye openness (typically 0=closed … 1=open)."""

    times: np.ndarray
    openness: np.ndarray
    eye: str  # "left" or "right"
    source_values: str = ""
    source_time: str = ""


@dataclass
class GazeOriginTrace:
    """SRanipal gaze origin (≈ eye center) in HMD-local millimeters."""

    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    eye: str  # "left" or "right"
    source_origin: str = ""
    source_time: str = ""


@dataclass
class GazeOriginStats:
    """Mean/SD/jitter of one eye's HMD-local gaze origin (mm)."""

    eye: str
    n_valid: int
    n_total: int
    mean_x: float
    mean_y: float
    mean_z: float
    sd_x: float
    sd_y: float
    sd_z: float
    # Mean |Δ| between consecutive valid samples (temporal jitter), mm.
    jitter_x: float
    jitter_y: float
    jitter_z: float
    jitter_3d: float  # mean consecutive 3D step length, mm
    source_origin: str = ""


@dataclass
class GazeOriginSymmetry:
    """Left vs right gaze-origin comparison in HMD-local mm."""

    left: GazeOriginStats | None
    right: GazeOriginStats | None
    delta_x: float  # left_mean - right_mean
    delta_y: float
    delta_z: float
    ipd_mm: float  # |delta_x|


def _mean_abs_consecutive_diff(values: np.ndarray, valid: np.ndarray) -> float:
    """Mean |Δ| between temporally adjacent samples that are both valid."""
    if len(values) < 2:
        return float("nan")
    pair = valid[:-1] & valid[1:]
    if not np.any(pair):
        return float("nan")
    return float(np.mean(np.abs(np.diff(values)[pair])))


def summarize_gaze_origin(origin: GazeOriginTrace) -> GazeOriginStats:
    valid = (
        np.isfinite(origin.x) & np.isfinite(origin.y) & np.isfinite(origin.z)
    )
    n_total = int(len(origin.x))
    n_valid = int(np.count_nonzero(valid))
    nan = float("nan")
    if n_valid == 0:
        return GazeOriginStats(
            eye=origin.eye,
            n_valid=0,
            n_total=n_total,
            mean_x=nan,
            mean_y=nan,
            mean_z=nan,
            sd_x=nan,
            sd_y=nan,
            sd_z=nan,
            jitter_x=nan,
            jitter_y=nan,
            jitter_z=nan,
            jitter_3d=nan,
            source_origin=origin.source_origin,
        )
    xs = origin.x[valid]
    ys = origin.y[valid]
    zs = origin.z[valid]
    jitter_x = _mean_abs_consecutive_diff(origin.x, valid)
    jitter_y = _mean_abs_consecutive_diff(origin.y, valid)
    jitter_z = _mean_abs_consecutive_diff(origin.z, valid)
    pair = valid[:-1] & valid[1:]
    if np.any(pair):
        step = np.sqrt(
            np.diff(origin.x)[pair] ** 2
            + np.diff(origin.y)[pair] ** 2
            + np.diff(origin.z)[pair] ** 2
        )
        jitter_3d = float(np.mean(step))
    else:
        jitter_3d = nan
    return GazeOriginStats(
        eye=origin.eye,
        n_valid=n_valid,
        n_total=n_total,
        mean_x=float(np.mean(xs)),
        mean_y=float(np.mean(ys)),
        mean_z=float(np.mean(zs)),
        sd_x=float(np.std(xs)),
        sd_y=float(np.std(ys)),
        sd_z=float(np.std(zs)),
        jitter_x=jitter_x,
        jitter_y=jitter_y,
        jitter_z=jitter_z,
        jitter_3d=jitter_3d,
        source_origin=origin.source_origin,
    )


def compare_gaze_origins(
    left: GazeOriginTrace | None,
    right: GazeOriginTrace | None,
) -> GazeOriginSymmetry:
    left_stats = summarize_gaze_origin(left) if left is not None else None
    right_stats = summarize_gaze_origin(right) if right is not None else None
    nan = float("nan")
    if left_stats is None or right_stats is None:
        return GazeOriginSymmetry(
            left=left_stats,
            right=right_stats,
            delta_x=nan,
            delta_y=nan,
            delta_z=nan,
            ipd_mm=nan,
        )
    if left_stats.n_valid == 0 or right_stats.n_valid == 0:
        return GazeOriginSymmetry(
            left=left_stats,
            right=right_stats,
            delta_x=nan,
            delta_y=nan,
            delta_z=nan,
            ipd_mm=nan,
        )
    dx = left_stats.mean_x - right_stats.mean_x
    dy = left_stats.mean_y - right_stats.mean_y
    dz = left_stats.mean_z - right_stats.mean_z
    return GazeOriginSymmetry(
        left=left_stats,
        right=right_stats,
        delta_x=float(dx),
        delta_y=float(dy),
        delta_z=float(dz),
        ipd_mm=float(abs(dx)),
    )


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
    pupil_left: PupilTrace | None = None
    pupil_right: PupilTrace | None = None
    source_format: str = "ush2a"  # "ush2a" | "tobii_glasses3"
    # Per-eye traces (Tobii Glasses 3); None for Unity/Vive exports.
    elevation_left_deg: np.ndarray | None = None
    elevation_right_deg: np.ndarray | None = None
    azimuth_left_deg: np.ndarray | None = None
    azimuth_right_deg: np.ndarray | None = None
    openness_left: EyeOpennessTrace | None = None
    openness_right: EyeOpennessTrace | None = None
    origin_left: GazeOriginTrace | None = None
    origin_right: GazeOriginTrace | None = None
    heading: HeadingTrace | None = None

    def has_per_eye_gaze(self) -> bool:
        return (
            self.elevation_left_deg is not None
            and self.elevation_right_deg is not None
        )

    def has_eye_openness(self) -> bool:
        return self.openness_left is not None or self.openness_right is not None

    def has_pupil_sensor(self) -> bool:
        return self.pupil_left is not None or self.pupil_right is not None

    def has_gaze_origins(self) -> bool:
        return self.origin_left is not None or self.origin_right is not None

    def has_heading(self) -> bool:
        return self.heading is not None


@dataclass
class HeadingTrace:
    """HMD heading over time, recovered from ``gazeRotations.txt``.

    Glance stores ``Inverse(headingRotation)`` in ``gazeRotations.txt``. We invert
    that again to recover heading, then express orientation relative to the first
    valid sample (Δ roll / pitch / yaw) so a fixed rig reads near zero wiggle.
    """

    times: np.ndarray
    roll_deg: np.ndarray
    pitch_deg: np.ndarray
    yaw_deg: np.ndarray
    angle_from_start_deg: np.ndarray
    source_rotations: str = ""
    source_time: str = ""


@dataclass
class HeadingWiggleStats:
    """Summary of headset orientation change over a trial."""

    n_valid: int
    n_total: int
    roll_sd: float
    pitch_sd: float
    yaw_sd: float
    roll_ptp: float
    pitch_ptp: float
    yaw_ptp: float
    max_angle_from_start: float
    source_rotations: str = ""


def summarize_heading_wiggle(heading: HeadingTrace) -> HeadingWiggleStats:
    valid = (
        np.isfinite(heading.roll_deg)
        & np.isfinite(heading.pitch_deg)
        & np.isfinite(heading.yaw_deg)
        & np.isfinite(heading.angle_from_start_deg)
    )
    n_total = int(len(heading.times))
    n_valid = int(np.count_nonzero(valid))
    nan = float("nan")
    if n_valid == 0:
        return HeadingWiggleStats(
            n_valid=0,
            n_total=n_total,
            roll_sd=nan,
            pitch_sd=nan,
            yaw_sd=nan,
            roll_ptp=nan,
            pitch_ptp=nan,
            yaw_ptp=nan,
            max_angle_from_start=nan,
            source_rotations=heading.source_rotations,
        )

    def _ptp(arr: np.ndarray) -> float:
        return float(np.nanmax(arr) - np.nanmin(arr))

    roll = heading.roll_deg[valid]
    pitch = heading.pitch_deg[valid]
    yaw = heading.yaw_deg[valid]
    ang = heading.angle_from_start_deg[valid]
    return HeadingWiggleStats(
        n_valid=n_valid,
        n_total=n_total,
        roll_sd=float(np.std(roll)),
        pitch_sd=float(np.std(pitch)),
        yaw_sd=float(np.std(yaw)),
        roll_ptp=_ptp(roll),
        pitch_ptp=_ptp(pitch),
        yaw_ptp=_ptp(yaw),
        max_angle_from_start=float(np.max(np.abs(ang))),
        source_rotations=heading.source_rotations,
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


def discover_eye_openness_files(
    trial_dir: str | Path,
    eye: str,
) -> tuple[Path, Path] | None:
    """Return SRanipal eye-openness value + time files for one eye, if present."""
    trial_dir = Path(trial_dir)
    eye_norm = eye.strip().lower()
    if eye_norm not in {"left", "right"}:
        raise ValueError(f"eye must be 'left' or 'right', got {eye!r}")

    prefix = "sranipalRight" if eye_norm == "right" else "sranipalLeft"
    values_path = trial_dir / f"{prefix}EyeOpenness.txt"
    if not values_path.is_file():
        return None
    time_path = trial_dir / f"{prefix}EyeOpennessTimes.txt"
    if not time_path.is_file():
        return None
    return values_path, time_path


def load_sranipal_eye_openness(
    values_path: str | Path,
    time_path: str | Path,
    *,
    eye: str,
) -> EyeOpennessTrace:
    """Load ``sranipal*EyeOpenness.txt`` + matching ``*Times.txt``."""
    values_path = Path(values_path)
    time_path = Path(time_path)
    if not values_path.is_file():
        raise FileNotFoundError(values_path)
    if not time_path.is_file():
        raise FileNotFoundError(time_path)

    openness = np.atleast_1d(np.loadtxt(values_path, dtype=float)).ravel()
    times = _parse_gaze_times(time_path)
    if len(openness) != len(times):
        raise ValueError(
            f"Length mismatch: {len(openness)} openness samples vs {len(times)} timestamps"
        )

    eye_norm = eye.strip().lower()
    if eye_norm not in {"left", "right"}:
        raise ValueError(f"eye must be 'left' or 'right', got {eye!r}")

    return EyeOpennessTrace(
        times=times.astype(float),
        openness=openness.astype(float),
        eye=eye_norm,
        source_values=str(values_path.resolve()),
        source_time=str(time_path.resolve()),
    )


def attach_eye_openness(
    trial: GazeTrial,
    trial_dir: str | Path,
) -> tuple[EyeOpennessTrace | None, EyeOpennessTrace | None]:
    """Discover and attach left/right SRanipal eye-openness traces, if present."""
    trial_dir = Path(trial_dir)
    left: EyeOpennessTrace | None = None
    right: EyeOpennessTrace | None = None

    left_files = discover_eye_openness_files(trial_dir, "left")
    if left_files is not None:
        left = load_sranipal_eye_openness(left_files[0], left_files[1], eye="left")

    right_files = discover_eye_openness_files(trial_dir, "right")
    if right_files is not None:
        right = load_sranipal_eye_openness(right_files[0], right_files[1], eye="right")

    trial.openness_left = left
    trial.openness_right = right
    return left, right


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


def attach_sranipal_pupils(
    trial: GazeTrial,
    trial_dir: str | Path,
    *,
    viewing_eye: str | None = None,
) -> tuple[PupilTrace | None, PupilTrace | None]:
    """Discover and attach left and right SRanipal pupil traces, if present.

    Also sets ``trial.pupil`` to the viewing-eye (or RE/LE-inferred) trace for
    backward compatibility with older callers.
    """
    trial_dir = Path(trial_dir)
    left: PupilTrace | None = None
    right: PupilTrace | None = None

    left_files = discover_pupil_files(trial_dir, viewing_eye="left")
    if left_files is not None:
        left = load_sranipal_pupil(left_files[0], left_files[1], eye="left")
    right_files = discover_pupil_files(trial_dir, viewing_eye="right")
    if right_files is not None:
        right = load_sranipal_pupil(right_files[0], right_files[1], eye="right")

    trial.pupil_left = left
    trial.pupil_right = right

    eye = viewing_eye or infer_viewing_eye(trial_id=trial.trial_id)
    trial.pupil = right if eye == "right" else left
    if trial.pupil is None:
        trial.pupil = left if left is not None else right
    return left, right


def discover_gaze_origin_files(
    trial_dir: str | Path,
    eye: str,
) -> tuple[Path, Path] | None:
    """Return SRanipal gaze-origin + time files for one eye, if present."""
    trial_dir = Path(trial_dir)
    eye_norm = eye.strip().lower()
    if eye_norm not in {"left", "right"}:
        raise ValueError(f"eye must be 'left' or 'right', got {eye!r}")

    prefix = "sranipalRight" if eye_norm == "right" else "sranipalLeft"
    origin_path = trial_dir / f"{prefix}GazeOrigins.txt"
    if not origin_path.is_file():
        return None
    time_path = trial_dir / f"{prefix}GazeTime.txt"
    if not time_path.is_file():
        return None
    return origin_path, time_path


def load_sranipal_gaze_origin(
    origin_path: str | Path,
    time_path: str | Path,
    *,
    eye: str,
) -> GazeOriginTrace:
    """Load ``sranipal*GazeOrigins.txt`` + matching ``*GazeTime.txt`` (mm, HMD-local)."""
    origin_path = Path(origin_path)
    time_path = Path(time_path)
    if not origin_path.is_file():
        raise FileNotFoundError(origin_path)
    if not time_path.is_file():
        raise FileNotFoundError(time_path)

    xs, ys, zs = _parse_rotated_gaze(origin_path)
    times = _parse_gaze_times(time_path)
    if len(xs) != len(times):
        raise ValueError(
            f"Length mismatch: {len(xs)} gaze-origin samples vs {len(times)} timestamps"
        )

    eye_norm = eye.strip().lower()
    if eye_norm not in {"left", "right"}:
        raise ValueError(f"eye must be 'left' or 'right', got {eye!r}")

    return GazeOriginTrace(
        times=times.astype(float),
        x=xs.astype(float),
        y=ys.astype(float),
        z=zs.astype(float),
        eye=eye_norm,
        source_origin=str(origin_path.resolve()),
        source_time=str(time_path.resolve()),
    )


def attach_sranipal_gaze_origins(
    trial: GazeTrial,
    trial_dir: str | Path,
) -> tuple[GazeOriginTrace | None, GazeOriginTrace | None]:
    """Discover and attach left/right SRanipal gaze-origin traces, if present."""
    trial_dir = Path(trial_dir)
    left: GazeOriginTrace | None = None
    right: GazeOriginTrace | None = None

    left_files = discover_gaze_origin_files(trial_dir, "left")
    if left_files is not None:
        left = load_sranipal_gaze_origin(left_files[0], left_files[1], eye="left")
    right_files = discover_gaze_origin_files(trial_dir, "right")
    if right_files is not None:
        right = load_sranipal_gaze_origin(right_files[0], right_files[1], eye="right")

    trial.origin_left = left
    trial.origin_right = right
    return left, right


def _parse_quaternions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse one (x, y, z, w) quaternion per non-empty line (Unity order)."""
    tuple_pattern = re.compile(
        r"^\(\s*([^,)]+)\s*,\s*([^,)]+)\s*,\s*([^,)]+)\s*,\s*([^,)]+)\s*\)\s*$",
        re.IGNORECASE,
    )
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    ws: list[float] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = tuple_pattern.match(stripped)
        if not match:
            raise ValueError(f"No (x, y, z, w) quaternion found in line: {stripped!r}")
        xs.append(_parse_gaze_component(match.group(1)))
        ys.append(_parse_gaze_component(match.group(2)))
        zs.append(_parse_gaze_component(match.group(3)))
        ws.append(_parse_gaze_component(match.group(4)))
    if not xs:
        raise ValueError(f"No quaternions found in {path}")
    return (
        np.array(xs, dtype=float),
        np.array(ys, dtype=float),
        np.array(zs, dtype=float),
        np.array(ws, dtype=float),
    )


def _quat_multiply(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    aw: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    bw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hamilton product a⊗b for arrays of quaternions (x,y,z,w)."""
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return x, y, z, w


def _quat_to_euler_deg(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert unit quaternions to roll/pitch/yaw (degrees, XYZ Tait–Bryan)."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    pitch = np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def discover_heading_files(trial_dir: str | Path) -> tuple[Path, Path] | None:
    """Return ``gazeRotations.txt`` + ``gazeTime.txt`` if both exist."""
    trial_dir = Path(trial_dir)
    rot_path = trial_dir / "gazeRotations.txt"
    time_path = trial_dir / "gazeTime.txt"
    if rot_path.is_file() and time_path.is_file():
        return rot_path, time_path
    return None


def load_heading_trace(
    rotations_path: str | Path,
    time_path: str | Path,
) -> HeadingTrace:
    """Load heading wiggle from Glance ``gazeRotations.txt`` + ``gazeTime.txt``.

    File stores ``Inverse(heading)``; we invert to heading, then report orientation
    relative to the first valid sample.
    """
    rotations_path = Path(rotations_path)
    time_path = Path(time_path)
    if not rotations_path.is_file():
        raise FileNotFoundError(rotations_path)
    if not time_path.is_file():
        raise FileNotFoundError(time_path)

    # Stored = Inverse(heading). For unit quaternions, Inverse = conjugate.
    ix, iy, iz, iw = _parse_quaternions(rotations_path)
    times = _parse_gaze_times(time_path)
    if len(ix) != len(times):
        raise ValueError(
            f"Length mismatch: {len(ix)} rotations vs {len(times)} timestamps"
        )

    # Normalize stored quats, then heading = conjugate(stored) = Inverse(stored)
    norms = np.sqrt(ix * ix + iy * iy + iz * iz + iw * iw)
    good = np.isfinite(norms) & (norms > 0)
    ix = np.divide(ix, norms, out=np.full_like(ix, np.nan), where=good)
    iy = np.divide(iy, norms, out=np.full_like(iy, np.nan), where=good)
    iz = np.divide(iz, norms, out=np.full_like(iz, np.nan), where=good)
    iw = np.divide(iw, norms, out=np.full_like(iw, np.nan), where=good)
    hx, hy, hz, hw = -ix, -iy, -iz, iw

    valid = (
        np.isfinite(hx)
        & np.isfinite(hy)
        & np.isfinite(hz)
        & np.isfinite(hw)
        & np.isfinite(times)
    )
    roll = np.full(len(times), np.nan, dtype=float)
    pitch = np.full(len(times), np.nan, dtype=float)
    yaw = np.full(len(times), np.nan, dtype=float)
    ang = np.full(len(times), np.nan, dtype=float)
    if not np.any(valid):
        return HeadingTrace(
            times=times.astype(float),
            roll_deg=roll,
            pitch_deg=pitch,
            yaw_deg=yaw,
            angle_from_start_deg=ang,
            source_rotations=str(rotations_path.resolve()),
            source_time=str(time_path.resolve()),
        )

    first = int(np.flatnonzero(valid)[0])
    # Relative rotation: q_rel = conj(q0) ⊗ q_i
    q0x, q0y, q0z, q0w = hx[first], hy[first], hz[first], hw[first]
    # conj(q0)
    c0x, c0y, c0z, c0w = -q0x, -q0y, -q0z, q0w
    rx, ry, rz, rw = _quat_multiply(
        np.full(len(times), c0x),
        np.full(len(times), c0y),
        np.full(len(times), c0z),
        np.full(len(times), c0w),
        hx,
        hy,
        hz,
        hw,
    )
    # Prefer shorter arc: if w < 0, flip sign
    flip = rw < 0
    rx = np.where(flip, -rx, rx)
    ry = np.where(flip, -ry, ry)
    rz = np.where(flip, -rz, rz)
    rw = np.where(flip, -rw, rw)

    r_roll, r_pitch, r_yaw = _quat_to_euler_deg(rx, ry, rz, rw)
    # Total rotation angle from start
    r_ang = np.degrees(2.0 * np.arccos(np.clip(rw, -1.0, 1.0)))

    roll[valid] = r_roll[valid]
    pitch[valid] = r_pitch[valid]
    yaw[valid] = r_yaw[valid]
    ang[valid] = r_ang[valid]

    return HeadingTrace(
        times=times.astype(float),
        roll_deg=roll,
        pitch_deg=pitch,
        yaw_deg=yaw,
        angle_from_start_deg=ang,
        source_rotations=str(rotations_path.resolve()),
        source_time=str(time_path.resolve()),
    )


def attach_heading_trace(
    trial: GazeTrial,
    trial_dir: str | Path,
) -> HeadingTrace | None:
    """Discover and attach heading wiggle from ``gazeRotations.txt``, if present."""
    discovered = discover_heading_files(trial_dir)
    if discovered is None:
        trial.heading = None
        return None
    rot_path, time_path = discovered
    trial.heading = load_heading_trace(rot_path, time_path)
    return trial.heading


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
