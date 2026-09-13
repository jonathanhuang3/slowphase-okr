"""Fixation precision / accuracy for custom HV9 (CustomValidation) targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CustomFixationPoint:
    """One CustomValidation target interval (ASC CUSTOM_VAL or log file)."""

    point_index: int
    label: str
    t_start: float
    t_end: float
    target_x_px: float
    target_y_px: float
    target_azimuth_deg: float
    target_elevation_deg: float
    rep: int = 1


@dataclass(frozen=True)
class FixationPrecisionResult:
    """Precision + accuracy for one marked (or ASC) fixation window."""

    point_index: int
    label: str
    segment_id: int | None
    t_start: float
    t_end: float
    n_samples: int
    target_azimuth_deg: float
    target_elevation_deg: float
    mean_azimuth_deg: float
    mean_elevation_deg: float
    sd_azimuth_deg: float
    sd_elevation_deg: float
    sd_2d_deg: float
    rms_s2s_deg: float
    signed_error_azimuth_deg: float
    signed_error_elevation_deg: float
    unsigned_error_deg: float
    azimuth_samples_deg: np.ndarray
    elevation_samples_deg: np.ndarray


_CUSTOM_VAL_RE_PARTS = (
    r"CUSTOM_VAL\s+POINT_(?P<kind>START|END)\s+"
    r"(?P<idx>\d+)\s+(?P<label>\S+)\s+at\s+"
    r"(?P<x>[-\d.]+)\s*,\s*(?P<y>[-\d.]+)"
)


def parse_custom_val_points_ms(lines: list[str]) -> list[dict]:
    """Parse ASC CUSTOM_VAL POINT_START/END rows (absolute Host ms)."""
    import re

    pattern = re.compile(_CUSTOM_VAL_RE_PARTS, re.IGNORECASE)
    starts: dict[tuple[int, str], dict] = {}
    points: list[dict] = []
    for line in lines:
        if "CUSTOM_VAL" not in line or not line.startswith("MSG"):
            continue
        m = pattern.search(line)
        if not m:
            continue
        try:
            t_ms = int(line.split("\t", 2)[1].split()[0])
        except (IndexError, ValueError):
            continue
        idx = int(m.group("idx"))
        label = m.group("label")
        x_px = float(m.group("x"))
        y_px = float(m.group("y"))
        key = (idx, label)
        kind = m.group("kind").upper()
        if kind == "START":
            starts[key] = {
                "point_index": idx,
                "label": label,
                "t_start_ms": t_ms,
                "target_x_px": x_px,
                "target_y_px": y_px,
            }
        elif kind == "END" and key in starts:
            start = starts.pop(key)
            points.append(
                {
                    **start,
                    "t_end_ms": t_ms,
                    "target_x_px": x_px,
                    "target_y_px": y_px,
                }
            )
    points.sort(key=lambda p: p["t_start_ms"])
    return points


def custom_fixation_points_from_asc(
    lines: list[str],
    *,
    t0_ms: int,
    center_x: float,
    center_y: float,
    px_per_deg_x: float,
    px_per_deg_y: float,
) -> list[CustomFixationPoint]:
    """Convert ASC CUSTOM_VAL intervals to trial-relative seconds + degrees."""
    out: list[CustomFixationPoint] = []
    for raw in parse_custom_val_points_ms(lines):
        t_start = (raw["t_start_ms"] - t0_ms) / 1000.0
        t_end = (raw["t_end_ms"] - t0_ms) / 1000.0
        if t_end < 0:
            continue
        x_px = float(raw["target_x_px"])
        y_px = float(raw["target_y_px"])
        az = (x_px - center_x) / px_per_deg_x
        el = (center_y - y_px) / px_per_deg_y
        out.append(
            CustomFixationPoint(
                point_index=int(raw["point_index"]),
                label=str(raw["label"]),
                t_start=float(max(0.0, t_start)),
                t_end=float(t_end),
                target_x_px=x_px,
                target_y_px=y_px,
                target_azimuth_deg=float(az),
                target_elevation_deg=float(el),
            )
        )
    return out


def load_custom_validation_log(
    path: str | Path,
    *,
    center_x: float,
    center_y: float,
    px_per_deg_x: float,
    px_per_deg_y: float,
    synctime_offset_sec: float = 0.0,
) -> list[CustomFixationPoint]:
    """Load Bassoon CustomValidation_Log_*.txt (times relative to SYNCTIME).

    ``synctime_offset_sec`` is SYNCTIME − first_sample in trial seconds
    (usually positive), so log times map onto the loaded ASC timeline.
    """
    path = Path(path)
    points: list[CustomFixationPoint] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("eventIndex"):
            continue
        parts = stripped.split("\t")
        if len(parts) < 8:
            continue
        try:
            point_index = int(parts[1])
            label = parts[2]
            rep = int(parts[3])
            t_start = float(parts[4]) + synctime_offset_sec
            t_end = float(parts[5]) + synctime_offset_sec
            x_px = float(parts[6])
            y_px = float(parts[7])
        except (ValueError, IndexError):
            continue
        az = (x_px - center_x) / px_per_deg_x
        el = (center_y - y_px) / px_per_deg_y
        points.append(
            CustomFixationPoint(
                point_index=point_index,
                label=label,
                t_start=t_start,
                t_end=t_end,
                target_x_px=x_px,
                target_y_px=y_px,
                target_azimuth_deg=float(az),
                target_elevation_deg=float(el),
                rep=rep,
            )
        )
    points.sort(key=lambda p: p.t_start)
    return points


def rms_sample_to_sample(x: np.ndarray, y: np.ndarray) -> float:
    """RMS of successive 2D sample-to-sample displacements (degrees)."""
    if len(x) < 2:
        return float("nan")
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sqrt(np.mean(dx * dx + dy * dy)))


def covariance_ellipse_points(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_std: float = 1.0,
    n_points: int = 72,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (xs, ys) for an n_std covariance ellipse around the sample cloud.

    Uses the eigen-decomposition of the 2×2 covariance (axis-aligned in the
    principal-component frame). ``n_std=1`` is the usual 1σ ellipse.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or n_std <= 0:
        return None
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return None
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    order = vals.argsort()[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    a = float(n_std) * float(np.sqrt(vals[0]))
    b = float(n_std) * float(np.sqrt(vals[1]))
    # Degenerate / near-zero variance: still draw a tiny circle so the point shows.
    if a < 1e-6 and b < 1e-6:
        a = b = 1e-3
    local = np.vstack((a * np.cos(theta), b * np.sin(theta)))
    world = vecs @ local
    return mean_x + world[0], mean_y + world[1]

def precision_accuracy_for_window(
    times: np.ndarray,
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    *,
    t_start: float,
    t_end: float,
    target_azimuth_deg: float,
    target_elevation_deg: float,
    point_index: int,
    label: str,
    segment_id: int | None = None,
    settle_sec: float = 0.0,
    min_samples: int = 5,
) -> FixationPrecisionResult | None:
    """Compute SD, RMS-S2S, and signed/unsigned error on binocular gaze."""
    times = np.asarray(times, dtype=float)
    az = np.asarray(azimuth_deg, dtype=float)
    el = np.asarray(elevation_deg, dtype=float)
    lo = float(t_start) + max(0.0, float(settle_sec))
    hi = float(t_end)
    if hi <= lo:
        return None
    mask = (times >= lo) & (times <= hi) & np.isfinite(az) & np.isfinite(el)
    if int(np.count_nonzero(mask)) < min_samples:
        return None
    az_s = az[mask]
    el_s = el[mask]
    mean_az = float(np.mean(az_s))
    mean_el = float(np.mean(el_s))
    sd_az = float(np.std(az_s, ddof=1)) if len(az_s) > 1 else 0.0
    sd_el = float(np.std(el_s, ddof=1)) if len(el_s) > 1 else 0.0
    sd_2d = float(np.sqrt(sd_az * sd_az + sd_el * sd_el))
    signed_az = mean_az - float(target_azimuth_deg)
    signed_el = mean_el - float(target_elevation_deg)
    unsigned = float(np.hypot(signed_az, signed_el))
    return FixationPrecisionResult(
        point_index=int(point_index),
        label=str(label),
        segment_id=segment_id,
        t_start=lo,
        t_end=hi,
        n_samples=int(len(az_s)),
        target_azimuth_deg=float(target_azimuth_deg),
        target_elevation_deg=float(target_elevation_deg),
        mean_azimuth_deg=mean_az,
        mean_elevation_deg=mean_el,
        sd_azimuth_deg=sd_az,
        sd_elevation_deg=sd_el,
        sd_2d_deg=sd_2d,
        rms_s2s_deg=rms_sample_to_sample(az_s, el_s),
        signed_error_azimuth_deg=signed_az,
        signed_error_elevation_deg=signed_el,
        unsigned_error_deg=unsigned,
        azimuth_samples_deg=az_s,
        elevation_samples_deg=el_s,
    )


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def match_segment_to_point(
    t_start: float,
    t_end: float,
    points: list[CustomFixationPoint],
) -> CustomFixationPoint | None:
    """Pick the CustomValidation point with greatest time overlap (else nearest)."""
    if not points:
        return None
    best: CustomFixationPoint | None = None
    best_overlap = -1.0
    for pt in points:
        ov = _overlap(t_start, t_end, pt.t_start, pt.t_end)
        if ov > best_overlap:
            best_overlap = ov
            best = pt
    if best is not None and best_overlap > 0:
        return best
    seg_mid = 0.5 * (t_start + t_end)
    return min(points, key=lambda p: abs(0.5 * (p.t_start + p.t_end) - seg_mid))


def compute_precision_from_segments(
    times: np.ndarray,
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    segments: list,
    points: list[CustomFixationPoint],
    *,
    settle_sec: float = 0.0,
) -> list[FixationPrecisionResult]:
    """Match accepted segments to targets and compute metrics."""
    results: list[FixationPrecisionResult] = []
    for seg in segments:
        pt = match_segment_to_point(seg.t_start, seg.t_end, points)
        if pt is None:
            continue
        result = precision_accuracy_for_window(
            times,
            azimuth_deg,
            elevation_deg,
            t_start=seg.t_start,
            t_end=seg.t_end,
            target_azimuth_deg=pt.target_azimuth_deg,
            target_elevation_deg=pt.target_elevation_deg,
            point_index=pt.point_index,
            label=pt.label,
            segment_id=getattr(seg, "segment_id", None),
            settle_sec=settle_sec,
        )
        if result is not None:
            results.append(result)
    results.sort(key=lambda r: (r.t_start, r.point_index))
    return results


def compute_precision_from_asc_windows(
    times: np.ndarray,
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    points: list[CustomFixationPoint],
    *,
    settle_sec: float = 0.5,
) -> list[FixationPrecisionResult]:
    """Use ASC CUSTOM_VAL POINT windows (with optional settle trim)."""
    results: list[FixationPrecisionResult] = []
    for pt in points:
        result = precision_accuracy_for_window(
            times,
            azimuth_deg,
            elevation_deg,
            t_start=pt.t_start,
            t_end=pt.t_end,
            target_azimuth_deg=pt.target_azimuth_deg,
            target_elevation_deg=pt.target_elevation_deg,
            point_index=pt.point_index,
            label=pt.label,
            segment_id=None,
            settle_sec=settle_sec,
        )
        if result is not None:
            results.append(result)
    return results


def summarize_precision_results(
    results: list[FixationPrecisionResult],
) -> dict[str, float]:
    """Mean of per-point metrics plus spatial homogeneity across points."""
    empty = {
        "n_points": 0.0,
        "mean_sd_2d_deg": float("nan"),
        "mean_rms_s2s_deg": float("nan"),
        "mean_signed_error_azimuth_deg": float("nan"),
        "mean_signed_error_elevation_deg": float("nan"),
        "mean_unsigned_error_deg": float("nan"),
        "mean_abs_signed_azimuth_deg": float("nan"),
        "mean_abs_signed_elevation_deg": float("nan"),
        "max_unsigned_error_deg": float("nan"),
        "min_unsigned_error_deg": float("nan"),
        # Homogeneity = how consistent metrics are across the 9 targets
        # (lower = more uniform spatial quality).
        "homog_unsigned_sd_deg": float("nan"),
        "homog_unsigned_range_deg": float("nan"),
        "homog_sd2d_sd_deg": float("nan"),
        "homog_rms_sd_deg": float("nan"),
    }
    if not results:
        return empty

    def _nanmean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        if not np.any(np.isfinite(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    def _nanstd(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite) < 2:
            return 0.0 if len(finite) == 1 else float("nan")
        return float(np.std(finite, ddof=1))

    unsigned = [r.unsigned_error_deg for r in results]
    sd2d = [r.sd_2d_deg for r in results]
    rms = [r.rms_s2s_deg for r in results]
    unsigned_arr = np.asarray(unsigned, dtype=float)
    finite_u = unsigned_arr[np.isfinite(unsigned_arr)]
    if len(finite_u) == 0:
        u_min = u_max = float("nan")
        u_range = float("nan")
    else:
        u_min = float(np.min(finite_u))
        u_max = float(np.max(finite_u))
        u_range = u_max - u_min

    return {
        "n_points": float(len(results)),
        "mean_sd_2d_deg": _nanmean(sd2d),
        "mean_rms_s2s_deg": _nanmean(rms),
        "mean_signed_error_azimuth_deg": _nanmean(
            [r.signed_error_azimuth_deg for r in results]
        ),
        "mean_signed_error_elevation_deg": _nanmean(
            [r.signed_error_elevation_deg for r in results]
        ),
        "mean_unsigned_error_deg": _nanmean(unsigned),
        "mean_abs_signed_azimuth_deg": _nanmean(
            [abs(r.signed_error_azimuth_deg) for r in results]
        ),
        "mean_abs_signed_elevation_deg": _nanmean(
            [abs(r.signed_error_elevation_deg) for r in results]
        ),
        "max_unsigned_error_deg": u_max,
        "min_unsigned_error_deg": u_min,
        "homog_unsigned_sd_deg": _nanstd(unsigned),
        "homog_unsigned_range_deg": u_range,
        "homog_sd2d_sd_deg": _nanstd(sd2d),
        "homog_rms_sd_deg": _nanstd(rms),
    }
