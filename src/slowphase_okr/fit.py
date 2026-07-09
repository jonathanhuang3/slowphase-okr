"""Linear fit and OKR gain for marked slow-phase segments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass
class SegmentFit:
    """One manually marked slow-phase segment."""

    segment_id: int
    idx_start: int
    idx_end: int
    t_start: float
    t_end: float
    n_samples: int
    slope_deg_s: float
    intercept_deg: float
    gain: float
    r2: float
    direction_upward: bool
    stimulus_velocity: float

    def to_row(self, trial_id: str, software_version: str) -> dict[str, Any]:
        row = asdict(self)
        row["trial_id"] = trial_id
        row["software_version"] = software_version
        return row


def fit_segment(
    times: np.ndarray,
    elevation_deg: np.ndarray,
    idx_start: int,
    idx_end: int,
    stimulus_velocity: float,
    segment_id: int,
    min_samples: int = 3,
) -> SegmentFit:
    """Fit elevation vs time on inclusive index range [idx_start, idx_end]."""
    if idx_end < idx_start:
        idx_start, idx_end = idx_end, idx_start

    seg_times = times[idx_start : idx_end + 1]
    seg_elev = elevation_deg[idx_start : idx_end + 1]
    valid = ~np.isnan(seg_elev)
    seg_times = seg_times[valid]
    seg_elev = seg_elev[valid]

    if len(seg_times) < min_samples:
        raise ValueError(
            f"Segment has only {len(seg_times)} valid samples (need >= {min_samples})"
        )

    slope, intercept = np.polyfit(seg_times, seg_elev, 1)
    predicted = slope * seg_times + intercept
    ss_tot = np.sum((seg_elev - np.mean(seg_elev)) ** 2)
    if ss_tot <= 0:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum((seg_elev - predicted) ** 2) / ss_tot)

    return SegmentFit(
        segment_id=segment_id,
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        t_start=float(seg_times[0]),
        t_end=float(seg_times[-1]),
        n_samples=int(len(seg_times)),
        slope_deg_s=float(slope),
        intercept_deg=float(intercept),
        gain=float(slope / stimulus_velocity),
        r2=r2,
        direction_upward=bool(slope > 0),
        stimulus_velocity=float(stimulus_velocity),
    )


def trial_summary_median_gain(segments: list[SegmentFit]) -> float:
    """Median gain across accepted segments."""
    if not segments:
        return float("nan")
    return float(np.median([s.gain for s in segments]))


def refit_segment_by_time(
    times: np.ndarray,
    values: np.ndarray,
    t_start: float,
    t_end: float,
    stimulus_velocity: float,
    segment_id: int,
    valid_mask: np.ndarray,
) -> SegmentFit:
    """Refit a segment using time boundaries on a (possibly different) signal."""
    idx_start = snap_index(times, t_start, valid_mask)
    idx_end = snap_index(times, t_end, valid_mask)
    return fit_segment(
        times,
        values,
        idx_start,
        idx_end,
        stimulus_velocity,
        segment_id,
    )


def snap_index(times: np.ndarray, click_time: float, valid_mask: np.ndarray) -> int:
    """Snap a click time to the nearest sample index within valid_mask."""
    candidates = np.where(valid_mask)[0]
    if len(candidates) == 0:
        raise ValueError("No samples in analysis window")
    dist = np.abs(times[candidates] - click_time)
    return int(candidates[int(np.argmin(dist))])
