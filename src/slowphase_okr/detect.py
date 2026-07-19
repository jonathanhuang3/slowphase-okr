"""Conservative semi-automatic slow-phase detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slowphase_okr.fit import SegmentFit, fit_segment
from slowphase_okr.okr_log import OkrLog


@dataclass(frozen=True)
class DetectParams:
    direction: str = "auto"  # "up", "down", or "auto" (per OKR block when available)
    min_duration_sec: float = 0.05
    min_r2: float = 0.75
    max_saccade_velocity_deg_s: float = 100.0
    restrict_to_blocks: bool = False
    merge_gap_sec: float = 0.04
    window_sec: float = 0.10
    refine_boundaries: bool = True
    collapse_duplicate_times: bool = True


@dataclass(frozen=True)
class DetectWindow:
    start: float
    end: float
    direction: str | None = None  # "up" / "down" from OKR block when auto


@dataclass(frozen=True)
class CollapsedSeries:
    times: np.ndarray
    elevation_deg: np.ndarray
    first_orig_idx: np.ndarray
    last_orig_idx: np.ndarray


def _okr_direction_to_detect(direction: str | None) -> str | None:
    # "up"/"down" mean positive/negative slope in the active signal.
    # For azimuth: Right → increasing (positive), Left → decreasing (negative).
    if direction in ("Up", "Right"):
        return "up"
    if direction in ("Down", "Left"):
        return "down"
    return None


def _search_windows(
    t0: float,
    t1: float,
    okr_log: OkrLog | None,
    restrict_to_blocks: bool,
) -> list[DetectWindow]:
    if not restrict_to_blocks or okr_log is None or not okr_log.block_markers:
        return [DetectWindow(t0, t1)]

    event_starts = sorted(
        {m.start_time for m in okr_log.block_markers}
        | {m.start_time for m in okr_log.fixation_markers}
    )
    windows: list[DetectWindow] = []
    for block in sorted(okr_log.block_markers, key=lambda m: m.start_time):
        later = [s for s in event_starts if s > block.start_time]
        end = later[0] if later else t1
        start = max(block.start_time, t0)
        end = min(end, t1)
        if end > start:
            windows.append(
                DetectWindow(
                    start,
                    end,
                    direction=_okr_direction_to_detect(block.direction),
                )
            )
    return windows


def _resolve_direction(params: DetectParams, window: DetectWindow) -> str:
    if params.direction in ("up", "down"):
        return params.direction
    if window.direction in ("up", "down"):
        return window.direction
    return "up"


def collapse_duplicate_timestamps(
    times: np.ndarray,
    elevation_deg: np.ndarray,
    valid_mask: np.ndarray,
) -> CollapsedSeries:
    """Average elevation at repeated timestamps; keep first/last original indices."""
    valid_idx = np.where(valid_mask & ~np.isnan(elevation_deg))[0]
    if len(valid_idx) == 0:
        return CollapsedSeries(
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )

    t = times[valid_idx]
    e = elevation_deg[valid_idx]
    order = np.argsort(t, kind="stable")
    t = t[order]
    e = e[order]
    orig = valid_idx[order]

    unique_times: list[float] = []
    mean_elev: list[float] = []
    first_orig: list[int] = []
    last_orig: list[int] = []

    i = 0
    while i < len(t):
        j = i + 1
        while j < len(t) and t[j] == t[i]:
            j += 1
        unique_times.append(float(t[i]))
        mean_elev.append(float(np.mean(e[i:j])))
        first_orig.append(int(orig[i]))
        last_orig.append(int(orig[j - 1]))
        i = j

    return CollapsedSeries(
        times=np.array(unique_times, dtype=float),
        elevation_deg=np.array(mean_elev, dtype=float),
        first_orig_idx=np.array(first_orig, dtype=int),
        last_orig_idx=np.array(last_orig, dtype=int),
    )


def _segment_has_saccade(
    times: np.ndarray,
    elevation_deg: np.ndarray,
    idx_start: int,
    idx_end: int,
    max_velocity_deg_s: float,
) -> bool:
    seg_times = times[idx_start : idx_end + 1]
    seg_elev = elevation_deg[idx_start : idx_end + 1]
    valid = ~np.isnan(seg_elev)
    if np.count_nonzero(valid) < 2:
        return True
    velocity = np.gradient(seg_elev[valid], seg_times[valid])
    return bool(np.max(np.abs(velocity)) > max_velocity_deg_s)


def _fit_passes_direction(fit: SegmentFit, direction: str) -> bool:
    if direction == "up":
        return fit.direction_upward
    return not fit.direction_upward


def _try_fit(
    times: np.ndarray,
    elevation_deg: np.ndarray,
    idx_start: int,
    idx_end: int,
    stimulus_velocity: float,
    segment_id: int,
) -> SegmentFit | None:
    try:
        return fit_segment(
            times,
            elevation_deg,
            idx_start,
            idx_end,
            stimulus_velocity,
            segment_id=segment_id,
        )
    except ValueError:
        return None


def _refine_segment_bounds(
    times: np.ndarray,
    elevation_deg: np.ndarray,
    collapsed: CollapsedSeries,
    pos_start: int,
    pos_end: int,
    stimulus_velocity: float,
    params: DetectParams,
    direction: str,
) -> tuple[int, int] | None:
    """Expand collapsed bounds while R² stays above min_r2 and direction holds."""
    n = len(collapsed.times)
    if pos_end <= pos_start:
        return None

    best_start = pos_start
    best_end = pos_end
    best_fit = _try_fit(
        times,
        elevation_deg,
        int(collapsed.first_orig_idx[best_start]),
        int(collapsed.last_orig_idx[best_end]),
        stimulus_velocity,
        1,
    )
    if best_fit is None or best_fit.r2 != best_fit.r2 or best_fit.r2 < params.min_r2:
        return None
    if not _fit_passes_direction(best_fit, direction):
        return None

    improved = True
    while improved:
        improved = False
        if best_start > 0:
            cand_start = best_start - 1
            cand_fit = _try_fit(
                times,
                elevation_deg,
                int(collapsed.first_orig_idx[cand_start]),
                int(collapsed.last_orig_idx[best_end]),
                stimulus_velocity,
                1,
            )
            if (
                cand_fit is not None
                and cand_fit.r2 == cand_fit.r2
                and cand_fit.r2 >= params.min_r2
                and cand_fit.r2 >= best_fit.r2 - 1e-6
                and _fit_passes_direction(cand_fit, direction)
                and not _segment_has_saccade(
                    times,
                    elevation_deg,
                    int(collapsed.first_orig_idx[cand_start]),
                    int(collapsed.last_orig_idx[best_end]),
                    params.max_saccade_velocity_deg_s,
                )
            ):
                best_start = cand_start
                best_fit = cand_fit
                improved = True

        if best_end < n - 1:
            cand_end = best_end + 1
            cand_fit = _try_fit(
                times,
                elevation_deg,
                int(collapsed.first_orig_idx[best_start]),
                int(collapsed.last_orig_idx[cand_end]),
                stimulus_velocity,
                1,
            )
            if (
                cand_fit is not None
                and cand_fit.r2 == cand_fit.r2
                and cand_fit.r2 >= params.min_r2
                and cand_fit.r2 >= best_fit.r2 - 1e-6
                and _fit_passes_direction(cand_fit, direction)
                and not _segment_has_saccade(
                    times,
                    elevation_deg,
                    int(collapsed.first_orig_idx[best_start]),
                    int(collapsed.last_orig_idx[cand_end]),
                    params.max_saccade_velocity_deg_s,
                )
            ):
                best_end = cand_end
                best_fit = cand_fit
                improved = True

    return best_start, best_end


def _merge_position_ranges(
    hits: list[tuple[int, int]],
    times: np.ndarray,
    merge_gap_sec: float,
) -> list[tuple[int, int]]:
    if not hits:
        return []
    hits = sorted(hits, key=lambda h: (h[0], h[1]))
    merged: list[list[int]] = [[hits[0][0], hits[0][1]]]
    for start, end in hits[1:]:
        prev_start, prev_end = merged[-1]
        gap = float(times[start] - times[prev_end])
        if start <= prev_end + 1 or gap <= merge_gap_sec:
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start <= b_end and b_start <= a_end


def _drop_overlapping(candidates: list[SegmentFit]) -> list[SegmentFit]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda s: (-(s.t_end - s.t_start), s.t_start))
    kept: list[SegmentFit] = []
    for cand in ordered:
        if any(
            _overlaps(cand.t_start, cand.t_end, kept_seg.t_start, kept_seg.t_end)
            for kept_seg in kept
        ):
            continue
        kept.append(cand)
    return sorted(kept, key=lambda s: s.t_start)


def _sliding_window_hits(
    collapsed: CollapsedSeries,
    window_sec: float,
    params: DetectParams,
    direction: str,
    times: np.ndarray,
    elevation_deg: np.ndarray,
    stimulus_velocity: float,
) -> list[tuple[int, int]]:
    n = len(collapsed.times)
    if n < 3:
        return []

    hits: list[tuple[int, int]] = []
    for i in range(n):
        t0 = collapsed.times[i]
        j = i + 1
        while j < n and collapsed.times[j] - t0 < window_sec:
            j += 1
        if j < n:
            end_pos = j
        else:
            end_pos = n - 1
        if collapsed.times[end_pos] - t0 < window_sec:
            continue

        idx_start = int(collapsed.first_orig_idx[i])
        idx_end = int(collapsed.last_orig_idx[end_pos])
        if float(times[idx_end] - times[idx_start]) < params.min_duration_sec:
            continue
        if _segment_has_saccade(
            times, elevation_deg, idx_start, idx_end, params.max_saccade_velocity_deg_s
        ):
            continue

        fit = _try_fit(times, elevation_deg, idx_start, idx_end, stimulus_velocity, 1)
        if fit is None or fit.r2 != fit.r2 or fit.r2 < params.min_r2:
            continue
        if not _fit_passes_direction(fit, direction):
            continue
        hits.append((i, end_pos))

    return hits


def detect_slow_phases(
    times: np.ndarray,
    elevation_deg: np.ndarray,
    valid_mask: np.ndarray,
    stimulus_velocity: float,
    params: DetectParams,
    *,
    okr_log: OkrLog | None = None,
    exclude: list[SegmentFit] | None = None,
) -> list[SegmentFit]:
    """Propose slow-phase segments sorted by start time."""
    if params.direction not in ("up", "down", "auto"):
        raise ValueError("direction must be 'up', 'down', or 'auto'")
    if params.min_duration_sec <= 0:
        raise ValueError("min_duration_sec must be positive")
    if params.max_saccade_velocity_deg_s <= 0:
        raise ValueError("max_saccade_velocity_deg_s must be positive")
    if params.window_sec <= 0:
        raise ValueError("window_sec must be positive")
    if params.merge_gap_sec < 0:
        raise ValueError("merge_gap_sec must be non-negative")
    if params.direction == "auto" and not params.restrict_to_blocks:
        raise ValueError(
            "Auto direction requires contrast-block windows from an OKR log."
        )

    exclude = exclude or []
    t0 = float(times[0])
    t1 = float(times[-1])
    windows = _search_windows(t0, t1, okr_log, params.restrict_to_blocks)

    if params.collapse_duplicate_times:
        collapsed = collapse_duplicate_timestamps(times, elevation_deg, valid_mask)
    else:
        valid_idx = np.where(valid_mask & ~np.isnan(elevation_deg))[0]
        collapsed = CollapsedSeries(
            times=times[valid_idx],
            elevation_deg=elevation_deg[valid_idx],
            first_orig_idx=valid_idx.astype(int),
            last_orig_idx=valid_idx.astype(int),
        )

    candidates: list[SegmentFit] = []
    next_id = 1

    for window in windows:
        direction = _resolve_direction(params, window)
        in_window = (collapsed.times >= window.start) & (collapsed.times <= window.end)
        positions = np.where(in_window)[0]
        if len(positions) < 3:
            continue

        sub = CollapsedSeries(
            times=collapsed.times[positions],
            elevation_deg=collapsed.elevation_deg[positions],
            first_orig_idx=collapsed.first_orig_idx[positions],
            last_orig_idx=collapsed.last_orig_idx[positions],
        )

        hits = _sliding_window_hits(
            sub,
            params.window_sec,
            params,
            direction,
            times,
            elevation_deg,
            stimulus_velocity,
        )
        merged = _merge_position_ranges(hits, sub.times, params.merge_gap_sec)

        for pos_start, pos_end in merged:
            if params.refine_boundaries:
                refined = _refine_segment_bounds(
                    times,
                    elevation_deg,
                    sub,
                    pos_start,
                    pos_end,
                    stimulus_velocity,
                    params,
                    direction,
                )
                if refined is None:
                    continue
                pos_start, pos_end = refined

            idx_start = int(sub.first_orig_idx[pos_start])
            idx_end = int(sub.last_orig_idx[pos_end])
            duration = float(times[idx_end] - times[idx_start])
            if duration < params.min_duration_sec:
                continue
            if _segment_has_saccade(
                times,
                elevation_deg,
                idx_start,
                idx_end,
                params.max_saccade_velocity_deg_s,
            ):
                continue

            fit = _try_fit(
                times,
                elevation_deg,
                idx_start,
                idx_end,
                stimulus_velocity,
                next_id,
            )
            if fit is None or fit.r2 != fit.r2 or fit.r2 < params.min_r2:
                continue
            if not _fit_passes_direction(fit, direction):
                continue
            if any(_overlaps(fit.t_start, fit.t_end, ex.t_start, ex.t_end) for ex in exclude):
                continue
            if any(
                _overlaps(fit.t_start, fit.t_end, cand.t_start, cand.t_end)
                for cand in candidates
            ):
                continue

            candidates.append(fit)
            next_id += 1

    return _drop_overlapping(candidates)
