"""Conservative frame-wise OKR gain calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slowphase_okr.okr_log import OkrLog, OkrLogBlockMarker, format_block_condition


@dataclass(frozen=True)
class ConservativeBlockGain:
    """Conservative gain summary for one stimulus block."""

    block_label: str
    condition: str
    direction: str | None
    start_time: float
    end_time: float
    n_velocity_frames: int
    n_slow_frames: int
    pct_saccade_frames: float
    mean_velocity_deg_s: float
    gain: float
    gain_sd: float
    error: str = ""
    contrast_level: float | None = None
    is_persistent: bool | None = None


def _marker_is_persistent(marker: OkrLogBlockMarker) -> bool | None:
    event_lower = marker.event_type.lower()
    if "persistent" in event_lower:
        return True
    if "flicker" in event_lower:
        return False
    return marker.use_persistent_dots


def sort_conservative_gains_for_contrast_plot(
    results: list[ConservativeBlockGain],
) -> list[ConservativeBlockGain]:
    """Order blocks by contrast (low→high); persistent blocks last.

    Blocks without a contrast level sort after numbered contrasts within their
    flicker/persistent group. Start time breaks remaining ties.
    """

    def sort_key(result: ConservativeBlockGain) -> tuple[int, float, float]:
        persistent_rank = 1 if result.is_persistent else 0
        if result.contrast_level is None or not np.isfinite(result.contrast_level):
            contrast_rank = float("inf")
        else:
            contrast_rank = float(result.contrast_level)
        return (persistent_rank, contrast_rank, float(result.start_time))

    return sorted(results, key=sort_key)


def contrast_plot_x_label(result: ConservativeBlockGain) -> str:
    """Short categorical x-axis label for the gain-vs-contrast plot."""
    if result.contrast_level is not None and np.isfinite(result.contrast_level):
        contrast = f"{result.contrast_level:g}"
    else:
        contrast = "?"
    if result.is_persistent:
        return f"Persistent\n({contrast})"
    return contrast


def _direction_sign(direction: str | None) -> int:
    """Expected signal sign for a stimulus direction."""
    if direction in ("Down", "Left"):
        return -1
    return 1


def _block_end(
    marker: OkrLogBlockMarker,
    okr_log: OkrLog,
    trial_end: float,
) -> float:
    if marker.end_time is not None:
        return min(float(marker.end_time), trial_end)
    later_starts = [
        event.start_time
        for event in (*okr_log.block_markers, *okr_log.fixation_markers)
        if event.start_time > marker.start_time
    ]
    return min(later_starts, default=trial_end)


def compute_conservative_gain_for_window(
    times: np.ndarray,
    signal_deg: np.ndarray,
    *,
    start_time: float,
    end_time: float,
    stimulus_velocity_deg_s: float,
    saccade_threshold_deg_s: float = 20.0,
    direction: str | None = "Up",
    block_label: str = "Full trial",
    condition: str = "Full trial",
    min_slow_frames: int = 10,
    contrast_level: float | None = None,
    is_persistent: bool | None = None,
) -> ConservativeBlockGain:
    """Calculate Mean20-style gain in one time window.

    Velocity is ``diff(signal) / diff(time)`` at the earlier sample time, matching
    the MATLAB cohort calculation. Frames over the absolute velocity threshold are
    treated as saccades. Only velocity in the expected stimulus direction is
    averaged, and gain is reported as a positive direction-matched magnitude.
    ``gain_sd`` is the sample standard deviation of the per-frame gains
    (direction-matched velocity ÷ stimulus speed).
    """
    times = np.asarray(times, dtype=float)
    signal_deg = np.asarray(signal_deg, dtype=float)
    if times.ndim != 1 or signal_deg.ndim != 1 or len(times) != len(signal_deg):
        raise ValueError("times and signal_deg must be equal-length 1D arrays")
    if len(times) < 2:
        raise ValueError("At least two gaze samples are required")
    if not np.isfinite(stimulus_velocity_deg_s) or stimulus_velocity_deg_s == 0:
        raise ValueError("Stimulus velocity must be finite and non-zero")
    if not np.isfinite(saccade_threshold_deg_s) or saccade_threshold_deg_s <= 0:
        raise ValueError("Saccade threshold must be a positive number")

    frame_dwell = np.diff(times)
    displacement = np.diff(signal_deg)
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = displacement / frame_dwell
    velocity_times = times[:-1]
    in_window = (velocity_times >= start_time) & (velocity_times <= end_time)
    n_total = int(np.count_nonzero(in_window))

    invalid_or_saccade = ~np.isfinite(velocity) | (
        np.abs(velocity) > saccade_threshold_deg_s
    )
    n_saccade = int(np.count_nonzero(in_window & invalid_or_saccade))
    pct_saccade = 100.0 * n_saccade / n_total if n_total else float("nan")

    sign = _direction_sign(direction)
    slow = in_window & ~invalid_or_saccade & ((sign * velocity) > 0)
    selected = velocity[slow]
    n_slow = int(selected.size)
    error = ""
    if n_total == 0:
        error = "No velocity frames in block"
    elif n_slow < min_slow_frames:
        error = f"Too few direction-matched slow frames ({n_slow})"

    if n_slow:
        frame_gains = sign * selected / abs(stimulus_velocity_deg_s)
        mean_velocity = float(np.mean(selected))
        gain = float(np.mean(frame_gains))
        gain_sd = (
            float(np.std(frame_gains, ddof=1)) if n_slow >= 2 else float("nan")
        )
    else:
        mean_velocity = float("nan")
        gain = float("nan")
        gain_sd = float("nan")

    return ConservativeBlockGain(
        block_label=block_label,
        condition=condition,
        direction=direction,
        start_time=float(start_time),
        end_time=float(end_time),
        n_velocity_frames=n_total,
        n_slow_frames=n_slow,
        pct_saccade_frames=pct_saccade,
        mean_velocity_deg_s=mean_velocity,
        gain=gain,
        gain_sd=gain_sd,
        error=error,
        contrast_level=contrast_level,
        is_persistent=is_persistent,
    )


def compute_conservative_gains_by_block(
    times: np.ndarray,
    signal_deg: np.ndarray,
    *,
    stimulus_velocity_deg_s: float,
    saccade_threshold_deg_s: float = 20.0,
    okr_log: OkrLog | None = None,
    min_slow_frames: int = 10,
) -> list[ConservativeBlockGain]:
    """Calculate conservative gain for every logged block (or the full trial)."""
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        raise ValueError("At least two gaze samples are required")

    if okr_log is None or not okr_log.block_markers:
        return [
            compute_conservative_gain_for_window(
                times,
                signal_deg,
                start_time=float(times[0]),
                end_time=float(times[-1]),
                stimulus_velocity_deg_s=stimulus_velocity_deg_s,
                saccade_threshold_deg_s=saccade_threshold_deg_s,
                direction="Up",
                block_label="Full trial",
                condition="No OKR log - upward velocities",
                min_slow_frames=min_slow_frames,
            )
        ]

    trial_start = float(times[0])
    trial_end = float(times[-1])
    results: list[ConservativeBlockGain] = []
    for marker in sorted(okr_log.block_markers, key=lambda item: item.start_time):
        start = max(float(marker.start_time), trial_start)
        end = _block_end(marker, okr_log, trial_end)
        results.append(
            compute_conservative_gain_for_window(
                times,
                signal_deg,
                start_time=start,
                end_time=end,
                stimulus_velocity_deg_s=stimulus_velocity_deg_s,
                saccade_threshold_deg_s=saccade_threshold_deg_s,
                direction=marker.direction,
                block_label=marker.label,
                condition=format_block_condition(marker),
                min_slow_frames=min_slow_frames,
                contrast_level=marker.contrast_level,
                is_persistent=_marker_is_persistent(marker),
            )
        )
    return results
