"""unidirectionalUp.mlx-style net elevation displacement per OKR block.

Matches ``computeMlxDisplacementInWindow`` in ``okrGain_100pctFlickerUp_console.m``:

1. Zero frames with total (2D) gaze speed > threshold (default 40 deg/s)
2. Cumsum primary-axis displacements
3. ``medfilt1`` (order 3) + ~500 ms moving mean on velocity
4. Cumsum again; take end value (``net_disp``)
5. Normalize: ``norm_disp = net_disp / (1 - fraction_failures)``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slowphase_okr.okr_log import OkrLog, OkrLogBlockMarker, format_block_condition


@dataclass(frozen=True)
class NetDisplacementBlockResult:
    """Net-displacement summary for one stimulus block."""

    block_label: str
    condition: str
    direction: str | None
    start_time: float
    end_time: float
    axis: str
    n_samples: int
    pct_saccade_frames: float
    fraction_failures: float
    net_disp_deg: float
    norm_disp_deg: float
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


def sort_net_displacements_for_contrast_plot(
    results: list[NetDisplacementBlockResult],
) -> list[NetDisplacementBlockResult]:
    """Order blocks by contrast (low→high); persistent blocks last.

    Blocks without a contrast level sort after numbered contrasts within their
    flicker/persistent group. Start time breaks remaining ties.
    """

    def sort_key(result: NetDisplacementBlockResult) -> tuple[int, float, float]:
        persistent_rank = 1 if result.is_persistent else 0
        if result.contrast_level is None or not np.isfinite(result.contrast_level):
            contrast_rank = float("inf")
        else:
            contrast_rank = float(result.contrast_level)
        return (persistent_rank, contrast_rank, float(result.start_time))

    return sorted(results, key=sort_key)


def net_disp_contrast_plot_x_label(result: NetDisplacementBlockResult) -> str:
    """Short categorical x-axis label for the net-displacement contrast plot."""
    if result.contrast_level is not None and np.isfinite(result.contrast_level):
        contrast = f"{result.contrast_level:g}"
    else:
        contrast = "?"
    if result.is_persistent:
        return f"Persistent\n({contrast})"
    return contrast


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


def _primary_axis(direction: str | None) -> str:
    """Elevation for vertical OKR; azimuth for horizontal."""
    if direction in ("Left", "Right"):
        return "azimuth"
    return "elevation"


def _medfilt1(values: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Centered median filter (MATLAB ``medfilt1``-like, edge-padded)."""
    x = np.asarray(values, dtype=float)
    k = int(kernel_size)
    if k < 1:
        raise ValueError("kernel_size must be >= 1")
    if k % 2 == 0:
        k += 1
    if k == 1 or x.size == 0:
        return x.copy()
    pad = k // 2
    padded = np.pad(x, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(windows, axis=-1)


def _movmean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving mean with shrinking endpoints (MATLAB ``movmean`` default)."""
    x = np.asarray(values, dtype=float)
    n = int(x.size)
    w = max(1, int(window))
    if n == 0 or w == 1:
        return x.copy()
    half_left = (w - 1) // 2
    half_right = w // 2
    out = np.empty(n, dtype=float)
    csum = np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(x), x, 0.0))))
    finite = np.concatenate(([0], np.cumsum(np.isfinite(x).astype(int))))
    for i in range(n):
        left = max(0, i - half_left)
        right = min(n, i + half_right + 1)
        count = finite[right] - finite[left]
        if count == 0:
            out[i] = 0.0
        else:
            out[i] = (csum[right] - csum[left]) / count
    return out


def compute_mlx_net_displacement_for_window(
    times: np.ndarray,
    elevation_deg: np.ndarray,
    azimuth_deg: np.ndarray | None,
    *,
    start_time: float,
    end_time: float,
    skip_duration_s: float = 0.0,
    max_speed_deg_s: float = 40.0,
    smooth_window_s: float = 0.5,
    medfilt_order: int = 3,
    min_samples: int = 20,
    direction: str | None = "Up",
    block_label: str = "Full trial",
    condition: str = "Full trial",
    axis: str | None = None,
    contrast_level: float | None = None,
    is_persistent: bool | None = None,
) -> NetDisplacementBlockResult:
    """Compute mlx-style net displacement in one time window."""
    times = np.asarray(times, dtype=float)
    elevation_deg = np.asarray(elevation_deg, dtype=float)
    if azimuth_deg is None:
        azimuth = np.zeros_like(elevation_deg)
        have_az = False
    else:
        azimuth = np.asarray(azimuth_deg, dtype=float)
        have_az = True
    if times.ndim != 1 or elevation_deg.ndim != 1 or len(times) != len(elevation_deg):
        raise ValueError("times and elevation_deg must be equal-length 1D arrays")
    if have_az and len(azimuth) != len(times):
        raise ValueError("azimuth_deg must match times length")
    if len(times) < 2:
        raise ValueError("At least two gaze samples are required")
    if not np.isfinite(max_speed_deg_s) or max_speed_deg_s <= 0:
        raise ValueError("Max speed must be a positive number")
    if skip_duration_s < 0 or not np.isfinite(skip_duration_s):
        raise ValueError("skip_duration_s must be a non-negative finite number")

    axis_name = axis or _primary_axis(direction)
    if axis_name not in ("elevation", "azimuth"):
        raise ValueError(f"Unsupported axis {axis_name!r}")
    if axis_name == "azimuth" and not have_az:
        return NetDisplacementBlockResult(
            block_label=block_label,
            condition=condition,
            direction=direction,
            start_time=float(start_time),
            end_time=float(end_time),
            axis=axis_name,
            n_samples=0,
            pct_saccade_frames=float("nan"),
            fraction_failures=float("nan"),
            net_disp_deg=float("nan"),
            norm_disp_deg=float("nan"),
            error="Azimuth unavailable for this trial",
            contrast_level=contrast_level,
            is_persistent=is_persistent,
        )

    t0 = float(start_time) + float(skip_duration_s)
    t1 = float(end_time)
    in_samp = (times >= t0) & (times <= t1)
    n_samples = int(np.count_nonzero(in_samp))
    if n_samples < min_samples:
        return NetDisplacementBlockResult(
            block_label=block_label,
            condition=condition,
            direction=direction,
            start_time=float(start_time),
            end_time=float(end_time),
            axis=axis_name,
            n_samples=n_samples,
            pct_saccade_frames=float("nan"),
            fraction_failures=float("nan"),
            net_disp_deg=float("nan"),
            norm_disp_deg=float("nan"),
            error=f"Too few samples in window ({n_samples})",
            contrast_level=contrast_level,
            is_persistent=is_persistent,
        )

    el = elevation_deg[in_samp].copy()
    az = azimuth[in_samp].copy()
    tt = times[in_samp]
    fail_win = ~np.isfinite(el) | (have_az & ~np.isfinite(az))
    frac_fail = float(np.count_nonzero(fail_win) / max(len(fail_win), 1))
    if frac_fail >= 1.0:
        return NetDisplacementBlockResult(
            block_label=block_label,
            condition=condition,
            direction=direction,
            start_time=float(start_time),
            end_time=float(end_time),
            axis=axis_name,
            n_samples=n_samples,
            pct_saccade_frames=float("nan"),
            fraction_failures=frac_fail,
            net_disp_deg=float("nan"),
            norm_disp_deg=float("nan"),
            error="All samples are tracker failures",
            contrast_level=contrast_level,
            is_persistent=is_persistent,
        )

    az_disp = np.diff(az)
    el_disp = np.diff(el)
    frame_dwells = np.diff(tt)
    good_dw = (frame_dwells > 0) & np.isfinite(frame_dwells)
    if not np.any(good_dw):
        return NetDisplacementBlockResult(
            block_label=block_label,
            condition=condition,
            direction=direction,
            start_time=float(start_time),
            end_time=float(end_time),
            axis=axis_name,
            n_samples=n_samples,
            pct_saccade_frames=float("nan"),
            fraction_failures=frac_fail,
            net_disp_deg=float("nan"),
            norm_disp_deg=float("nan"),
            error="No positive frame dwells in window",
            contrast_level=contrast_level,
            is_persistent=is_persistent,
        )

    if have_az:
        total_disp = np.sqrt(az_disp * az_disp + el_disp * el_disp)
    else:
        total_disp = np.abs(el_disp)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = total_disp / frame_dwells
    sacc = (~np.isfinite(speed)) | (speed > max_speed_deg_s) | (~good_dw)
    az_disp = az_disp.copy()
    el_disp = el_disp.copy()
    az_disp[sacc] = 0.0
    el_disp[sacc] = 0.0
    nan_edge = (
        ~np.isfinite(el[:-1])
        | ~np.isfinite(el[1:])
        | (have_az & (~np.isfinite(az[:-1]) | ~np.isfinite(az[1:])))
    )
    az_disp[nan_edge] = 0.0
    el_disp[nan_edge] = 0.0
    pct_sacc = 100.0 * float(np.count_nonzero(sacc)) / max(int(sacc.size), 1)

    primary_disp = az_disp if axis_name == "azimuth" else el_disp
    no_sacc = np.cumsum(np.concatenate(([0.0], primary_disp)))
    filtered = _medfilt1(no_sacc, medfilt_order)
    filtered = np.where(np.isfinite(filtered), filtered, 0.0)

    mean_dw = float(np.mean(frame_dwells[good_dw]))
    with np.errstate(divide="ignore", invalid="ignore"):
        elev_velo = np.diff(filtered) / frame_dwells
    elev_velo = np.where(np.isfinite(elev_velo), elev_velo, 0.0)
    win_samps = max(1, int(round(smooth_window_s * round(1.0 / mean_dw))))
    elev_velo2 = np.concatenate(([0.0], _movmean(elev_velo, win_samps))) * mean_dw
    elevation_final = np.cumsum(elev_velo2)
    net_disp = float(elevation_final[-1])
    norm_disp = net_disp / (1.0 - frac_fail)

    return NetDisplacementBlockResult(
        block_label=block_label,
        condition=condition,
        direction=direction,
        start_time=float(start_time),
        end_time=float(end_time),
        axis=axis_name,
        n_samples=n_samples,
        pct_saccade_frames=pct_sacc,
        fraction_failures=frac_fail,
        net_disp_deg=net_disp,
        norm_disp_deg=norm_disp,
        error="",
        contrast_level=contrast_level,
        is_persistent=is_persistent,
    )


def compute_mlx_net_displacements_by_block(
    times: np.ndarray,
    elevation_deg: np.ndarray,
    azimuth_deg: np.ndarray | None = None,
    *,
    okr_log: OkrLog | None = None,
    skip_duration_s: float = 0.0,
    max_speed_deg_s: float = 40.0,
    smooth_window_s: float = 0.5,
    medfilt_order: int = 3,
    min_samples: int = 20,
) -> list[NetDisplacementBlockResult]:
    """Compute mlx-style net displacement for every logged block (or full trial)."""
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        raise ValueError("At least two gaze samples are required")

    if okr_log is None or not okr_log.block_markers:
        return [
            compute_mlx_net_displacement_for_window(
                times,
                elevation_deg,
                azimuth_deg,
                start_time=float(times[0]),
                end_time=float(times[-1]),
                skip_duration_s=skip_duration_s,
                max_speed_deg_s=max_speed_deg_s,
                smooth_window_s=smooth_window_s,
                medfilt_order=medfilt_order,
                min_samples=min_samples,
                direction="Up",
                block_label="Full trial",
                condition="No OKR log - elevation net displacement",
                axis="elevation",
            )
        ]

    trial_start = float(times[0])
    trial_end = float(times[-1])
    results: list[NetDisplacementBlockResult] = []
    for marker in sorted(okr_log.block_markers, key=lambda item: item.start_time):
        start = max(float(marker.start_time), trial_start)
        end = _block_end(marker, okr_log, trial_end)
        results.append(
            compute_mlx_net_displacement_for_window(
                times,
                elevation_deg,
                azimuth_deg,
                start_time=start,
                end_time=end,
                skip_duration_s=skip_duration_s,
                max_speed_deg_s=max_speed_deg_s,
                smooth_window_s=smooth_window_s,
                medfilt_order=medfilt_order,
                min_samples=min_samples,
                direction=marker.direction,
                block_label=marker.label,
                condition=format_block_condition(marker),
                axis=_primary_axis(marker.direction),
                contrast_level=marker.contrast_level,
                is_persistent=_marker_is_persistent(marker),
            )
        )
    return results
