"""Export segment fits and trial summaries to Excel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from slowphase_okr.fit import (
    SegmentFit,
    fit_segment,
    summarize_gains_by_block,
)
from slowphase_okr.okr_log import OkrLog, segment_condition_fields

# Per-segment elevation and azimuth fits. The UI-selected gain is not written.
DUAL_SIGNAL_COLUMNS = (
    "elevation_slope_deg_s",
    "elevation_gain",
    "azimuth_slope_deg_s",
    "azimuth_gain",
)
DUAL_BLOCK_COLUMNS = (
    "median_elevation_slope_deg_s",
    "median_elevation_gain",
    "mean_elevation_gain",
    "median_azimuth_slope_deg_s",
    "median_azimuth_gain",
    "mean_azimuth_gain",
)
# These repeat whichever signal is selected in the UI. Excel writes the
# elevation and azimuth fits instead. In-app summaries still use them.
SELECTED_SIGNAL_GAIN_COLUMNS = (
    "gain",
    "median_gain",
    "mean_gain",
    "median_slope_deg_s",
)


def _blank_axis() -> tuple[float, float]:
    return float("nan"), float("nan")


def _fit_axis(
    segment: SegmentFit,
    times: np.ndarray | None,
    values: np.ndarray | None,
) -> tuple[float, float]:
    """Slope and gain of one segment on one trace.

    A missing trace, a length mismatch, or too few finite samples leaves the
    pair blank instead of failing the export. This does not substitute another
    signal.
    """
    if times is None or values is None:
        return _blank_axis()
    try:
        times_arr = np.asarray(times, dtype=float)
        values_arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return _blank_axis()
    if (
        times_arr.ndim != 1
        or values_arr.ndim != 1
        or times_arr.shape != values_arr.shape
        or times_arr.size == 0
    ):
        return _blank_axis()
    idx_start = int(segment.idx_start)
    idx_end = int(segment.idx_end)
    if min(idx_start, idx_end) < 0 or max(idx_start, idx_end) >= times_arr.size:
        return _blank_axis()
    try:
        fit = fit_segment(
            times_arr,
            values_arr,
            idx_start,
            idx_end,
            float(segment.stimulus_velocity),
            int(segment.segment_id),
        )
    except (ValueError, IndexError, FloatingPointError):
        return _blank_axis()
    return float(fit.slope_deg_s), float(fit.gain)


def dual_signal_fields(
    segment: SegmentFit,
    times: np.ndarray | None = None,
    elevation_deg: np.ndarray | None = None,
    azimuth_deg: np.ndarray | None = None,
) -> dict[str, float]:
    """Fit ``segment`` on elevation and on azimuth.

    Either side is NaN when that trace is missing or cannot be fit.
    """
    elevation_slope, elevation_gain = _fit_axis(segment, times, elevation_deg)
    azimuth_slope, azimuth_gain = _fit_axis(segment, times, azimuth_deg)
    return {
        "elevation_slope_deg_s": elevation_slope,
        "elevation_gain": elevation_gain,
        "azimuth_slope_deg_s": azimuth_slope,
        "azimuth_gain": azimuth_gain,
    }


def _insert_before(row: dict, key: str, extra: dict) -> dict:
    if key not in row:
        merged = dict(row)
        merged.update(extra)
        return merged
    out: dict = {}
    for name, value in row.items():
        if name == key:
            out.update(extra)
        out[name] = value
    return out


def _finite_stat(values, statistic: str) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    if statistic == "mean":
        return float(np.mean(finite))
    return float(np.median(finite))


def _dual_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or "elevation_gain" not in frame.columns:
        return {name: float("nan") for name in DUAL_BLOCK_COLUMNS}
    return {
        "median_elevation_slope_deg_s": _finite_stat(
            frame["elevation_slope_deg_s"], "median"
        ),
        "median_elevation_gain": _finite_stat(frame["elevation_gain"], "median"),
        "mean_elevation_gain": _finite_stat(frame["elevation_gain"], "mean"),
        "median_azimuth_slope_deg_s": _finite_stat(
            frame["azimuth_slope_deg_s"], "median"
        ),
        "median_azimuth_gain": _finite_stat(frame["azimuth_gain"], "median"),
        "mean_azimuth_gain": _finite_stat(frame["azimuth_gain"], "mean"),
    }


def _with_id_tail(frame: pd.DataFrame) -> pd.DataFrame:
    tail = [name for name in ("trial_id", "software_version") if name in frame.columns]
    front = [name for name in frame.columns if name not in tail]
    return frame[front + tail]


def _without_selected_signal_gains(frame: pd.DataFrame) -> pd.DataFrame:
    drop = [name for name in SELECTED_SIGNAL_GAIN_COLUMNS if name in frame.columns]
    if not drop:
        return frame
    return frame.drop(columns=drop)


def export_to_excel(
    segments: list[SegmentFit],
    trial_id: str,
    software_version: str,
    output_path: str | Path,
    gaze_source: str = "",
    time_source: str = "",
    okr_log: OkrLog | None = None,
    times: np.ndarray | None = None,
    elevation_deg: np.ndarray | None = None,
    azimuth_deg: np.ndarray | None = None,
) -> Path:
    """Write segment-level, trial summary, and per-block gain sheets.

    Each segment is fit on ``elevation_deg`` and ``azimuth_deg`` when that
    trace is present. A missing trace, or a segment that cannot be fit on one
    side, leaves that side blank and does not fail the export. The workbook
    does not include the UI-selected ``gain`` or the summary columns that
    repeat it (``median_gain``, ``mean_gain``, ``median_slope_deg_s``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if segments:
        seg_rows = []
        for s in segments:
            row = s.to_row(trial_id, software_version)
            row = _insert_before(
                row,
                "trial_id",
                dual_signal_fields(s, times, elevation_deg, azimuth_deg),
            )
            row.update(segment_condition_fields(okr_log, s.t_start, s.t_end))
            seg_rows.append(row)
        seg_df = pd.DataFrame(seg_rows)
    else:
        seg_df = pd.DataFrame(
            columns=[
                "segment_id",
                "idx_start",
                "idx_end",
                "t_start",
                "t_end",
                "duration_s",
                "n_samples",
                "slope_deg_s",
                "velocity_deg_s",
                "intercept_deg",
                "r2",
                "rmse_deg",
                "direction_upward",
                "stimulus_velocity",
                *DUAL_SIGNAL_COLUMNS,
                "trial_id",
                "software_version",
                "block_label",
                "block_index",
                "event_type",
                "direction",
                "contrast_level",
                "threshold_multiplier",
                "is_anchor100",
                "flicker_mode",
                "dot_color",
                "dot_start_size",
                "emission_rate",
                "eye_patch",
                "viewing_eye",
                "condition",
                "session_tags",
            ]
        )

    dual_trial = _dual_summary(seg_df)
    summary = {
        "trial_id": trial_id,
        "software_version": software_version,
        "n_segments": len(segments),
        "median_elevation_gain": dual_trial["median_elevation_gain"],
        "mean_elevation_gain": dual_trial["mean_elevation_gain"],
        "median_elevation_slope_deg_s": dual_trial["median_elevation_slope_deg_s"],
        "median_azimuth_gain": dual_trial["median_azimuth_gain"],
        "mean_azimuth_gain": dual_trial["mean_azimuth_gain"],
        "median_azimuth_slope_deg_s": dual_trial["median_azimuth_slope_deg_s"],
        "gaze_source": gaze_source,
        "time_source": time_source,
        "okr_log_source": okr_log.source_path if okr_log is not None else "",
    }
    if segments:
        summary["median_r2"] = float(seg_df["r2"].median())
        summary["median_rmse_deg"] = float(seg_df["rmse_deg"].median())
        summary["n_upward"] = int(seg_df["direction_upward"].sum())
        summary["n_not_upward"] = int((~seg_df["direction_upward"]).sum())

    summary_df = pd.DataFrame([summary])

    block_summaries = summarize_gains_by_block(segments, okr_log)
    if block_summaries:
        by_block_df = pd.DataFrame(
            [b.to_row(trial_id, software_version) for b in block_summaries]
        )
    else:
        by_block_df = pd.DataFrame(
            columns=[
                "block_label",
                "block_index",
                "condition",
                "direction",
                "contrast_level",
                "threshold_multiplier",
                "is_anchor100",
                "flicker_mode",
                "dot_color",
                "dot_start_size",
                "emission_rate",
                "eye_patch",
                "viewing_eye",
                "session_tags",
                "n_segments",
                "median_r2",
                "median_rmse_deg",
                "mean_rmse_deg",
                "n_upward",
                "n_not_upward",
                *DUAL_BLOCK_COLUMNS,
                "trial_id",
                "software_version",
            ]
        )

    if not by_block_df.empty:
        grouped = {
            str(label): _dual_summary(group)
            for label, group in seg_df.groupby("block_label", sort=False)
        }
        for name in DUAL_BLOCK_COLUMNS:
            by_block_df[name] = [
                grouped.get(str(label), {}).get(name, float("nan"))
                for label in by_block_df["block_label"]
            ]
        by_block_df = _with_id_tail(by_block_df)

    seg_df = _without_selected_signal_gains(seg_df)
    by_block_df = _without_selected_signal_gains(by_block_df)
    summary_df = _without_selected_signal_gains(summary_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        seg_df.to_excel(writer, sheet_name="segments", index=False)
        by_block_df.to_excel(writer, sheet_name="by_block", index=False)
        summary_df.to_excel(writer, sheet_name="trial_summary", index=False)

    return output_path
