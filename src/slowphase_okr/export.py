"""Export segment fits and trial summaries to Excel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from slowphase_okr.fit import (
    SegmentFit,
    summarize_gains_by_block,
    trial_summary_median_gain,
)
from slowphase_okr.okr_log import OkrLog, segment_condition_fields


def export_to_excel(
    segments: list[SegmentFit],
    trial_id: str,
    software_version: str,
    output_path: str | Path,
    gaze_source: str = "",
    time_source: str = "",
    okr_log: OkrLog | None = None,
) -> Path:
    """Write segment-level, trial summary, and per-block gain sheets."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if segments:
        seg_rows = []
        for s in segments:
            row = s.to_row(trial_id, software_version)
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
                "n_samples",
                "slope_deg_s",
                "intercept_deg",
                "gain",
                "r2",
                "direction_upward",
                "stimulus_velocity",
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
                "eye_patch",
                "viewing_eye",
                "condition",
                "session_tags",
            ]
        )

    summary = {
        "trial_id": trial_id,
        "software_version": software_version,
        "n_segments": len(segments),
        "median_gain": trial_summary_median_gain(segments),
        "mean_gain": float(seg_df["gain"].mean()) if len(segments) else float("nan"),
        "gaze_source": gaze_source,
        "time_source": time_source,
        "okr_log_source": okr_log.source_path if okr_log is not None else "",
    }
    if segments:
        summary["median_r2"] = float(seg_df["r2"].median())
        summary["median_slope_deg_s"] = float(seg_df["slope_deg_s"].median())
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
                "eye_patch",
                "viewing_eye",
                "session_tags",
                "n_segments",
                "median_gain",
                "mean_gain",
                "median_r2",
                "median_slope_deg_s",
                "n_upward",
                "n_not_upward",
                "trial_id",
                "software_version",
            ]
        )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        seg_df.to_excel(writer, sheet_name="segments", index=False)
        by_block_df.to_excel(writer, sheet_name="by_block", index=False)
        summary_df.to_excel(writer, sheet_name="trial_summary", index=False)

    return output_path
