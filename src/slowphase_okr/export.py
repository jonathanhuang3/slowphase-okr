"""Export segment fits and trial summaries to Excel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from slowphase_okr.fit import SegmentFit, trial_summary_median_gain


def export_to_excel(
    segments: list[SegmentFit],
    trial_id: str,
    software_version: str,
    output_path: str | Path,
    gaze_source: str = "",
    time_source: str = "",
) -> Path:
    """Write segment-level sheet and trial summary sheet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if segments:
        seg_rows = [s.to_row(trial_id, software_version) for s in segments]
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
    }
    if segments:
        summary["median_r2"] = float(seg_df["r2"].median())
        summary["median_slope_deg_s"] = float(seg_df["slope_deg_s"].median())
        summary["n_upward"] = int(seg_df["direction_upward"].sum())
        summary["n_not_upward"] = int((~seg_df["direction_upward"]).sum())

    summary_df = pd.DataFrame([summary])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        seg_df.to_excel(writer, sheet_name="segments", index=False)
        summary_df.to_excel(writer, sheet_name="trial_summary", index=False)

    return output_path
