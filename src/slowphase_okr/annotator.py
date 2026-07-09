"""Tkinter + matplotlib GUI for manual OKR slow-phase annotation."""

from __future__ import annotations

from dataclasses import replace
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.transforms import blended_transform_factory

from slowphase_okr import __version__
from slowphase_okr.autosave import (
    autosave_matches_trial,
    autosave_path,
    load_autosave,
    save_autosave,
    segments_from_autosave,
)
from slowphase_okr.detect import DetectParams, detect_slow_phases
from slowphase_okr.export import export_to_excel
from slowphase_okr.fit import (
    SegmentFit,
    fit_segment,
    refit_segment_by_time,
    snap_index,
    trial_summary_median_gain,
)
from slowphase_okr.gaze import (
    GazeTrial,
    analysis_window_mask,
    attach_sranipal_pupil,
    infer_viewing_eye,
    load_ush2a_trial,
)
from slowphase_okr.okr_log import OkrLog, load_okr_log


HELP_TEXT = """Keyboard shortcuts
────────────────────────────────────────
Marking
  Click × 2        Mark slow-phase start, then end (snaps to nearest data point)
  A                Accept pending manual segment (priority), hovered proposed, or selected
  Esc              Clear pending segment (start over)

Navigation
  Scroll wheel     Pan time forward / backward (vertical scroll only)
  ← / →            Pan view by 20% of visible window
  View menu        1 s, 2 s, 5 s, 10 s, or Full trial window

Segments (select one in the list first)
  Del              Delete selected segment
  U                Undo last accepted segment
  [  ]             Nudge start to previous / next data point
  ,  .             Nudge end to previous / next data point

Other
  Enter            Apply stimulus velocity (in velocity field)
  ?                Show this help

Notes
  • Analysis window spans the full trial (first to last timestamp).
  • Signal menu: elevation (rotated gaze) or SRanipal pupil X/Y for an alternate gain trace.
  • Pupil gain uses the same slope / stimulus-velocity formula; slopes are in normalized pupil units/s.
  • R² is logged and exported but segments are not auto-rejected.
  • Annotations autosave to JSON in the trial folder; restore on reload.
  • Optional OKR log marks contrast-block and fixation-cross start times on the plot.
  • Auto-detect proposes segments (blue); review, nudge, accept (A), or delete.

Auto-detect review
  Propose segments   Sliding-window detector (see panel parameters)
  Direction Auto     Uses Up/Down from OKR log per contrast block
  N / P              Jump to next / previous proposed segment
  A                  Accept pending manual segment, hovered proposed, or selected proposed
  Del                Delete selected accepted or proposed segment

Selected segment
  Hover + A          Accept the proposed segment under the cursor
  Drag start/end     Hover near a segment edge to highlight it, then drag
  [  ]  ,  .         Nudge start / end by one data point
"""


DETECT_TOOLTIPS = {
    "direction": (
        "Which way the eye should move during a slow phase.\n"
        "Auto: read Up or Down from each contrast block in the OKR log.\n"
        "Up/Down: force one direction for the whole search."
    ),
    "saccade": (
        "Reject a candidate if eye speed spikes above this (deg/s) inside the segment.\n"
        "Saccades are fast jumps; slow phases are slower.\n"
        "Lower = stricter (fewer false positives). Typical range: 60–120."
    ),
    "duration": (
        "Shortest segment length to keep, in milliseconds.\n"
        "Shorter = more proposals; longer = fewer, longer segments only."
    ),
    "r2": (
        "Minimum straightness of the line fit (0 to 1).\n"
        "1.0 = perfectly linear. Higher = stricter. Try 0.6–0.8 for noisy data."
    ),
    "window": (
        "Length of each sliding analysis window, in milliseconds.\n"
        "Should be shorter than a typical slow phase but long enough to fit a line.\n"
        "Typical range: 80–150 ms."
    ),
    "merge_gap": (
        "Join nearby fragment proposals if they are separated by less than this (ms).\n"
        "Helps when one slow phase was split into two proposals."
    ),
    "refine": (
        "After detecting, expand segment start/end while the line fit stays good.\n"
        "Usually improves boundaries; turn off to see raw detector output."
    ),
    "blocks": (
        "Only search for slow phases during contrast blocks from the OKR log,\n"
        "not during fixation crosses. Requires an OKR log file."
    ),
    "propose": "Run auto-detect with the settings above. Results appear as proposed (?) segments.",
    "clear": "Remove all proposed segments without deleting accepted ones.",
}


class _ToolTip:
    """Hover tooltip for a widget (only one visible at a time)."""

    _active: "_ToolTip | None" = None
    _show_after_id: str | None = None

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 400) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule_show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    @classmethod
    def _cancel_scheduled(cls) -> None:
        if cls._active is not None and cls._show_after_id is not None:
            try:
                cls._active.widget.after_cancel(cls._show_after_id)
            except tk.TclError:
                pass
            cls._show_after_id = None

    @classmethod
    def _hide_active(cls) -> None:
        cls._cancel_scheduled()
        if cls._active is not None:
            if cls._active._tip is not None:
                cls._active._tip.destroy()
                cls._active._tip = None
            cls._active = None

    def _schedule_show(self, _event=None) -> None:
        self._cancel_scheduled()
        _ToolTip._active = self
        self._show_after_id = self.widget.after(self.delay_ms, self._show)

    def _show(self) -> None:
        self._show_after_id = None
        if _ToolTip._active is not self:
            return
        if self._tip is not None:
            return
        if not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tip,
            text=self.text,
            justify=tk.LEFT,
            relief=tk.SOLID,
            borderwidth=1,
            background="#ffffe0",
            wraplength=300,
            padx=6,
            pady=4,
        ).pack()

    def _hide(self, _event=None) -> None:
        if _ToolTip._active is self:
            _ToolTip._hide_active()


def _bind_tooltip(widget: tk.Widget, text: str) -> None:
    _ToolTip(widget, text)


class AnnotatorApp:
    SIGNAL_ELEVATION = "elevation"
    SIGNAL_PUPIL_Y = "pupil_y"
    SIGNAL_PUPIL_X = "pupil_x"
    SIGNAL_CHOICES = (
        ("Elevation (rotated gaze)", SIGNAL_ELEVATION),
        ("Pupil Y (SRanipal)", SIGNAL_PUPIL_Y),
        ("Pupil X (SRanipal)", SIGNAL_PUPIL_X),
    )

    DEFAULT_VIEW_DURATION = 2.0
    VIEW_PRESET_LABELS = ("1 s", "2 s", "5 s", "10 s", "Full trial")
    VIEW_PRESET_SECONDS = {
        "1 s": 1.0,
        "2 s": 2.0,
        "5 s": 5.0,
        "10 s": 10.0,
    }
    MIN_VIEW_DURATION = 0.5
    SCROLL_PAN_FRACTION = 0.2  # fraction of visible window per pan step

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"slowphase-okr v{__version__}")
        self.root.minsize(1100, 680)

        self.gaze_path: Path | None = None
        self.time_path: Path | None = None
        self.okr_log_path: Path | None = None
        self.okr_log: OkrLog | None = None
        self.trial: GazeTrial | None = None
        self.window_mask: np.ndarray | None = None  # type: ignore[name-defined]

        self.segments: list[SegmentFit] = []
        self.proposed_segments: list[SegmentFit] = []
        self.pending_start_idx: int | None = None
        self.pending_end_idx: int | None = None
        self.pending_fit: SegmentFit | None = None
        self.selected_segment_key: tuple[str, int] | None = None

        self._preview_line = None
        self._click_markers: list = []
        self._hover_idx: int | None = None
        self._hover_artists: list = []
        self._boundary_drag: dict | None = None
        self._hover_boundary: str | None = None
        self._hover_proposed_key: tuple[str, int] | None = None
        self._applied_stimulus_velocity: float = 31.0

        self.analysis_t0: float = 0.0
        self.analysis_t1: float = 0.0
        self.signal_ylim: tuple[float, float] | None = None
        self._signal_mode_map = {label: mode for label, mode in self.SIGNAL_CHOICES}
        self._signal_label_map = {mode: label for label, mode in self.SIGNAL_CHOICES}
        self.view_xmin: float | None = None
        self.view_xmax: float | None = None

        self._build_controls()
        self._build_main_area()
        self._build_plot()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._set_status("Load gaze and time files to begin.")

    @property
    def analysis_duration(self) -> float:
        return max(self.analysis_t1 - self.analysis_t0, self.MIN_VIEW_DURATION)

    def _signal_mode(self) -> str:
        return self._signal_mode_map.get(
            self.signal_var.get(), self.SIGNAL_ELEVATION
        )

    def _pupil_available(self) -> bool:
        return self.trial is not None and self.trial.pupil is not None

    def _active_times(self) -> np.ndarray:
        assert self.trial is not None
        mode = self._signal_mode()
        if mode in (self.SIGNAL_PUPIL_Y, self.SIGNAL_PUPIL_X):
            assert self.trial.pupil is not None
            return self.trial.pupil.times
        return self.trial.times

    def _active_values(self) -> np.ndarray:
        assert self.trial is not None
        mode = self._signal_mode()
        if mode == self.SIGNAL_PUPIL_Y:
            assert self.trial.pupil is not None
            return self.trial.pupil.y
        if mode == self.SIGNAL_PUPIL_X:
            assert self.trial.pupil is not None
            return self.trial.pupil.x
        return self.trial.elevation_deg

    def _signal_plot_label(self) -> str:
        mode = self._signal_mode()
        if mode == self.SIGNAL_PUPIL_Y:
            return "Pupil Y"
        if mode == self.SIGNAL_PUPIL_X:
            return "Pupil X"
        return "Elevation"

    def _signal_y_label(self) -> str:
        if self._signal_mode() == self.SIGNAL_ELEVATION:
            return "Elevation (deg)"
        return "Pupil position (norm.)"

    def _signal_slope_unit(self) -> str:
        if self._signal_mode() == self.SIGNAL_ELEVATION:
            return "deg/s"
        return "norm./s"

    def _analysis_mask(self) -> np.ndarray:
        if self.trial is None:
            return np.array([], dtype=bool)
        return analysis_window_mask(
            self._active_times(), self.analysis_t0, t_end=self.analysis_t1
        )

    def _update_pupil_signal_options(self) -> None:
        if self._pupil_available():
            assert self.trial is not None and self.trial.pupil is not None
            pupil = self.trial.pupil
            self.signal_combo.configure(
                state="readonly",
                values=[label for label, _ in self.SIGNAL_CHOICES],
            )
            self.pupil_file_label.config(
                text=(
                    f"Pupil ({pupil.eye} eye): "
                    f"{Path(pupil.source_position).name}, "
                    f"{Path(pupil.source_time).name}"
                )
            )
        else:
            self.signal_var.set(self._signal_label_map[self.SIGNAL_ELEVATION])
            self.signal_combo.configure(
                state="readonly",
                values=[self._signal_label_map[self.SIGNAL_ELEVATION]],
            )
            self.pupil_file_label.config(text="")

    def _on_signal_selected(self, _event=None) -> None:
        mode = self._signal_mode()
        if mode != self.SIGNAL_ELEVATION and not self._pupil_available():
            self.signal_var.set(self._signal_label_map[self.SIGNAL_ELEVATION])
            messagebox.showwarning(
                "Pupil data unavailable",
                "No SRanipal pupil files were found for this trial.",
            )
            return
        if self.pending_start_idx is not None or self.pending_fit is not None:
            self._clear_pending(redraw=False)
            self._set_status("Cleared pending segment after signal change.")
        self._refit_all_segments()
        self._update_signal_ylim()
        self._redraw()

    def _refit_all_segments(self) -> None:
        if self.trial is None:
            return
        try:
            vel = self._stimulus_velocity()
        except ValueError:
            return
        valid = self._valid_click_mask()
        if not np.any(valid):
            return
        times = self._active_times()
        values = self._active_values()

        def refit_list(segments: list[SegmentFit]) -> list[SegmentFit]:
            refitted: list[SegmentFit] = []
            for seg in segments:
                try:
                    refitted.append(
                        refit_segment_by_time(
                            times,
                            values,
                            seg.t_start,
                            seg.t_end,
                            vel,
                            seg.segment_id,
                            valid,
                        )
                    )
                except ValueError:
                    refitted.append(seg)
            return refitted

        self.segments = refit_list(self.segments)
        self.proposed_segments = refit_list(self.proposed_segments)

    def _build_controls(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Browse gaze file…", command=self._browse_gaze).grid(
            row=0, column=0, padx=4, pady=2, sticky=tk.W
        )
        ttk.Button(top, text="Browse time file…", command=self._browse_time).grid(
            row=0, column=1, padx=4, pady=2, sticky=tk.W
        )
        ttk.Button(top, text="Load trial", command=self._load_trial).grid(
            row=0, column=2, padx=4, pady=2, sticky=tk.W
        )
        ttk.Button(top, text="Browse OKR log…", command=self._browse_okr_log).grid(
            row=0, column=3, padx=4, pady=2, sticky=tk.W
        )
        top.columnconfigure(5, weight=1)
        ttk.Button(top, text="Help", command=self._show_help).grid(
            row=0, column=5, padx=4, pady=2, sticky=tk.E
        )

        ttk.Label(top, text="Trial ID:").grid(row=1, column=0, sticky=tk.E, padx=4)
        self.trial_id_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.trial_id_var, width=28).grid(
            row=1, column=1, columnspan=2, sticky=tk.W, padx=4
        )

        vel_frame = ttk.Frame(top)
        vel_frame.grid(row=1, column=3, columnspan=2, sticky=tk.W, padx=4)
        ttk.Label(vel_frame, text="Stimulus velocity (deg/s):").pack(side=tk.LEFT)
        self.stim_vel_var = tk.StringVar(value="31")
        self.stim_vel_entry = ttk.Entry(vel_frame, textvariable=self.stim_vel_var, width=8)
        self.stim_vel_entry.pack(side=tk.LEFT, padx=(4, 0))
        self.stim_vel_applied_var = tk.StringVar(value="Applied: 31 deg/s")
        ttk.Label(
            vel_frame,
            textvariable=self.stim_vel_applied_var,
            foreground="forestgreen",
        ).pack(side=tk.LEFT, padx=(6, 0))
        self.stim_vel_entry.bind("<Return>", self._apply_stimulus_velocity)
        self.stim_vel_entry.bind("<FocusOut>", self._apply_stimulus_velocity_on_blur)
        self.stim_vel_var.trace_add("write", self._on_stimulus_velocity_edited)

        self.gaze_file_label = ttk.Label(top, text="")
        self.gaze_file_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=4)
        self.time_file_label = ttk.Label(top, text="")
        self.time_file_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=4)
        self.okr_log_file_label = ttk.Label(top, text="")
        self.okr_log_file_label.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=4)

        signal_row = ttk.Frame(top)
        signal_row.grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=4, pady=(2, 0))
        ttk.Label(signal_row, text="Signal:").pack(side=tk.LEFT)
        self.signal_var = tk.StringVar(value=self.SIGNAL_CHOICES[0][0])
        self.signal_combo = ttk.Combobox(
            signal_row,
            textvariable=self.signal_var,
            values=[label for label, _ in self.SIGNAL_CHOICES],
            state="readonly",
            width=28,
        )
        self.signal_combo.pack(side=tk.LEFT, padx=(4, 0))
        self.signal_combo.bind("<<ComboboxSelected>>", self._on_signal_selected)
        self.pupil_file_label = ttk.Label(top, text="")
        self.pupil_file_label.grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=4)

        view_row = ttk.Frame(top)
        view_row.grid(row=2, column=3, columnspan=2, rowspan=2, sticky=tk.W, padx=4, pady=2)
        ttk.Label(view_row, text="Window:").pack(side=tk.LEFT, padx=(0, 4))
        self.view_var = tk.StringVar(value="2 s")
        self.view_combo = ttk.Combobox(
            view_row,
            textvariable=self.view_var,
            values=self.VIEW_PRESET_LABELS,
            state="readonly",
            width=10,
        )
        self.view_combo.pack(side=tk.LEFT, padx=2)
        self.view_combo.bind("<<ComboboxSelected>>", self._on_view_selected)
        ttk.Button(view_row, text="Reset", width=6, command=self._reset_view).pack(
            side=tk.LEFT, padx=2
        )

        action = ttk.Frame(self.root, padding=8)
        action.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(action, text="Accept segment (A)", command=self._accept_segment).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(action, text="Undo last (U)", command=self._undo_segment).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(action, text="Clear pending (Esc)", command=self._clear_pending).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(action, text="Export Excel…", command=self._export).pack(
            side=tk.RIGHT, padx=4
        )

        self.status_var = tk.StringVar()
        ttk.Label(action, textvariable=self.status_var, wraplength=700).pack(
            side=tk.LEFT, padx=12, fill=tk.X, expand=True
        )

        help_text = (
            "Click start then end of each upward slow phase (snaps to nearest data point). "
            "Hover highlights the sample that will be used (red ring). Scroll vertically to pan time. See Help top right."
        )
        ttk.Label(self.root, text=help_text, padding=(8, 0)).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_main_area(self) -> None:
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.plot_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(self.plot_frame, weight=3)

        seg_panel = ttk.LabelFrame(self.main_pane, text="Segments", padding=6)
        self.main_pane.add(seg_panel, weight=1)

        detect_panel = ttk.LabelFrame(seg_panel, text="Auto-detect", padding=4)
        detect_panel.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        dir_row = ttk.Frame(detect_panel)
        dir_row.pack(fill=tk.X, pady=1)
        dir_label = ttk.Label(dir_row, text="Direction:")
        dir_label.pack(side=tk.LEFT)
        self.detect_dir_var = tk.StringVar(value="Auto")
        dir_combo = ttk.Combobox(
            dir_row,
            textvariable=self.detect_dir_var,
            values=("Auto", "Up", "Down"),
            state="readonly",
            width=8,
        )
        dir_combo.pack(side=tk.LEFT, padx=(4, 0))
        _bind_tooltip(dir_row, DETECT_TOOLTIPS["direction"])

        vel_row = ttk.Frame(detect_panel)
        vel_row.pack(fill=tk.X, pady=1)
        saccade_label = ttk.Label(vel_row, text="Max saccade velocity (deg/s):")
        saccade_label.pack(side=tk.LEFT)
        self.detect_saccade_var = tk.StringVar(value="100")
        saccade_entry = ttk.Entry(vel_row, textvariable=self.detect_saccade_var, width=8)
        saccade_entry.pack(side=tk.LEFT, padx=(4, 0))
        _bind_tooltip(vel_row, DETECT_TOOLTIPS["saccade"])

        dur_row = ttk.Frame(detect_panel)
        dur_row.pack(fill=tk.X, pady=1)
        dur_label = ttk.Label(dur_row, text="Min duration (ms):")
        dur_label.pack(side=tk.LEFT)
        self.detect_min_dur_var = tk.StringVar(value="50")
        dur_entry = ttk.Entry(dur_row, textvariable=self.detect_min_dur_var, width=8)
        dur_entry.pack(side=tk.LEFT, padx=(4, 0))
        _bind_tooltip(dur_row, DETECT_TOOLTIPS["duration"])

        r2_row = ttk.Frame(detect_panel)
        r2_row.pack(fill=tk.X, pady=1)
        r2_label = ttk.Label(r2_row, text="Min R²:")
        r2_label.pack(side=tk.LEFT)
        self.detect_min_r2_var = tk.StringVar(value="0.75")
        r2_entry = ttk.Entry(r2_row, textvariable=self.detect_min_r2_var, width=8)
        r2_entry.pack(side=tk.LEFT, padx=(4, 0))
        _bind_tooltip(r2_row, DETECT_TOOLTIPS["r2"])

        win_row = ttk.Frame(detect_panel)
        win_row.pack(fill=tk.X, pady=1)
        win_label = ttk.Label(win_row, text="Window (ms):")
        win_label.pack(side=tk.LEFT)
        self.detect_window_var = tk.StringVar(value="100")
        win_entry = ttk.Entry(win_row, textvariable=self.detect_window_var, width=8)
        win_entry.pack(side=tk.LEFT, padx=(4, 0))
        _bind_tooltip(win_row, DETECT_TOOLTIPS["window"])

        gap_row = ttk.Frame(detect_panel)
        gap_row.pack(fill=tk.X, pady=1)
        gap_label = ttk.Label(gap_row, text="Merge gap (ms):")
        gap_label.pack(side=tk.LEFT)
        self.detect_merge_gap_var = tk.StringVar(value="40")
        gap_entry = ttk.Entry(gap_row, textvariable=self.detect_merge_gap_var, width=8)
        gap_entry.pack(side=tk.LEFT, padx=(4, 0))
        _bind_tooltip(gap_row, DETECT_TOOLTIPS["merge_gap"])

        self.detect_refine_var = tk.BooleanVar(value=True)
        refine_chk = ttk.Checkbutton(
            detect_panel,
            text="Refine boundaries (maximize R²)",
            variable=self.detect_refine_var,
        )
        refine_chk.pack(anchor=tk.W, pady=(2, 0))
        _bind_tooltip(refine_chk, DETECT_TOOLTIPS["refine"])

        self.detect_blocks_var = tk.BooleanVar(value=False)
        blocks_chk = ttk.Checkbutton(
            detect_panel,
            text="Only inside contrast blocks (OKR log)",
            variable=self.detect_blocks_var,
        )
        blocks_chk.pack(anchor=tk.W, pady=(2, 0))
        _bind_tooltip(blocks_chk, DETECT_TOOLTIPS["blocks"])

        detect_btns = ttk.Frame(detect_panel)
        detect_btns.pack(fill=tk.X, pady=(4, 0))
        propose_btn = ttk.Button(
            detect_btns, text="Propose segments", command=self._propose_segments
        )
        propose_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        clear_btn = ttk.Button(
            detect_btns, text="Clear proposed", command=self._clear_proposed
        )
        clear_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        _bind_tooltip(propose_btn, DETECT_TOOLTIPS["propose"])
        _bind_tooltip(clear_btn, DETECT_TOOLTIPS["clear"])

        tree_frame = ttk.Frame(seg_panel)
        tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        columns = ("stat", "id", "start", "end", "gain", "r2")
        self.seg_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=10, selectmode="browse"
        )
        self.seg_tree.heading("stat", text="")
        self.seg_tree.heading("id", text="#")
        self.seg_tree.heading("start", text="Start (s)")
        self.seg_tree.heading("end", text="End (s)")
        self.seg_tree.heading("gain", text="Gain")
        self.seg_tree.heading("r2", text="R²")
        self.seg_tree.column("stat", width=34, anchor=tk.CENTER)
        self.seg_tree.column("id", width=28, anchor=tk.CENTER)
        self.seg_tree.column("start", width=62, anchor=tk.E)
        self.seg_tree.column("end", width=62, anchor=tk.E)
        self.seg_tree.column("gain", width=48, anchor=tk.E)
        self.seg_tree.column("r2", width=40, anchor=tk.E)
        seg_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.seg_tree.yview)
        self.seg_tree.configure(yscrollcommand=seg_scroll.set)
        self.seg_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        seg_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.seg_tree.bind("<<TreeviewSelect>>", self._on_segment_select)
        self.seg_tree.bind("<Double-1>", self._on_segment_double_click)

        seg_actions = ttk.Frame(seg_panel)
        seg_actions.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        ttk.Button(
            seg_actions, text="Accept selected (A)", command=self._accept_segment
        ).pack(side=tk.TOP, fill=tk.X)
        ttk.Button(
            seg_actions, text="Delete selected (Del)", command=self._delete_selected_segment
        ).pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        nav_row = ttk.Frame(seg_actions)
        nav_row.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        ttk.Button(
            nav_row, text="Prev proposed (P)", command=lambda: self._jump_proposed(-1)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(
            nav_row, text="Next proposed (N)", command=lambda: self._jump_proposed(1)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        ttk.Label(
            seg_actions,
            text="Selected: hover segment edges to drag, or [ ] , . nudge (one data point)",
            wraplength=180,
        ).pack(side=tk.TOP, pady=(4, 0))

    def _build_plot(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Elevation (deg)")
        self.ax.set_title("Elevation — mark slow-phase start and end")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.hover_var = tk.StringVar(value="")
        ttk.Label(
            self.plot_frame,
            textvariable=self.hover_var,
            padding=(0, 4),
            foreground="gray",
            anchor=tk.W,
        ).pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _show_help(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Keyboard shortcuts")
        dlg.transient(self.root)
        dlg.resizable(True, True)
        dlg.minsize(560, 480)

        frame = ttk.Frame(dlg, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        text_frame = ttk.Frame(frame)
        text_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        text = tk.Text(
            text_frame,
            wrap=tk.NONE,
            width=72,
            height=26,
            font=("Courier", 12),
            relief=tk.FLAT,
            borderwidth=0,
            yscrollcommand=scroll.set,
        )
        scroll.config(command=text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text.insert("1.0", HELP_TEXT)
        text.config(state=tk.DISABLED)

        ttk.Button(frame, text="Close", command=dlg.destroy).pack(pady=(8, 0))
        dlg.bind("<Escape>", lambda _e: dlg.destroy())
        dlg.geometry(f"+{self.root.winfo_rootx() + 40}+{self.root.winfo_rooty() + 40}")

    def _bind_keys(self) -> None:
        self.root.bind_all("<KeyPress>", self._on_key, add="+")

    def _on_key(self, event) -> str | None:
        widget = event.widget
        if isinstance(widget, (tk.Entry, ttk.Entry, tk.Text)):
            return None

        keysym = event.keysym
        char = event.char

        if keysym in ("question",) or char == "?":
            self._show_help()
        elif keysym in ("a", "A"):
            self._accept_segment()
        elif keysym in ("u", "U"):
            self._undo_segment()
        elif keysym == "Escape":
            if self._boundary_drag is not None:
                self._boundary_drag = None
                self._set_boundary_hover(None)
                self._redraw()
                self._set_status("Cancelled boundary drag.")
            else:
                self._clear_pending()
        elif keysym == "Left":
            self._pan_view(-self._pan_step_sec())
        elif keysym == "Right":
            self._pan_view(self._pan_step_sec())
        elif keysym in ("Delete", "BackSpace"):
            self._delete_selected_segment()
        elif keysym in ("bracketleft",) or char == "[":
            self._nudge_segment_boundary("start", -1)
        elif keysym in ("bracketright",) or char == "]":
            self._nudge_segment_boundary("start", 1)
        elif keysym in ("comma",) or char == ",":
            self._nudge_segment_boundary("end", -1)
        elif keysym in ("period",) or char == ".":
            self._nudge_segment_boundary("end", 1)
        elif keysym in ("n", "N"):
            self._jump_proposed(1)
        elif keysym in ("p", "P"):
            self._jump_proposed(-1)
        return None

    def _set_boundary_hover(self, boundary: str | None) -> None:
        cursor = "sb_h_double_arrow" if boundary else ""
        self.canvas.get_tk_widget().config(cursor=cursor)
        if boundary == self._hover_boundary:
            return
        self._hover_boundary = boundary
        self._redraw()

    def _set_proposed_hover(self, key: tuple[str, int] | None) -> None:
        if key == self._hover_proposed_key:
            return
        self._hover_proposed_key = key
        self._redraw()

    def _proposed_at_time(self, t: float) -> SegmentFit | None:
        matches = [
            seg for seg in self.proposed_segments if seg.t_start <= t <= seg.t_end
        ]
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        return min(matches, key=lambda s: abs((s.t_start + s.t_end) / 2 - t))

    def _proposed_by_key(self, key: tuple[str, int]) -> SegmentFit | None:
        kind, seg_id = key
        if kind != "proposed":
            return None
        for seg in self.proposed_segments:
            if seg.segment_id == seg_id:
                return seg
        return None

    def _on_motion(self, event) -> None:
        if self._boundary_drag is not None:
            if event.inaxes != self.ax or self.trial is None or event.xdata is None:
                return
            try:
                mask = self._valid_click_mask()
                idx = snap_index(self._active_times(), float(event.xdata), mask)
            except ValueError:
                return
            if self._boundary_drag.get("preview_idx") == idx:
                return
            self._boundary_drag["preview_idx"] = idx
            self._set_hover(idx)
            self._redraw()
            return

        if event.inaxes != self.ax or self.trial is None:
            self._set_boundary_hover(None)
            self._set_proposed_hover(None)
            self._set_hover(None)
            return
        if event.xdata is None:
            self._set_boundary_hover(None)
            self._set_proposed_hover(None)
            self._set_hover(None)
            return

        boundary = self._hit_test_boundary(float(event.xdata))
        self._set_boundary_hover(boundary)

        if boundary is not None:
            self._set_proposed_hover(None)
            ref = self._selected_segment_ref()
            if ref is not None:
                _kind, seg = ref
                idx = seg.idx_start if boundary == "start" else seg.idx_end
                self._set_hover(idx)
            return

        try:
            mask = self._valid_click_mask()
            if not np.any(mask):
                self._set_proposed_hover(None)
                self._set_hover(None)
                return
            idx = snap_index(self._active_times(), float(event.xdata), mask)
            t = float(self._active_times()[idx])
            proposed = self._proposed_at_time(t)
            self._set_proposed_hover(
                ("proposed", proposed.segment_id) if proposed else None
            )
            self._set_hover(idx)
        except ValueError:
            self._set_proposed_hover(None)
            self._set_hover(None)

    def _hover_hint(self) -> str:
        if self.trial is not None:
            if self.pending_fit is not None:
                return "Pending manual segment ready — press A to accept."
            if self.selected_segment_key is not None:
                return (
                    "Hover a proposed segment and press A to accept. "
                    "Hover near a segment edge to drag, or use [ ] , . to nudge."
                )
            if self.proposed_segments:
                return (
                    "Hover a proposed segment and press A to accept, "
                    "or hover a data point to mark a manual segment."
                )
            return "Hover a data point to preview where a click will snap."
        return ""

    def _set_hover(self, idx: int | None) -> None:
        if idx is None:
            self._hover_idx = None
            self.hover_var.set(self._hover_hint())
            self._clear_hover_artists()
            return
        assert self.trial is not None
        t = float(self._active_times()[idx])
        value = float(self._active_values()[idx])
        value_label = (
            f"elevation = {value:.2f}°"
            if self._signal_mode() == self.SIGNAL_ELEVATION
            else f"{self._signal_plot_label().lower()} = {value:.4f}"
        )
        if self.pending_fit is not None:
            self.hover_var.set(
                f"t = {t:.3f} s, {value_label} — "
                f"pending manual segment: press A to accept"
            )
        else:
            proposed = self._proposed_at_time(t)
            if proposed is not None:
                label = self._segment_display_label("proposed", proposed)
                self.hover_var.set(
                    f"t = {t:.3f} s, {value_label} — "
                    f"proposed {label}: press A to accept"
                )
            else:
                self.hover_var.set(f"t = {t:.3f} s, {value_label}")
        if idx == self._hover_idx and self._hover_artists:
            return
        self._hover_idx = idx
        self._draw_hover_highlight(idx)

    def _clear_hover_artists(self) -> None:
        for artist in self._hover_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._hover_artists.clear()

    def _draw_hover_highlight(self, idx: int) -> None:
        if self.trial is None:
            return
        self._clear_hover_artists()
        t = float(self._active_times()[idx])
        value = float(self._active_values()[idx])
        (ring,) = self.ax.plot(
            t,
            value,
            "o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=1.2,
            zorder=10,
            clip_on=True,
        )
        self._hover_artists.append(ring)
        self.canvas.draw_idle()

    def _position_in_valid(self, idx: int, valid: np.ndarray) -> int:
        """Index of ``idx`` within the ``valid`` index array."""
        matches = np.where(valid == idx)[0]
        if len(matches):
            return int(matches[0])
        return int(np.clip(np.searchsorted(valid, idx), 0, len(valid) - 1))

    def _x_tolerance_sec(self, pixels: float = 10.0) -> float:
        bbox = self.ax.get_window_extent()
        if bbox.width <= 0:
            return 0.01
        x0, x1 = self.ax.get_xlim()
        return abs(x1 - x0) * pixels / bbox.width

    def _hit_test_boundary(self, xdata: float) -> str | None:
        ref = self._selected_segment_ref()
        if ref is None:
            return None
        _kind, seg = ref
        tol = self._x_tolerance_sec()
        d_start = abs(xdata - seg.t_start)
        d_end = abs(xdata - seg.t_end)
        if d_start > tol and d_end > tol:
            return None
        if d_start <= d_end:
            return "start"
        return "end"

    def _update_files_label(self) -> None:
        self.gaze_file_label.config(
            text=f"Gaze: {self.gaze_path.name}" if self.gaze_path else ""
        )
        self.time_file_label.config(
            text=f"Time: {self.time_path.name}" if self.time_path else ""
        )
        if self.okr_log_path and self.okr_log is not None:
            n_blocks = len(self.okr_log.block_markers)
            n_fix = len(self.okr_log.fixation_markers)
            self.okr_log_file_label.config(
                text=(
                    f"OKR log uploaded ({n_blocks} block start(s), "
                    f"{n_fix} fixation start(s))"
                )
            )
        elif self.okr_log_path:
            self.okr_log_file_label.config(text="OKR log selected (not loaded)")
        else:
            self.okr_log_file_label.config(text="")

    def _pan_step_sec(self) -> float:
        if self.view_xmin is None or self.view_xmax is None:
            return self.DEFAULT_VIEW_DURATION * self.SCROLL_PAN_FRACTION
        return (self.view_xmax - self.view_xmin) * self.SCROLL_PAN_FRACTION

    def _browse_gaze(self) -> None:
        path = filedialog.askopenfilename(
            title="Select gaze file",
            filetypes=[
                ("Gaze files", "*.txt *.csv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.gaze_path = Path(path)
            self._update_files_label()
            if not self.trial_id_var.get().strip():
                self.trial_id_var.set(self.gaze_path.parent.name)

    def _browse_time(self) -> None:
        path = filedialog.askopenfilename(
            title="Select time file",
            filetypes=[
                ("Time files", "*.txt *.csv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.time_path = Path(path)
            self._update_files_label()

    def _browse_okr_log(self) -> None:
        path = filedialog.askopenfilename(
            title="Select OKR log file (optional)",
            filetypes=[
                ("OKR log files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.okr_log_path = Path(path)
        if not self._parse_okr_log(show_error=True):
            self.okr_log_path = None
            self.okr_log = None
        self._update_files_label()
        if self.trial is not None:
            self._redraw()

    def _parse_okr_log(self, show_error: bool = False) -> bool:
        if not self.okr_log_path:
            self.okr_log = None
            return True
        try:
            self.okr_log = load_okr_log(self.okr_log_path)
            self.detect_dir_var.set("Auto")
            self.detect_blocks_var.set(True)
            if self.trial is not None and self.gaze_path is not None:
                viewing_eye = infer_viewing_eye(
                    eye_patch=self.okr_log.stimulus_eye_patch,
                    trial_id=self.trial.trial_id,
                )
                attach_sranipal_pupil(
                    self.trial, self.gaze_path.parent, viewing_eye=viewing_eye
                )
                self._update_pupil_signal_options()
            return True
        except Exception as exc:
            self.okr_log = None
            if show_error:
                messagebox.showerror("OKR log failed", str(exc))
            return False

    def _stimulus_velocity(self) -> float:
        try:
            return float(self.stim_vel_var.get().strip())
        except ValueError as exc:
            raise ValueError("Stimulus velocity must be a number.") from exc

    def _on_stimulus_velocity_edited(self, *_args) -> None:
        raw = self.stim_vel_var.get().strip()
        if not raw:
            self.stim_vel_applied_var.set("Enter a value, then press Enter")
            return
        try:
            vel = float(raw)
        except ValueError:
            self.stim_vel_applied_var.set("Invalid — enter a number")
            return
        if abs(vel - self._applied_stimulus_velocity) < 1e-9:
            self.stim_vel_applied_var.set(f"Applied: {vel:g} deg/s")
        else:
            self.stim_vel_applied_var.set(f"Press Enter to apply ({vel:g} deg/s)")

    def _apply_stimulus_velocity_on_blur(self, _event=None) -> str | None:
        try:
            vel = self._stimulus_velocity()
        except ValueError:
            return None
        if abs(vel - self._applied_stimulus_velocity) < 1e-9:
            return None
        self._apply_stimulus_velocity(_event)
        return None

    def _recalculate_gains_for_velocity(self, vel: float) -> None:
        self.segments = [
            replace(seg, gain=seg.slope_deg_s / vel, stimulus_velocity=vel)
            for seg in self.segments
        ]
        self.proposed_segments = [
            replace(seg, gain=seg.slope_deg_s / vel, stimulus_velocity=vel)
            for seg in self.proposed_segments
        ]
        if (
            self.pending_start_idx is not None
            and self.pending_end_idx is not None
            and self.trial is not None
        ):
            try:
                self.pending_fit = fit_segment(
                    self._active_times(),
                    self._active_values(),
                    self.pending_start_idx,
                    self.pending_end_idx,
                    vel,
                    segment_id=len(self.segments) + 1,
                )
            except ValueError:
                pass

    def _has_unsaved_work(self) -> bool:
        return bool(self.segments) or bool(self.proposed_segments) or self.pending_fit is not None

    def _confirm_discard_work(self, action: str) -> bool:
        if not self._has_unsaved_work():
            return True
        n = len(self.segments)
        p = len(self.proposed_segments)
        extra = f" and a pending segment" if self.pending_fit else ""
        proposed_note = f", {p} proposed" if p else ""
        return messagebox.askyesno(
            "Discard annotations?",
            f"You have {n} accepted segment(s){proposed_note}{extra}.\n\n{action}?",
            icon="warning",
        )

    def _autosave_file(self) -> Path | None:
        if not self.gaze_path:
            return None
        trial_id = self.trial_id_var.get().strip() or self.gaze_path.parent.name
        return autosave_path(self.gaze_path.parent, trial_id)

    def _write_autosave(self) -> None:
        if not self.gaze_path or not self.time_path or not self.trial:
            return
        path = self._autosave_file()
        if path is None:
            return
        try:
            vel = self._stimulus_velocity()
        except ValueError:
            vel = 31.0
        save_autosave(
            path,
            trial_id=self.trial.trial_id,
            gaze_source=str(self.gaze_path.resolve()),
            time_source=str(self.time_path.resolve()),
            stimulus_velocity=vel,
            segments=self.segments,
            software_version=__version__,
            signal_mode=self._signal_mode(),
        )

    def _try_restore_autosave(self, trial_id: str) -> None:
        if not self.gaze_path or not self.time_path:
            return
        path = autosave_path(self.gaze_path.parent, trial_id)
        data = load_autosave(path)
        if data is None:
            return
        if not autosave_matches_trial(
            data,
            str(self.gaze_path.resolve()),
            str(self.time_path.resolve()),
        ):
            return
        restored = segments_from_autosave(data)
        if not restored:
            return
        n = len(restored)
        if not messagebox.askyesno(
            "Restore autosave",
            f"Found autosave with {n} segment(s) for this trial.\nRestore?",
        ):
            return
        self.segments = restored
        vel = data.get("stimulus_velocity")
        if vel is not None:
            self.stim_vel_var.set(str(vel))
            self._applied_stimulus_velocity = float(vel)
            self.stim_vel_applied_var.set(f"Applied: {float(vel):g} deg/s")
        signal_mode = data.get("signal_mode")
        if (
            isinstance(signal_mode, str)
            and signal_mode in self._signal_label_map
            and (
                signal_mode == self.SIGNAL_ELEVATION
                or self._pupil_available()
            )
        ):
            self.signal_var.set(self._signal_label_map[signal_mode])
            self._refit_all_segments()
            self._update_signal_ylim()
        self.selected_segment_key = None
        self._clear_pending(redraw=False)
        self._refresh_segment_list()
        self._redraw()
        med = trial_summary_median_gain(self.segments)
        self._set_status(
            f"Restored {n} segment(s) from autosave. Trial median gain={med:.3f}"
        )

    def _apply_stimulus_velocity(self, _event=None) -> None:
        try:
            vel = self._stimulus_velocity()
        except ValueError as exc:
            messagebox.showwarning("Invalid velocity", str(exc))
            self.stim_vel_applied_var.set("Invalid — enter a number")
            return

        if vel == 0:
            messagebox.showwarning(
                "Invalid velocity", "Stimulus velocity cannot be zero."
            )
            self.stim_vel_applied_var.set("Invalid — cannot be zero")
            return

        n_accepted = len(self.segments)
        n_proposed = len(self.proposed_segments)
        had_segments = n_accepted or n_proposed or self.pending_fit is not None

        self._applied_stimulus_velocity = vel
        self.stim_vel_applied_var.set(f"Applied: {vel:g} deg/s")

        if had_segments:
            self._recalculate_gains_for_velocity(vel)
            self._refresh_segment_list()
            self._redraw()
            self._write_autosave()

        status = f"Stimulus velocity applied: {vel:g} deg/s."
        if had_segments:
            parts: list[str] = []
            if n_accepted:
                parts.append(f"{n_accepted} accepted")
            if n_proposed:
                parts.append(f"{n_proposed} proposed")
            status += f" Recalculated gain for {', '.join(parts)} segment(s)."
            med = trial_summary_median_gain(self.segments)
            if n_accepted:
                status += f" Trial median gain={med:.3f}."
        else:
            status += " Will be used for new segments and export."
        self._set_status(status)

    def _load_trial(self) -> None:
        if not self.gaze_path or not self.time_path:
            messagebox.showwarning("Missing files", "Select both gaze and time files.")
            return
        if not self._confirm_discard_work("Load this trial anyway"):
            return
        try:
            trial_id = self.trial_id_var.get().strip() or self.gaze_path.parent.name
            self.trial = load_ush2a_trial(
                self.gaze_path, self.time_path, trial_id=trial_id
            )
            viewing_eye = infer_viewing_eye(
                eye_patch=(
                    self.okr_log.stimulus_eye_patch if self.okr_log else None
                ),
                trial_id=trial_id,
            )
            pupil = attach_sranipal_pupil(
                self.trial, self.gaze_path.parent, viewing_eye=viewing_eye
            )
            t0 = float(self.trial.times[0])
            t1 = float(self.trial.times[-1])
            self.analysis_t0 = t0
            self.analysis_t1 = t1
            self.window_mask = analysis_window_mask(self.trial.times, t0, t_end=t1)
            self._update_pupil_signal_options()
            self._update_signal_ylim()
            self.segments.clear()
            self.proposed_segments.clear()
            self.selected_segment_key = None
            self._clear_pending(redraw=False)
            self.view_var.set("2 s")
            self._reset_view()
            self._try_restore_autosave(trial_id)
            self._refresh_segment_list()
            if self.okr_log_path and not self._parse_okr_log(show_error=True):
                self.okr_log_path = None
            if self.okr_log is not None:
                self.detect_dir_var.set("Auto")
                self.detect_blocks_var.set(True)
            self._update_files_label()
            self._redraw()
            duration = t1 - t0
            status = (
                f"Loaded {trial_id}: {len(self.trial.times)} gaze samples, "
                f"analysis {t0:.2f}–{t1:.2f} s ({duration:.1f} s)"
            )
            if pupil is not None:
                status += f"; pupil ({pupil.eye}): {len(pupil.times)} samples"
            if self.okr_log is not None:
                status += (
                    f"; OKR log: {len(self.okr_log.block_markers)} blocks, "
                    f"{len(self.okr_log.fixation_markers)} fixations"
                )
            self._set_status(status)
            self.hover_var.set(self._hover_hint())
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def _preset_duration_sec(self, label: str | None = None) -> float:
        label = label if label is not None else self.view_var.get()
        return self.VIEW_PRESET_SECONDS.get(label, self.DEFAULT_VIEW_DURATION)

    def _on_view_selected(self, _event=None) -> None:
        choice = self.view_var.get()
        if choice == "Full trial":
            self._view_full()
        else:
            self._set_view_duration(self._preset_duration_sec(choice))

    def _reset_view(self) -> None:
        duration = self._preset_duration_sec()
        self.view_xmin = self.analysis_t0
        self.view_xmax = min(self.analysis_t0 + duration, self.analysis_t1)
        if self.trial is not None:
            self._redraw()

    def _set_view_duration(self, duration_sec: float) -> None:
        if self.trial is None:
            return
        center = None
        if self.view_xmin is not None and self.view_xmax is not None:
            center = (self.view_xmin + self.view_xmax) / 2
        else:
            center = self.analysis_t0 + duration_sec / 2
        half = duration_sec / 2
        self.view_xmin = max(self.analysis_t0, center - half)
        self.view_xmax = min(self.analysis_t1, center + half)
        if self.view_xmax - self.view_xmin < duration_sec:
            if self.view_xmin == self.analysis_t0:
                self.view_xmax = min(self.analysis_t0 + duration_sec, self.analysis_t1)
            else:
                self.view_xmin = max(self.analysis_t1 - duration_sec, self.analysis_t0)
        self._clamp_view()
        self._redraw()

    def _view_full(self) -> None:
        if self.trial is None:
            return
        self.view_xmin = self.analysis_t0
        self.view_xmax = self.analysis_t1
        self._redraw()

    def _clamp_view(self) -> None:
        if self.view_xmin is None or self.view_xmax is None:
            self._reset_view()
            return
        width = self.view_xmax - self.view_xmin
        width = float(
            np.clip(width, self.MIN_VIEW_DURATION, self.analysis_duration)
        )
        if self.view_xmin < self.analysis_t0:
            self.view_xmin = self.analysis_t0
            self.view_xmax = self.view_xmin + width
        if self.view_xmax > self.analysis_t1:
            self.view_xmax = self.analysis_t1
            self.view_xmin = self.view_xmax - width
        if self.view_xmin < self.analysis_t0:
            self.view_xmin = self.analysis_t0

    def _pan_view(self, delta_sec: float) -> None:
        if self.trial is None or self.view_xmin is None or self.view_xmax is None:
            return
        width = self.view_xmax - self.view_xmin
        self.view_xmin += delta_sec
        self.view_xmax += delta_sec
        if self.view_xmin < self.analysis_t0:
            self.view_xmin = self.analysis_t0
            self.view_xmax = self.view_xmin + width
        if self.view_xmax > self.analysis_t1:
            self.view_xmax = self.analysis_t1
            self.view_xmin = self.view_xmax - width
        self._redraw()

    def _is_vertical_scroll(self, event) -> bool:
        """Ignore horizontal trackpad / mouse-wheel gestures."""
        if getattr(event, "key", None) == "shift":
            return False
        gui = getattr(event, "guiEvent", None)
        if gui is None:
            return True
        num = getattr(gui, "num", None)
        if num in (6, 7):
            return False
        delta_x = getattr(gui, "deltaX", None)
        delta_y = getattr(gui, "deltaY", None)
        if delta_x is not None and delta_y is not None:
            ax = abs(int(delta_x))
            ay = abs(int(delta_y))
            if ax > ay:
                return False
        state = getattr(gui, "state", 0) or 0
        if state & 0x0001:
            return False
        return True

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax or self.trial is None:
            return
        if self.view_xmin is None or self.view_xmax is None:
            return
        if not self._is_vertical_scroll(event):
            return

        scroll_forward = event.step > 0 if getattr(event, "step", 0) else event.button == "up"
        delta = self._pan_step_sec()
        if not scroll_forward:
            delta = -delta
        self._pan_view(delta)

    def _valid_click_mask(self) -> np.ndarray:
        if self.trial is None:
            return np.array([], dtype=bool)
        return self._analysis_mask() & ~np.isnan(self._active_values())

    def _update_signal_ylim(self) -> None:
        """Fix y-axis to full-trial signal range so panning time does not rescale."""
        if self.trial is None:
            self.signal_ylim = None
            return
        valid = self._valid_click_mask()
        if not np.any(valid):
            self.signal_ylim = None
            return
        values = self._active_values()
        pad = 0.5 if self._signal_mode() == self.SIGNAL_ELEVATION else 0.002
        ymin = float(np.nanmin(values[valid])) - pad
        ymax = float(np.nanmax(values[valid])) + pad
        if ymin < ymax:
            self.signal_ylim = (ymin, ymax)
        else:
            self.signal_ylim = None

    def _valid_indices(self) -> np.ndarray:
        return np.where(self._valid_click_mask())[0]

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or self.trial is None:
            return
        if event.button != 1:
            return
        if event.xdata is None:
            return

        boundary = self._hit_test_boundary(float(event.xdata))
        if boundary is not None:
            ref = self._selected_segment_ref()
            if ref is None:
                return
            kind, seg = ref
            try:
                mask = self._valid_click_mask()
                idx = snap_index(self._active_times(), float(event.xdata), mask)
            except ValueError:
                return
            self._boundary_drag = {
                "kind": kind,
                "segment_id": seg.segment_id,
                "which": boundary,
                "preview_idx": idx,
            }
            self._set_hover(idx)
            self._redraw()
            return

        try:
            mask = self._valid_click_mask()
            idx = snap_index(self._active_times(), float(event.xdata), mask)
        except ValueError:
            return

        if self.pending_start_idx is None:
            self.pending_start_idx = idx
            self.pending_end_idx = None
            self.pending_fit = None
            t_start = self._active_times()[idx]
            self._set_status(
                f"Start marked at {t_start:.3f} s — click end of slow phase"
            )
        else:
            self.pending_end_idx = idx
            try:
                self.pending_fit = fit_segment(
                    self._active_times(),
                    self._active_values(),
                    self.pending_start_idx,
                    self.pending_end_idx,
                    self._stimulus_velocity(),
                    segment_id=len(self.segments) + 1,
                )
            except ValueError as exc:
                messagebox.showwarning("Segment too short", str(exc))
                self._clear_pending()
                return
            self._set_status(
                f"Pending segment: gain={self.pending_fit.gain:.3f}, "
                f"R²={self.pending_fit.r2:.3f} — press A to accept"
            )

        self._redraw()

    def _on_release(self, event) -> None:
        if self._boundary_drag is None:
            return
        drag = self._boundary_drag
        self._boundary_drag = None
        preview_idx = drag.get("preview_idx")
        if preview_idx is None:
            self._redraw()
            return
        kind = drag["kind"]
        segment_id = drag["segment_id"]
        which = drag["which"]
        self.selected_segment_key = (kind, segment_id)
        self._set_segment_boundary_idx(which, int(preview_idx))

    def _set_segment_boundary_idx(self, which: str, new_idx: int) -> None:
        ref = self._selected_segment_ref()
        if ref is None or self.trial is None:
            return
        kind, seg = ref
        valid = self._valid_indices()
        if len(valid) < 2:
            return

        start_pos = self._position_in_valid(seg.idx_start, valid)
        end_pos = self._position_in_valid(seg.idx_end, valid)
        new_pos = self._position_in_valid(new_idx, valid)

        if which == "start":
            if new_pos < 0 or new_pos >= end_pos:
                self._set_status("Start must stay before end.")
                self._redraw()
                return
            new_start = int(valid[new_pos])
            new_end = seg.idx_end
        else:
            if new_pos <= start_pos or new_pos >= len(valid):
                self._set_status("End must stay after start.")
                self._redraw()
                return
            new_end = int(valid[new_pos])
            new_start = seg.idx_start

        try:
            updated = fit_segment(
                self._active_times(),
                self._active_values(),
                new_start,
                new_end,
                self._stimulus_velocity(),
                segment_id=seg.segment_id,
            )
        except ValueError as exc:
            self._set_status(str(exc))
            self._redraw()
            return

        self._replace_segment(kind, updated)
        self._refresh_segment_list()
        self._redraw()
        if kind == "accepted":
            self._write_autosave()
        boundary = "start" if which == "start" else "end"
        t = updated.t_start if which == "start" else updated.t_end
        label = self._segment_display_label(kind, updated)
        self._set_status(
            f"{label} {boundary} → {t:.3f} s "
            f"(gain={updated.gain:.3f}, R²={updated.r2:.3f})"
        )

    def _segment_display_map(self) -> dict[tuple[str, int], int]:
        return {
            (kind, seg.segment_id): i
            for i, (kind, seg) in enumerate(self._segments_sorted_by_time(), start=1)
        }

    def _segment_display_label(self, kind: str, seg: SegmentFit) -> str:
        n = self._segment_display_map().get((kind, seg.segment_id), seg.segment_id)
        return f"#{n}" if kind == "accepted" else f"?{n}"

    def _sort_accepted_segments(self) -> None:
        self.segments.sort(key=lambda s: s.t_start)

    def _renumber_segments(self) -> None:
        for i, seg in enumerate(self.segments, start=1):
            if seg.segment_id != i:
                self.segments[i - 1] = replace(seg, segment_id=i)

    def _renumber_proposed_segments(self) -> None:
        for i, seg in enumerate(self.proposed_segments, start=1):
            if seg.segment_id != i:
                self.proposed_segments[i - 1] = replace(seg, segment_id=i)

    def _segment_tree_iid(self, kind: str, segment_id: int) -> str:
        return f"{kind}:{segment_id}"

    def _parse_segment_tree_iid(self, iid: str) -> tuple[str, int]:
        kind, sid = iid.split(":", 1)
        return kind, int(sid)

    def _segments_sorted_by_time(self) -> list[tuple[str, SegmentFit]]:
        rows: list[tuple[str, SegmentFit]] = []
        rows.extend(("accepted", seg) for seg in self.segments)
        rows.extend(("proposed", seg) for seg in self.proposed_segments)
        rows.sort(key=lambda item: item[1].t_start)
        return rows

    def _selected_segment_ref(self) -> tuple[str, SegmentFit] | None:
        if self.selected_segment_key is None:
            return None
        kind, seg_id = self.selected_segment_key
        source = self.segments if kind == "accepted" else self.proposed_segments
        for seg in source:
            if seg.segment_id == seg_id:
                return kind, seg
        return None

    def _selected_segment(self) -> SegmentFit | None:
        ref = self._selected_segment_ref()
        return ref[1] if ref else None

    def _detect_params(self) -> DetectParams:
        raw_dir = self.detect_dir_var.get().strip().lower()
        if raw_dir == "auto":
            direction = "auto"
        elif raw_dir == "up":
            direction = "up"
        elif raw_dir == "down":
            direction = "down"
        else:
            raise ValueError("Direction must be Auto, Up, or Down.")
        try:
            max_saccade = float(self.detect_saccade_var.get().strip())
            min_duration_ms = float(self.detect_min_dur_var.get().strip())
            min_r2 = float(self.detect_min_r2_var.get().strip())
            window_ms = float(self.detect_window_var.get().strip())
            merge_gap_ms = float(self.detect_merge_gap_var.get().strip())
        except ValueError as exc:
            raise ValueError("Auto-detect parameters must be numbers.") from exc
        if max_saccade <= 0:
            raise ValueError("Max saccade velocity must be positive.")
        if min_duration_ms <= 0:
            raise ValueError("Min duration must be positive.")
        if window_ms <= 0:
            raise ValueError("Window must be positive.")
        if merge_gap_ms < 0:
            raise ValueError("Merge gap must be non-negative.")
        if not 0.0 <= min_r2 <= 1.0:
            raise ValueError("Min R² must be between 0 and 1.")
        restrict = bool(self.detect_blocks_var.get())
        if direction == "auto" and (not restrict or self.okr_log is None):
            raise ValueError(
                "Auto direction needs an OKR log and 'Only inside contrast blocks' enabled."
            )
        if restrict and self.okr_log is None:
            raise ValueError("Load an OKR log to restrict detection to contrast blocks.")
        return DetectParams(
            direction=direction,
            min_duration_sec=min_duration_ms / 1000.0,
            min_r2=min_r2,
            max_saccade_velocity_deg_s=max_saccade,
            restrict_to_blocks=restrict,
            merge_gap_sec=merge_gap_ms / 1000.0,
            window_sec=window_ms / 1000.0,
            refine_boundaries=bool(self.detect_refine_var.get()),
        )

    def _propose_segments(self) -> None:
        if self.trial is None:
            messagebox.showwarning("No trial", "Load a trial before auto-detecting.")
            return
        if self.proposed_segments and not messagebox.askyesno(
            "Replace proposed segments?",
            "Clear existing proposed segments and run auto-detect again?",
        ):
            return
        try:
            params = self._detect_params()
            vel = self._stimulus_velocity()
        except ValueError as exc:
            messagebox.showwarning("Invalid auto-detect settings", str(exc))
            return
        if vel == 0:
            messagebox.showwarning("Invalid velocity", "Stimulus velocity cannot be zero.")
            return

        proposed = detect_slow_phases(
            self._active_times(),
            self._active_values(),
            self._valid_click_mask(),
            vel,
            params,
            okr_log=self.okr_log,
            exclude=self.segments,
        )
        self.proposed_segments = proposed
        self._renumber_proposed_segments()
        self.selected_segment_key = None
        self._refresh_segment_list()
        self._redraw()
        scope = "contrast blocks" if params.restrict_to_blocks else "full trial"
        self._set_status(
            f"Proposed {len(proposed)} {params.direction} segment(s) "
            f"({scope}, sorted by time). Review and accept or delete."
        )

    def _clear_proposed(self) -> None:
        if not self.proposed_segments:
            self._set_status("No proposed segments to clear.")
            return
        self.proposed_segments.clear()
        if self.selected_segment_key and self.selected_segment_key[0] == "proposed":
            self.selected_segment_key = None
        self._refresh_segment_list()
        self._redraw()
        self._set_status("Cleared proposed segments.")

    def _proposed_sorted(self) -> list[SegmentFit]:
        return sorted(self.proposed_segments, key=lambda s: s.t_start)

    def _center_view_on_segment(self, seg: SegmentFit) -> None:
        self._center_view_on_time((seg.t_start + seg.t_end) / 2)

    def _center_view_on_time(self, center: float) -> None:
        if self.view_xmin is not None and self.view_xmax is not None:
            width = self.view_xmax - self.view_xmin
        else:
            width = self._preset_duration_sec()
        half = width / 2
        self.view_xmin = max(self.analysis_t0, center - half)
        self.view_xmax = min(self.analysis_t1, center + half)
        self._clamp_view()

    def _jump_proposed(self, step: int) -> None:
        proposed = self._proposed_sorted()
        if not proposed:
            self._set_status("No proposed segments to navigate.")
            return

        current_idx: int | None = None
        if self.selected_segment_key and self.selected_segment_key[0] == "proposed":
            for i, seg in enumerate(proposed):
                if seg.segment_id == self.selected_segment_key[1]:
                    current_idx = i
                    break

        if current_idx is None:
            new_idx = 0 if step > 0 else len(proposed) - 1
        else:
            new_idx = (current_idx + step) % len(proposed)

        seg = proposed[new_idx]
        self.selected_segment_key = ("proposed", seg.segment_id)
        self._center_view_on_segment(seg)
        self._refresh_segment_list()
        self._redraw()
        self._set_status(
            f"Proposed {self._segment_display_label('proposed', seg)} "
            f"({new_idx + 1}/{len(proposed)}): "
            f"{seg.t_start:.2f}–{seg.t_end:.2f} s, gain={seg.gain:.3f}, R²={seg.r2:.2f}"
        )

    def _on_segment_select(self, _event=None) -> None:
        selection = self.seg_tree.selection()
        prev_key = self.selected_segment_key
        if not selection:
            self.selected_segment_key = None
        else:
            self.selected_segment_key = self._parse_segment_tree_iid(selection[0])
        self._hover_boundary = None
        self.canvas.get_tk_widget().config(cursor="")
        if self.selected_segment_key != prev_key:
            seg = self._selected_segment()
            if seg is not None:
                self._center_view_on_segment(seg)
        self._redraw()

    def _on_segment_double_click(self, event) -> None:
        iid = self.seg_tree.identify_row(event.y)
        if not iid:
            return
        kind, seg_id = self._parse_segment_tree_iid(iid)
        self.selected_segment_key = (kind, seg_id)
        self.seg_tree.selection_set(iid)
        self.seg_tree.focus(iid)
        seg = self._selected_segment()
        if seg is None:
            return
        self._hover_boundary = None
        self.canvas.get_tk_widget().config(cursor="")
        self._center_view_on_segment(seg)
        self._redraw()

    def _refresh_segment_list(self) -> None:
        for item in self.seg_tree.get_children():
            self.seg_tree.delete(item)
        display_id = 1
        for kind, seg in self._segments_sorted_by_time():
            stat = "✓" if kind == "accepted" else "?"
            r2 = f"{seg.r2:.2f}" if seg.r2 == seg.r2 else "—"
            iid = self._segment_tree_iid(kind, seg.segment_id)
            self.seg_tree.insert(
                "",
                tk.END,
                iid=iid,
                values=(
                    stat,
                    display_id,
                    f"{seg.t_start:.2f}",
                    f"{seg.t_end:.2f}",
                    f"{seg.gain:.3f}",
                    r2,
                ),
            )
            display_id += 1
            if self.selected_segment_key == (kind, seg.segment_id):
                self.seg_tree.selection_set(iid)
                self.seg_tree.see(iid)

    def _replace_segment(self, kind: str, updated: SegmentFit) -> None:
        target = self.segments if kind == "accepted" else self.proposed_segments
        for i, existing in enumerate(target):
            if existing.segment_id == updated.segment_id:
                target[i] = updated
                return

    def _nudge_segment_boundary(self, which: str, direction: int) -> None:
        ref = self._selected_segment_ref()
        if ref is None or self.trial is None:
            return
        kind, seg = ref
        valid = self._valid_indices()
        if len(valid) < 2:
            return

        start_pos = self._position_in_valid(seg.idx_start, valid)
        end_pos = self._position_in_valid(seg.idx_end, valid)

        if which == "start":
            new_pos = start_pos + direction
            if new_pos < 0 or new_pos >= end_pos:
                self._set_status("Start must stay before end (one data point at a time).")
                return
            new_idx = int(valid[new_pos])
        else:
            new_pos = end_pos + direction
            if new_pos <= start_pos or new_pos >= len(valid):
                self._set_status("End must stay after start (one data point at a time).")
                return
            new_idx = int(valid[new_pos])

        self._set_segment_boundary_idx(which, new_idx)

    def _delete_selected_segment(self) -> None:
        ref = self._selected_segment_ref()
        if ref is None:
            self._set_status("Select a segment in the list to delete.")
            return
        kind, seg = ref
        if kind == "accepted":
            self.segments = [s for s in self.segments if s.segment_id != seg.segment_id]
            self._renumber_segments()
            self._write_autosave()
        else:
            self.proposed_segments = [
                s for s in self.proposed_segments if s.segment_id != seg.segment_id
            ]
        self.selected_segment_key = None
        self._refresh_segment_list()
        self._redraw()
        if kind == "accepted":
            med = trial_summary_median_gain(self.segments)
            n = len(self.segments)
            status = f"Deleted accepted segment. {n} remaining."
            if n:
                status += f" Trial median gain={med:.3f}"
        else:
            status = f"Deleted proposed segment. {len(self.proposed_segments)} proposed remaining."
        self._set_status(status)

    def _accept_proposed_segment(self, seg: SegmentFit) -> bool:
        match = next(
            (s for s in self.proposed_segments if s.segment_id == seg.segment_id),
            None,
        )
        if match is None:
            return False
        seg = match
        proposed_label = self._segment_display_label("proposed", seg)
        self.proposed_segments = [
            s for s in self.proposed_segments if s.segment_id != seg.segment_id
        ]
        accepted = replace(seg, segment_id=len(self.segments) + 1)
        self.segments.append(accepted)
        self._sort_accepted_segments()
        self._renumber_segments()
        accepted = next(
            s
            for s in self.segments
            if s.idx_start == seg.idx_start and s.idx_end == seg.idx_end
        )
        self.selected_segment_key = ("accepted", accepted.segment_id)
        self._hover_proposed_key = None
        self._refresh_segment_list()
        self._redraw()
        self._write_autosave()
        med = trial_summary_median_gain(self.segments)
        accepted_label = self._segment_display_label("accepted", accepted)
        self._set_status(
            f"Accepted {proposed_label} as {accepted_label}: "
            f"gain={accepted.gain:.3f}, R²={accepted.r2:.3f}, "
            f"trial median gain={med:.3f} (n={len(self.segments)})"
        )
        return True

    def _accept_selected_proposed(self) -> bool:
        ref = self._selected_segment_ref()
        if ref is None or ref[0] != "proposed":
            return False
        return self._accept_proposed_segment(ref[1])

    def _accept_segment(self) -> None:
        if self.pending_fit is not None:
            pending = self.pending_fit
            self.segments.append(pending)
            self._sort_accepted_segments()
            self._renumber_segments()
            accepted = next(
                s
                for s in self.segments
                if s.idx_start == pending.idx_start and s.idx_end == pending.idx_end
            )
            self._clear_pending(redraw=False)
            self.selected_segment_key = ("accepted", accepted.segment_id)
            self._refresh_segment_list()
            self._redraw()
            self._write_autosave()
            med = trial_summary_median_gain(self.segments)
            self._set_status(
                f"Accepted {self._segment_display_label('accepted', accepted)}: "
                f"gain={accepted.gain:.3f}, R²={accepted.r2:.3f}, "
                f"trial median gain={med:.3f} (n={len(self.segments)})"
            )
            return

        if self._hover_proposed_key is not None:
            seg = self._proposed_by_key(self._hover_proposed_key)
            if seg is not None and self._accept_proposed_segment(seg):
                return

        if self._accept_selected_proposed():
            return

        self._set_status(
            "Mark a segment (two clicks), hover a proposed segment, "
            "or select one in the list, then press A."
        )

    def _undo_segment(self) -> None:
        if self.segments:
            removed = self.segments.pop()
            if self.selected_segment_key == ("accepted", removed.segment_id):
                self.selected_segment_key = None
            self._renumber_segments()
            self._refresh_segment_list()
            self._redraw()
            self._write_autosave()
            med = trial_summary_median_gain(self.segments)
            self._set_status(
                f"Removed last segment. Trial median gain={med:.3f} (n={len(self.segments)})"
            )
        else:
            self._set_status("No segments to undo.")

    def _clear_pending(self, redraw: bool = True) -> None:
        self.pending_start_idx = None
        self.pending_end_idx = None
        self.pending_fit = None
        if redraw:
            self._redraw()

    def _redraw(self) -> None:
        self.ax.clear()
        self._click_markers.clear()
        self._hover_artists.clear()

        if self.trial is None:
            self.ax.set_title("Load a trial to begin")
            self.canvas.draw_idle()
            return

        times = self._active_times()
        values = self._active_values()
        t0 = self.analysis_t0
        t1 = self.analysis_t1

        if self.view_xmin is None or self.view_xmax is None:
            self._reset_view()
        assert self.view_xmin is not None and self.view_xmax is not None
        vx0, vx1 = self.view_xmin, self.view_xmax

        plot_mask = (
            self._analysis_mask()
            & ~np.isnan(values)
            & (times >= vx0)
            & (times <= vx1)
        )
        self.ax.plot(
            times[plot_mask],
            values[plot_mask],
            linestyle="none",
            marker=".",
            markersize=4,
            color="0.25",
            label=self._signal_plot_label(),
        )

        self.ax.axvspan(t0, t1, color="steelblue", alpha=0.04)
        self.ax.axvline(t0, color="steelblue", linestyle="--", alpha=0.35, linewidth=0.8)
        self.ax.axvline(t1, color="steelblue", linestyle="--", alpha=0.35, linewidth=0.8)

        self._draw_okr_log_markers(vx0, vx1)

        display_map = self._segment_display_map()
        label_trans = blended_transform_factory(self.ax.transData, self.ax.transAxes)

        for kind, seg in self._segments_sorted_by_time():
            if seg.t_end < vx0 or seg.t_start > vx1:
                continue
            selected = self.selected_segment_key == (kind, seg.segment_id)
            hovered = (
                kind == "proposed"
                and self._hover_proposed_key == (kind, seg.segment_id)
            )
            highlighted = selected or hovered
            if kind == "accepted":
                span_color = "limegreen" if highlighted else "green"
                span_alpha = 0.35 if highlighted else 0.2
                line_color = "forestgreen" if highlighted else "darkgreen"
                line_width = 2.5 if highlighted else 1.5
            else:
                span_color = "cornflowerblue" if highlighted else "steelblue"
                span_alpha = 0.3 if highlighted else 0.15
                line_color = "royalblue" if highlighted else "steelblue"
                line_width = 2.5 if highlighted else 1.5
            display_n = display_map.get((kind, seg.segment_id), seg.segment_id)
            label_text = f"{'#' if kind == 'accepted' else '?'}{display_n}"
            self.ax.axvspan(seg.t_start, seg.t_end, color=span_color, alpha=span_alpha)
            seg_times = np.linspace(seg.t_start, seg.t_end, 50)
            self.ax.plot(
                seg_times,
                seg.slope_deg_s * seg_times + seg.intercept_deg,
                color=line_color,
                linewidth=line_width,
                linestyle="-" if kind == "accepted" else "--",
                clip_on=True,
            )
            mid_t = (seg.t_start + seg.t_end) / 2
            if vx0 <= mid_t <= vx1:
                self.ax.text(
                    mid_t,
                    0.97,
                    label_text,
                    transform=label_trans,
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="top",
                    clip_on=True,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=line_color,
                        alpha=0.85,
                    ),
                )
            if selected:
                active_boundary = self._hover_boundary
                if self._boundary_drag:
                    drag_kind = self._boundary_drag.get("kind")
                    drag_id = self._boundary_drag.get("segment_id")
                    if drag_kind == kind and drag_id == seg.segment_id:
                        active_boundary = self._boundary_drag.get("which")

                edge_color = "gold" if kind == "accepted" else "deepskyblue"
                for boundary, t_bound in (
                    ("start", seg.t_start),
                    ("end", seg.t_end),
                ):
                    if self._boundary_drag:
                        drag_kind = self._boundary_drag.get("kind")
                        drag_id = self._boundary_drag.get("segment_id")
                        which = self._boundary_drag.get("which")
                        if (
                            drag_kind == kind
                            and drag_id == seg.segment_id
                            and which == boundary
                        ):
                            preview_idx = self._boundary_drag.get("preview_idx")
                            if preview_idx is not None:
                                t_bound = float(times[preview_idx])

                    if active_boundary != boundary:
                        continue
                    if t_bound < vx0 or t_bound > vx1:
                        continue
                    self.ax.axvline(
                        t_bound,
                        color=edge_color,
                        linestyle="-",
                        linewidth=3.0,
                        alpha=0.95,
                        zorder=8,
                    )

        if self.pending_start_idx is not None:
            t_start = times[self.pending_start_idx]
            self.ax.axvline(t_start, color="orange", linestyle=":", linewidth=1.2)
            (marker,) = self.ax.plot(
                t_start, values[self.pending_start_idx], "o", color="orange", markersize=8
            )
            self._click_markers.append(marker)
            if self.pending_end_idx is None:
                self.ax.axvspan(
                    t_start,
                    min(t_start + 0.15 * self.analysis_duration, t1),
                    color="orange",
                    alpha=0.08,
                )

        if self.pending_end_idx is not None:
            t_end = times[self.pending_end_idx]
            self.ax.axvline(t_end, color="darkorange", linestyle=":", linewidth=1.2)
            (marker,) = self.ax.plot(
                t_end, values[self.pending_end_idx], "o", color="darkorange", markersize=8
            )
            self._click_markers.append(marker)

        if self.pending_fit is not None:
            self.ax.axvspan(
                self.pending_fit.t_start, self.pending_fit.t_end, color="orange", alpha=0.15
            )
            seg_times = np.linspace(
                self.pending_fit.t_start, self.pending_fit.t_end, 50
            )
            (line,) = self.ax.plot(
                seg_times,
                self.pending_fit.slope_deg_s * seg_times + self.pending_fit.intercept_deg,
                color="orangered",
                linewidth=2.0,
                label="Pending fit",
            )
            self._preview_line = line
            upward = "yes" if self.pending_fit.direction_upward else "FLAG: not upward"
            slope_unit = self._signal_slope_unit()
            slope_fmt = (
                f"{self.pending_fit.slope_deg_s:.2f}"
                if self._signal_mode() == self.SIGNAL_ELEVATION
                else f"{self.pending_fit.slope_deg_s:.4f}"
            )
            self.ax.set_title(
                f"Pending: slope={slope_fmt} {slope_unit}, "
                f"gain={self.pending_fit.gain:.3f}, R²={self.pending_fit.r2:.3f} ({upward})"
            )
        elif self.pending_start_idx is not None and self.pending_end_idx is None:
            t_start = times[self.pending_start_idx]
            self.ax.set_title(
                f"Click end of slow phase (start at {t_start:.3f} s)"
            )
        else:
            med = trial_summary_median_gain(self.segments)
            n = len(self.segments)
            p = len(self.proposed_segments)
            duration = t1 - t0
            title = f"{self.trial.trial_id} — {n} accepted"
            if p:
                title += f", {p} proposed"
            title += f", {duration:.1f} s trial"
            if n:
                title += f", median gain={med:.3f}"
            self.ax.set_title(title)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(self._signal_y_label())
        self.ax.set_xlim(vx0, vx1)

        if self.signal_ylim is not None:
            ymin, ymax = self.signal_ylim
            if ymin < ymax:
                self.ax.set_ylim(ymin, ymax)

        if self._hover_idx is not None:
            self._draw_hover_highlight(self._hover_idx)
        else:
            self.canvas.draw_idle()
            self.hover_var.set(self._hover_hint())

    def _draw_okr_log_markers(self, vx0: float, vx1: float) -> None:
        if self.okr_log is None:
            return

        label_trans = blended_transform_factory(self.ax.transData, self.ax.transAxes)

        for marker in self.okr_log.fixation_markers:
            t = marker.start_time
            if t < vx0 or t > vx1:
                continue
            self.ax.axvline(
                t,
                color="#5c6b73",
                linestyle=(0, (1, 2)),
                linewidth=1.0,
                alpha=0.75,
                zorder=1,
            )
            label = "F"
            self.ax.text(
                t,
                0.97,
                label,
                transform=label_trans,
                color="#5c6b73",
                fontsize=7,
                ha="center",
                va="top",
                clip_on=True,
            )

        for marker in self.okr_log.block_markers:
            t = marker.start_time
            if t < vx0 or t > vx1:
                continue
            self.ax.axvline(
                t,
                color="#7b2cbf",
                linestyle=(0, (4, 2)),
                linewidth=1.1,
                alpha=0.85,
                zorder=1,
            )
            self.ax.text(
                t,
                0.88,
                marker.label,
                transform=label_trans,
                color="#7b2cbf",
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="top",
                rotation=90,
                clip_on=True,
            )

    def _export(self) -> None:
        if not self.segments:
            messagebox.showwarning("Nothing to export", "Accept at least one segment first.")
            return
        try:
            vel = self._stimulus_velocity()
        except ValueError as exc:
            messagebox.showwarning("Invalid velocity", str(exc))
            return
        if vel == 0:
            messagebox.showwarning(
                "Invalid velocity", "Stimulus velocity cannot be zero."
            )
            return
        self._apply_stimulus_velocity()
        trial_id = self.trial.trial_id if self.trial else self.trial_id_var.get() or "trial"
        default_name = f"{trial_id}_slowphase_okr.xlsx"
        path = filedialog.asksaveasfilename(
            title="Save Excel export",
            defaultextension=".xlsx",
            initialfile=default_name,
            filetypes=[("Excel workbook", "*.xlsx")],
        )
        if not path:
            return
        try:
            out = export_to_excel(
                self.segments,
                trial_id=trial_id,
                software_version=__version__,
                output_path=path,
                gaze_source=str(self.gaze_path) if self.gaze_path else "",
                time_source=str(self.time_path) if self.time_path else "",
            )
            messagebox.showinfo("Exported", f"Saved:\n{out}")
            self._set_status(f"Exported {len(self.segments)} segment(s) to {out}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def _on_close(self) -> None:
        if self.segments:
            self._write_autosave()
        self.root.destroy()

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    if "aqua" in style.theme_names():
        style.theme_use("aqua")
    AnnotatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
