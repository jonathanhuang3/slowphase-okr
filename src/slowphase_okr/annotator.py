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

from slowphase_okr import __version__
from slowphase_okr.autosave import (
    autosave_matches_trial,
    autosave_path,
    load_autosave,
    save_autosave,
    segments_from_autosave,
)
from slowphase_okr.export import export_to_excel
from slowphase_okr.fit import SegmentFit, fit_segment, snap_index, trial_summary_median_gain
from slowphase_okr.gaze import GazeTrial, analysis_window_mask, load_ush2a_trial


HELP_TEXT = """Keyboard shortcuts
────────────────────────────────────────
Marking
  Click × 2        Mark slow-phase start, then end (snaps to nearest sample)
  A                Accept pending segment
  Esc              Clear pending segment (start over)

Navigation
  Scroll wheel     Zoom time axis (cursor-centered)
  ← / →            Pan view by 1 s
  View buttons     5 s, 10 s, Full trial, Reset (first 5 s)

Segments (select one in the list first)
  Del              Delete selected segment
  U                Undo last accepted segment
  [  ]             Nudge start earlier / later (one sample)
  ,  .             Nudge end earlier / later (one sample)

Other
  Enter            Apply stimulus velocity (in velocity field)
  ?                Show this help

Notes
  • Analysis window spans the full trial (first to last timestamp).
  • R² is logged and exported but segments are not auto-rejected.
  • Annotations autosave to JSON in the trial folder; restore on reload.
"""


class AnnotatorApp:
    DEFAULT_VIEW_DURATION = 5.0
    MIN_VIEW_DURATION = 0.5
    PAN_STEP_SEC = 1.0

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"slowphase-okr v{__version__}")
        self.root.minsize(1100, 680)

        self.gaze_path: Path | None = None
        self.time_path: Path | None = None
        self.trial: GazeTrial | None = None
        self.window_mask: np.ndarray | None = None  # type: ignore[name-defined]

        self.segments: list[SegmentFit] = []
        self.pending_start_idx: int | None = None
        self.pending_end_idx: int | None = None
        self.pending_fit: SegmentFit | None = None
        self.selected_segment_id: int | None = None

        self._preview_line = None
        self._click_markers: list = []

        self.analysis_t0: float = 0.0
        self.analysis_t1: float = 0.0
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
        ttk.Label(vel_frame, text="(Enter to apply)").pack(side=tk.LEFT, padx=(4, 0))
        self.stim_vel_entry.bind("<Return>", self._apply_stimulus_velocity)
        self.stim_vel_entry.bind("<FocusOut>", self._apply_stimulus_velocity)

        self.gaze_label = ttk.Label(top, text="Gaze: (none)", wraplength=420)
        self.gaze_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=4)
        self.time_label = ttk.Label(top, text="Time: (none)", wraplength=420)
        self.time_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=4)

        view_row = ttk.Frame(top)
        view_row.grid(row=2, column=3, columnspan=2, rowspan=2, sticky=tk.W, padx=4, pady=2)
        ttk.Label(view_row, text="View:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(view_row, text="5 s", width=5, command=lambda: self._set_view_duration(5.0)).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(view_row, text="10 s", width=5, command=lambda: self._set_view_duration(10.0)).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(view_row, text="Full", width=5, command=self._view_full).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(view_row, text="Reset", width=6, command=self._reset_view).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(view_row, text="?", width=3, command=self._show_help).pack(
            side=tk.LEFT, padx=(8, 2)
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
            "Click start then end of each upward slow phase (snaps to nearest sample). "
            "Scroll to zoom; Left/Right arrows to pan. Press ? for shortcuts."
        )
        ttk.Label(self.root, text=help_text, padding=(8, 0)).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_main_area(self) -> None:
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.plot_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(self.plot_frame, weight=3)

        seg_panel = ttk.LabelFrame(self.main_pane, text="Segments", padding=6)
        self.main_pane.add(seg_panel, weight=1)

        tree_frame = ttk.Frame(seg_panel)
        tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        columns = ("id", "start", "end", "gain", "r2", "up")
        self.seg_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=14, selectmode="browse"
        )
        self.seg_tree.heading("id", text="#")
        self.seg_tree.heading("start", text="Start (s)")
        self.seg_tree.heading("end", text="End (s)")
        self.seg_tree.heading("gain", text="Gain")
        self.seg_tree.heading("r2", text="R²")
        self.seg_tree.heading("up", text="Up")
        self.seg_tree.column("id", width=28, anchor=tk.CENTER)
        self.seg_tree.column("start", width=62, anchor=tk.E)
        self.seg_tree.column("end", width=62, anchor=tk.E)
        self.seg_tree.column("gain", width=52, anchor=tk.E)
        self.seg_tree.column("r2", width=44, anchor=tk.E)
        self.seg_tree.column("up", width=32, anchor=tk.CENTER)
        seg_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.seg_tree.yview)
        self.seg_tree.configure(yscrollcommand=seg_scroll.set)
        self.seg_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        seg_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.seg_tree.bind("<<TreeviewSelect>>", self._on_segment_select)

        seg_actions = ttk.Frame(seg_panel)
        seg_actions.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        ttk.Button(
            seg_actions, text="Delete selected (Del)", command=self._delete_selected_segment
        ).pack(side=tk.TOP, fill=tk.X)
        ttk.Label(
            seg_actions,
            text="Selected: [ ] nudge start, , . nudge end",
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
        ttk.Label(self.plot_frame, textvariable=self.hover_var, padding=(0, 4)).pack(
            side=tk.BOTTOM, fill=tk.X
        )

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _show_help(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Keyboard shortcuts")
        dlg.transient(self.root)
        dlg.resizable(False, False)

        frame = ttk.Frame(dlg, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        text = tk.Text(
            frame,
            wrap=tk.NONE,
            width=52,
            height=24,
            font=("Courier", 11),
            relief=tk.FLAT,
            borderwidth=0,
        )
        text.insert("1.0", HELP_TEXT)
        text.config(state=tk.DISABLED)
        text.pack(side=tk.TOP)

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
            self._clear_pending()
        elif keysym == "Left":
            self._pan_view(-self.PAN_STEP_SEC)
        elif keysym == "Right":
            self._pan_view(self.PAN_STEP_SEC)
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
        return None

    def _on_motion(self, event) -> None:
        if event.inaxes != self.ax or self.trial is None:
            self.hover_var.set("")
            return
        if event.xdata is None:
            self.hover_var.set("")
            return
        try:
            mask = self._valid_click_mask()
            if not np.any(mask):
                self.hover_var.set("")
                return
            idx = snap_index(self.trial.times, float(event.xdata), mask)
            t = float(self.trial.times[idx])
            elev = float(self.trial.elevation_deg[idx])
            self.hover_var.set(f"t = {t:.3f} s,  elevation = {elev:.2f}°")
        except ValueError:
            self.hover_var.set("")

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
            self.gaze_label.config(text=f"Gaze: {self.gaze_path}")
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
            self.time_label.config(text=f"Time: {self.time_path}")

    def _stimulus_velocity(self) -> float:
        try:
            return float(self.stim_vel_var.get().strip())
        except ValueError as exc:
            raise ValueError("Stimulus velocity must be a number.") from exc

    def _has_unsaved_work(self) -> bool:
        return bool(self.segments) or self.pending_fit is not None

    def _confirm_discard_work(self, action: str) -> bool:
        if not self._has_unsaved_work():
            return True
        n = len(self.segments)
        extra = f" and a pending segment" if self.pending_fit else ""
        return messagebox.askyesno(
            "Discard annotations?",
            f"You have {n} accepted segment(s){extra}.\n\n{action}?",
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
        self.selected_segment_id = None
        self._clear_pending(redraw=False)
        self._refresh_segment_list()
        self._redraw()
        med = trial_summary_median_gain(self.segments)
        self._set_status(
            f"Restored {n} segment(s) from autosave. Trial median gain={med:.3f}"
        )

    def _apply_stimulus_velocity(self, _event=None) -> None:
        if not self.segments and self.pending_fit is None:
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

        self.segments = [
            replace(seg, gain=seg.slope_deg_s / vel, stimulus_velocity=vel)
            for seg in self.segments
        ]

        if (
            self.pending_start_idx is not None
            and self.pending_end_idx is not None
            and self.trial is not None
        ):
            try:
                self.pending_fit = fit_segment(
                    self.trial.times,
                    self.trial.elevation_deg,
                    self.pending_start_idx,
                    self.pending_end_idx,
                    vel,
                    segment_id=len(self.segments) + 1,
                )
            except ValueError:
                pass

        self._refresh_segment_list()
        self._redraw()
        self._write_autosave()
        med = trial_summary_median_gain(self.segments)
        n = len(self.segments)
        status = f"Stimulus velocity set to {vel:.2f} deg/s."
        if n:
            status += f" Trial median gain={med:.3f} (n={n})"
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
            t0 = float(self.trial.times[0])
            t1 = float(self.trial.times[-1])
            self.analysis_t0 = t0
            self.analysis_t1 = t1
            self.window_mask = analysis_window_mask(self.trial.times, t0, t_end=t1)
            self.segments.clear()
            self.selected_segment_id = None
            self._clear_pending(redraw=False)
            self._reset_view()
            self._try_restore_autosave(trial_id)
            self._refresh_segment_list()
            self._redraw()
            duration = t1 - t0
            self._set_status(
                f"Loaded {trial_id}: {len(self.trial.times)} samples, "
                f"analysis {t0:.2f}–{t1:.2f} s ({duration:.1f} s)"
            )
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def _reset_view(self) -> None:
        self.view_xmin = self.analysis_t0
        self.view_xmax = min(
            self.analysis_t0 + self.DEFAULT_VIEW_DURATION, self.analysis_t1
        )
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

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax or self.trial is None:
            return
        if event.xdata is None or self.view_xmin is None or self.view_xmax is None:
            return

        if getattr(event, "step", 0):
            zoom_in = event.step > 0
        else:
            zoom_in = event.button == "up"

        scale = 0.8 if zoom_in else 1.25
        xdata = float(event.xdata)
        left = xdata - (xdata - self.view_xmin) * scale
        right = xdata + (self.view_xmax - xdata) * scale
        self.view_xmin = left
        self.view_xmax = right
        self._clamp_view()
        self._redraw()

    def _valid_click_mask(self) -> np.ndarray:
        if self.trial is None or self.window_mask is None:
            return np.array([], dtype=bool)
        return self.window_mask & ~np.isnan(self.trial.elevation_deg)

    def _valid_indices(self) -> np.ndarray:
        return np.where(self._valid_click_mask())[0]

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or self.trial is None:
            return
        if event.button != 1:
            return
        try:
            mask = self._valid_click_mask()
            idx = snap_index(self.trial.times, float(event.xdata), mask)
        except ValueError:
            return

        if self.pending_start_idx is None:
            self.pending_start_idx = idx
            self.pending_end_idx = None
            self.pending_fit = None
            t_start = self.trial.times[idx]
            self._set_status(
                f"Start marked at {t_start:.3f} s — click end of slow phase"
            )
        else:
            self.pending_end_idx = idx
            try:
                self.pending_fit = fit_segment(
                    self.trial.times,
                    self.trial.elevation_deg,
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

    def _renumber_segments(self) -> None:
        for i, seg in enumerate(self.segments, start=1):
            if seg.segment_id != i:
                self.segments[i - 1] = replace(seg, segment_id=i)

    def _selected_segment(self) -> SegmentFit | None:
        if self.selected_segment_id is None:
            return None
        for seg in self.segments:
            if seg.segment_id == self.selected_segment_id:
                return seg
        return None

    def _on_segment_select(self, _event=None) -> None:
        selection = self.seg_tree.selection()
        if not selection:
            self.selected_segment_id = None
        else:
            values = self.seg_tree.item(selection[0], "values")
            self.selected_segment_id = int(values[0])
        self._redraw()

    def _refresh_segment_list(self) -> None:
        for item in self.seg_tree.get_children():
            self.seg_tree.delete(item)
        for seg in self.segments:
            up = "✓" if seg.direction_upward else "!"
            r2 = f"{seg.r2:.2f}" if seg.r2 == seg.r2 else "—"
            iid = self.seg_tree.insert(
                "",
                tk.END,
                values=(
                    seg.segment_id,
                    f"{seg.t_start:.2f}",
                    f"{seg.t_end:.2f}",
                    f"{seg.gain:.3f}",
                    r2,
                    up,
                ),
            )
            if seg.segment_id == self.selected_segment_id:
                self.seg_tree.selection_set(iid)
                self.seg_tree.see(iid)

    def _nudge_segment_boundary(self, which: str, direction: int) -> None:
        seg = self._selected_segment()
        if seg is None or self.trial is None:
            return
        valid = self._valid_indices()
        if len(valid) == 0:
            return

        if which == "start":
            pos = int(np.searchsorted(valid, seg.idx_start))
            pos = int(np.clip(pos + direction, 0, len(valid) - 1))
            new_start = int(valid[pos])
            new_end = seg.idx_end
            if new_start == new_end:
                return
        else:
            pos = int(np.searchsorted(valid, seg.idx_end))
            pos = int(np.clip(pos + direction, 0, len(valid) - 1))
            new_end = int(valid[pos])
            new_start = seg.idx_start
            if new_start == new_end:
                return

        try:
            updated = fit_segment(
                self.trial.times,
                self.trial.elevation_deg,
                new_start,
                new_end,
                self._stimulus_velocity(),
                segment_id=seg.segment_id,
            )
        except ValueError as exc:
            self._set_status(str(exc))
            return

        for i, existing in enumerate(self.segments):
            if existing.segment_id == seg.segment_id:
                self.segments[i] = updated
                break

        self._refresh_segment_list()
        self._redraw()
        self._write_autosave()
        self._set_status(
            f"Segment #{updated.segment_id} adjusted: "
            f"gain={updated.gain:.3f}, R²={updated.r2:.3f}"
        )

    def _delete_selected_segment(self) -> None:
        seg = self._selected_segment()
        if seg is None:
            self._set_status("Select a segment in the list to delete.")
            return
        self.segments = [s for s in self.segments if s.segment_id != seg.segment_id]
        self._renumber_segments()
        self.selected_segment_id = None
        self._refresh_segment_list()
        self._redraw()
        self._write_autosave()
        med = trial_summary_median_gain(self.segments)
        n = len(self.segments)
        status = f"Deleted segment. {n} remaining."
        if n:
            status += f" Trial median gain={med:.3f}"
        self._set_status(status)

    def _accept_segment(self) -> None:
        if self.pending_fit is None:
            self._set_status("Mark a segment (two clicks) before accepting.")
            return
        self.segments.append(self.pending_fit)
        self._clear_pending(redraw=False)
        self._refresh_segment_list()
        self._redraw()
        self._write_autosave()
        med = trial_summary_median_gain(self.segments)
        self._set_status(
            f"Accepted segment {len(self.segments)}: "
            f"gain={self.segments[-1].gain:.3f}, R²={self.segments[-1].r2:.3f}, "
            f"trial median gain={med:.3f} (n={len(self.segments)})"
        )

    def _undo_segment(self) -> None:
        if self.segments:
            removed = self.segments.pop()
            if self.selected_segment_id == removed.segment_id:
                self.selected_segment_id = None
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

        if self.trial is None:
            self.ax.set_title("Load a trial to begin")
            self.canvas.draw_idle()
            return

        times = self.trial.times
        elev = self.trial.elevation_deg
        t0 = self.analysis_t0
        t1 = self.analysis_t1

        if self.view_xmin is None or self.view_xmax is None:
            self._reset_view()
        assert self.view_xmin is not None and self.view_xmax is not None
        vx0, vx1 = self.view_xmin, self.view_xmax

        plot_mask = (
            self.window_mask
            & ~np.isnan(elev)
            & (times >= vx0)
            & (times <= vx1)
        )
        self.ax.plot(
            times[plot_mask],
            elev[plot_mask],
            linestyle="none",
            marker=".",
            markersize=4,
            color="0.25",
            label="Elevation",
        )

        self.ax.axvspan(t0, t1, color="steelblue", alpha=0.04)
        self.ax.axvline(t0, color="steelblue", linestyle="--", alpha=0.35, linewidth=0.8)
        self.ax.axvline(t1, color="steelblue", linestyle="--", alpha=0.35, linewidth=0.8)

        for seg in self.segments:
            selected = seg.segment_id == self.selected_segment_id
            span_color = "limegreen" if selected else "green"
            span_alpha = 0.35 if selected else 0.2
            line_color = "forestgreen" if selected else "darkgreen"
            line_width = 2.5 if selected else 1.5
            self.ax.axvspan(seg.t_start, seg.t_end, color=span_color, alpha=span_alpha)
            seg_times = np.linspace(seg.t_start, seg.t_end, 50)
            self.ax.plot(
                seg_times,
                seg.slope_deg_s * seg_times + seg.intercept_deg,
                color=line_color,
                linewidth=line_width,
            )
            mid_t = (seg.t_start + seg.t_end) / 2
            mid_y = seg.slope_deg_s * mid_t + seg.intercept_deg
            self.ax.text(
                mid_t,
                mid_y,
                f"#{seg.segment_id}",
                color="white",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=line_color, alpha=0.85),
            )

        if self.pending_start_idx is not None:
            t_start = times[self.pending_start_idx]
            self.ax.axvline(t_start, color="orange", linestyle=":", linewidth=1.2)
            (marker,) = self.ax.plot(
                t_start, elev[self.pending_start_idx], "o", color="orange", markersize=8
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
                t_end, elev[self.pending_end_idx], "o", color="darkorange", markersize=8
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
            self.ax.set_title(
                f"Pending: slope={self.pending_fit.slope_deg_s:.2f} deg/s, "
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
            duration = t1 - t0
            title = f"{self.trial.trial_id} — {n} segment(s), {duration:.1f} s trial"
            if n:
                title += f", median gain={med:.3f}"
            self.ax.set_title(title)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Elevation (deg)")
        self.ax.set_xlim(vx0, vx1)

        vis_valid = plot_mask
        if np.any(vis_valid):
            pad = 0.5
            ymin = float(np.nanmin(elev[vis_valid])) - pad
            ymax = float(np.nanmax(elev[vis_valid])) + pad
            if ymin < ymax:
                self.ax.set_ylim(ymin, ymax)

        self.canvas.draw_idle()

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
