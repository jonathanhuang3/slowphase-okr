"""Tkinter + matplotlib GUI for manual OKR slow-phase annotation."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle

from slowphase_okr import __version__
from slowphase_okr.export import export_to_excel
from slowphase_okr.fit import SegmentFit, fit_segment, snap_index, trial_summary_median_gain
from slowphase_okr.gaze import GazeTrial, analysis_window_mask, load_ush2a_trial


class AnnotatorApp:
    ANALYSIS_DURATION = 40.0  # seconds (0 to 40 s from trial start)
    DEFAULT_VIEW_DURATION = 5.0  # visible time window at 90 Hz (~450 samples)
    MIN_VIEW_DURATION = 0.5
    MAX_VIEW_DURATION = ANALYSIS_DURATION
    PAN_STEP_SEC = 1.0

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"slowphase-okr v{__version__}")
        self.root.minsize(960, 640)

        self.gaze_path: Path | None = None
        self.time_path: Path | None = None
        self.trial: GazeTrial | None = None
        self.window_mask: np.ndarray | None = None  # type: ignore[name-defined]

        self.segments: list[SegmentFit] = []
        self.pending_start_idx: int | None = None
        self.pending_end_idx: int | None = None
        self.pending_fit: SegmentFit | None = None

        self._shade_patches: list[Rectangle] = []
        self._preview_line = None
        self._click_markers: list = []

        self.analysis_t0: float = 0.0
        self.analysis_t1: float = 40.0
        self.view_xmin: float | None = None
        self.view_xmax: float | None = None

        self._build_controls()
        self._build_plot()
        self._bind_keys()
        self._set_status("Load gaze and time files to begin.")

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

        ttk.Label(top, text="Stimulus velocity (deg/s):").grid(
            row=1, column=3, sticky=tk.E, padx=4
        )
        self.stim_vel_var = tk.StringVar(value="31")
        ttk.Entry(top, textvariable=self.stim_vel_var, width=8).grid(
            row=1, column=4, sticky=tk.W, padx=4
        )

        self.gaze_label = ttk.Label(top, text="Gaze: (none)", wraplength=420)
        self.gaze_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=4)
        self.time_label = ttk.Label(top, text="Time: (none)", wraplength=420)
        self.time_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=4)

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
            "Scroll to zoom time axis; Left/Right arrows to pan. "
            f"Default view: {self.DEFAULT_VIEW_DURATION:.0f} s. Analysis: 0–40 s. "
            "R² is logged, not filtered."
        )
        ttk.Label(self.root, text=help_text, padding=(8, 0)).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_plot(self) -> None:
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Elevation (deg)")
        self.ax.set_title("Elevation — mark slow-phase start and end")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _bind_keys(self) -> None:
        self.root.bind("a", lambda _e: self._accept_segment())
        self.root.bind("A", lambda _e: self._accept_segment())
        self.root.bind("u", lambda _e: self._undo_segment())
        self.root.bind("U", lambda _e: self._undo_segment())
        self.root.bind("<Escape>", lambda _e: self._clear_pending())
        self.root.bind("<Left>", lambda _e: self._pan_view(-self.PAN_STEP_SEC))
        self.root.bind("<Right>", lambda _e: self._pan_view(self.PAN_STEP_SEC))

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

    def _load_trial(self) -> None:
        if not self.gaze_path or not self.time_path:
            messagebox.showwarning("Missing files", "Select both gaze and time files.")
            return
        try:
            trial_id = self.trial_id_var.get().strip() or self.gaze_path.parent.name
            self.trial = load_ush2a_trial(
                self.gaze_path, self.time_path, trial_id=trial_id
            )
            t0 = float(self.trial.times[0])
            self.analysis_t0 = t0
            self.analysis_t1 = t0 + self.ANALYSIS_DURATION
            self.window_mask = analysis_window_mask(
                self.trial.times, t0, self.ANALYSIS_DURATION
            )
            self._reset_view()
            self.segments.clear()
            self._clear_pending(redraw=False)
            self._redraw()
            self._set_status(
                f"Loaded {trial_id}: {len(self.trial.times)} samples, "
                f"analysis {t0:.2f}–{t0 + self.ANALYSIS_DURATION:.1f} s"
            )
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def _reset_view(self) -> None:
        """Show the first DEFAULT_VIEW_DURATION seconds of the analysis window."""
        self.view_xmin = self.analysis_t0
        self.view_xmax = min(
            self.analysis_t0 + self.DEFAULT_VIEW_DURATION, self.analysis_t1
        )

    def _clamp_view(self) -> None:
        if self.view_xmin is None or self.view_xmax is None:
            self._reset_view()
            return
        width = self.view_xmax - self.view_xmin
        width = float(
            np.clip(width, self.MIN_VIEW_DURATION, self.MAX_VIEW_DURATION)
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

        # Scroll up = zoom in (narrower window); scroll down = zoom out
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

        self._redraw()

    def _accept_segment(self) -> None:
        if self.pending_fit is None:
            self._set_status("Mark a segment (two clicks) before accepting.")
            return
        self.segments.append(self.pending_fit)
        self._clear_pending(redraw=False)
        self._redraw()
        med = trial_summary_median_gain(self.segments)
        self._set_status(
            f"Accepted segment {len(self.segments)}: "
            f"gain={self.segments[-1].gain:.3f}, R²={self.segments[-1].r2:.3f}, "
            f"trial median gain={med:.3f} (n={len(self.segments)})"
        )

    def _undo_segment(self) -> None:
        if self.segments:
            self.segments.pop()
            self._redraw()
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
        self._shade_patches.clear()
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
            self.ax.axvspan(seg.t_start, seg.t_end, color="green", alpha=0.2)
            seg_times = np.linspace(seg.t_start, seg.t_end, 50)
            self.ax.plot(
                seg_times,
                seg.slope_deg_s * seg_times + seg.intercept_deg,
                color="darkgreen",
                linewidth=1.5,
            )

        if self.pending_start_idx is not None:
            t_start = times[self.pending_start_idx]
            self.ax.axvline(t_start, color="orange", linestyle=":", linewidth=1.2)
            (marker,) = self.ax.plot(t_start, elev[self.pending_start_idx], "o", color="orange")
            self._click_markers.append(marker)

        if self.pending_end_idx is not None:
            t_end = times[self.pending_end_idx]
            self.ax.axvline(t_end, color="darkorange", linestyle=":", linewidth=1.2)
            (marker,) = self.ax.plot(t_end, elev[self.pending_end_idx], "o", color="darkorange")
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
        else:
            med = trial_summary_median_gain(self.segments)
            n = len(self.segments)
            title = f"{self.trial.trial_id} — {n} segment(s)"
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
