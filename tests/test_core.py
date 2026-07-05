"""Tests for slowphase-okr."""

from pathlib import Path

import numpy as np
import pytest

from slowphase_okr.fit import fit_segment, snap_index, trial_summary_median_gain
from slowphase_okr.gaze import (
    analysis_window_mask,
    load_ush2a_trial,
    unity_gaze_direction,
)


def test_unity_gaze_direction_forward():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 0.0])
    z = np.array([1.0, 0.0])
    az, el, r = unity_gaze_direction(x, y, z)
    assert r[0] == pytest.approx(1.0)
    assert az[1] == pytest.approx(np.pi / 2)
    assert el[1] == pytest.approx(0.0)


def test_fit_segment_upward():
    times = np.linspace(0, 1, 100)
    elev = 25.0 * times + 1.0
    seg = fit_segment(times, elev, 10, 80, stimulus_velocity=31.0, segment_id=1)
    assert seg.slope_deg_s == pytest.approx(25.0, rel=1e-2)
    assert seg.gain == pytest.approx(25.0 / 31.0, rel=1e-2)
    assert seg.r2 == pytest.approx(1.0, abs=1e-6)
    assert seg.direction_upward is True


def test_snap_index():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    mask = np.array([True, True, True, True])
    assert snap_index(times, 0.12, mask) == 1
    assert snap_index(times, 0.29, mask) == 3


def test_analysis_window_mask():
    times = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    mask = analysis_window_mask(times, t0=0.0, duration_sec=40.0)
    assert list(mask) == [True, True, True, True, True, False]


def test_load_ush2a_trial(tmp_path: Path):
    gaze = tmp_path / "rotatedGaze.txt"
    gaze.write_text("(0.0, 0.1, 1.0)\n(0.0, 0.2, 1.0)\n(0.0, 0.3, 1.0)\n")
    timef = tmp_path / "gazeTime.txt"
    timef.write_text("0.0\n0.01\n0.02\n")
    trial = load_ush2a_trial(gaze, timef, trial_id="test")
    assert trial.trial_id == "test"
    assert len(trial.times) == 3
    assert np.all(np.diff(trial.elevation_deg) > 0)


def test_median_gain():
    from slowphase_okr.fit import SegmentFit

    segs = [
        SegmentFit(1, 0, 1, 0, 1, 2, 20, 0, 20 / 31, 0.99, True, 31),
        SegmentFit(2, 2, 3, 1, 2, 2, 30, 0, 30 / 31, 0.98, True, 31),
    ]
    assert trial_summary_median_gain(segs) == pytest.approx((20 / 31 + 30 / 31) / 2)
