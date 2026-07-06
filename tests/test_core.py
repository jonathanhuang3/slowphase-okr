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
from slowphase_okr.okr_log import load_okr_log


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


def test_analysis_window_mask_t_end():
    times = np.array([1.0, 5.0, 12.0, 25.0, 30.0])
    mask = analysis_window_mask(times, t0=1.0, t_end=25.0)
    assert list(mask) == [True, True, True, True, False]


def test_load_ush2a_trial(tmp_path: Path):
    gaze = tmp_path / "rotatedGaze.txt"
    gaze.write_text("(0.0, 0.1, 1.0)\n(0.0, 0.2, 1.0)\n(0.0, 0.3, 1.0)\n")
    timef = tmp_path / "gazeTime.txt"
    timef.write_text("0.0\n0.01\n0.02\n")
    trial = load_ush2a_trial(gaze, timef, trial_id="test")
    assert trial.trial_id == "test"
    assert len(trial.times) == 3
    assert np.all(np.diff(trial.elevation_deg[~np.isnan(trial.elevation_deg)]) > 0)


def test_load_ush2a_trial_with_nan_gaze(tmp_path: Path):
    gaze = tmp_path / "sranipalGazeSpace.txt"
    gaze.write_text(
        "(NaN, NaN, NaN)\n"
        "(NaN, NaN, NaN)\n"
        "(0.0, 0.1, 1.0)\n"
        "(0.0, 0.2, 1.0)\n"
    )
    timef = tmp_path / "sranipalGazeTime.txt"
    timef.write_text("1.0\n2.0\n3.0\n4.0\n")
    trial = load_ush2a_trial(gaze, timef, trial_id="nan_test")
    assert len(trial.times) == 4
    assert np.isnan(trial.elevation_deg[:2]).all()
    assert np.isfinite(trial.elevation_deg[2:]).all()


def test_median_gain():
    from slowphase_okr.fit import SegmentFit

    segs = [
        SegmentFit(1, 0, 1, 0, 1, 2, 20, 0, 20 / 31, 0.99, True, 31),
        SegmentFit(2, 2, 3, 1, 2, 2, 30, 0, 30 / 31, 0.98, True, 31),
    ]
    assert trial_summary_median_gain(segs) == pytest.approx((20 / 31 + 30 / 31) / 2)


def test_load_okr_log(tmp_path: Path):
    log = tmp_path / "OKR_Log_test.txt"
    log.write_text(
        "# OKR Condition Log\n"
        "eventIndex\teventType\teyePatch\tcontrastBlockIndex\tdotColor\t"
        "direction\tcontrastLevel\tdotStartSize\temissionRate\tusePersistentDots\t"
        "sessionContrastThreshold\tthresholdMultiplier\tisAnchor100\tstartTime\tendTime\n"
        "1\tInitialFixation\tLeft\tNA\tNA\tNA\tNA\tNA\tNA\t0\t0\t0\t0\t66.4\t70.4\n"
        "2\tAnchor100PersistentBlock\tLeft\t0\tWhite\tUp\t1.0\t0.7\t3000\t1\t0.076\tNA\t1\t70.42\t85.4\n"
        "3\tFixationITI\tLeft\t0\tNA\tNA\tNA\tNA\tNA\t0\t0\t0\t0\t85.41\t89.4\n"
        "6\tContrastBlock\tLeft\t2\tWhite\tDown\t0.1525\t0.7\t20000\t0\t0.076\t2\t0\t108.44\t123.4\n"
        "7\tCustomStimulusBlock\tLeft\t3\tWhite\tUp\t0.3\t0.7\t20000\t0\t0.076\t4\t0\t130.0\t145.0\n"
    )
    okr = load_okr_log(log)
    assert len(okr.block_markers) == 3
    assert len(okr.fixation_markers) == 2
    assert okr.block_markers[0].start_time == pytest.approx(70.42)
    assert okr.block_markers[0].label == "B0↑"
    assert okr.block_markers[1].label == "B2↓"
    assert okr.block_markers[2].event_type == "CustomStimulusBlock"
    assert okr.block_markers[2].label == "B3↑"
    assert okr.fixation_markers[0].event_type == "InitialFixation"
    assert okr.fixation_markers[1].start_time == pytest.approx(85.41)


def test_autosave_roundtrip(tmp_path: Path):
    from slowphase_okr.autosave import (
        autosave_matches_trial,
        load_autosave,
        save_autosave,
        segments_from_autosave,
    )
    from slowphase_okr.fit import SegmentFit

    gaze = tmp_path / "rotatedGaze.txt"
    timef = tmp_path / "gazeTime.txt"
    segments = [
        SegmentFit(1, 0, 1, 0.0, 0.1, 2, 20.0, 0.0, 20.0 / 31.0, 0.99, True, 31.0),
    ]
    out = save_autosave(
        tmp_path / "trial_slowphase_okr_autosave.json",
        trial_id="trial",
        gaze_source=str(gaze.resolve()),
        time_source=str(timef.resolve()),
        stimulus_velocity=31.0,
        segments=segments,
        software_version="0.1.1",
    )
    data = load_autosave(out)
    assert data is not None
    assert autosave_matches_trial(data, str(gaze.resolve()), str(timef.resolve()))
    restored = segments_from_autosave(data)
    assert len(restored) == 1
    assert restored[0].gain == pytest.approx(20.0 / 31.0)
