"""Tests for slowphase-okr."""

from pathlib import Path

import numpy as np
import pytest

from slowphase_okr.fit import fit_segment, snap_index, trial_summary_median_gain
from slowphase_okr.gaze import (
    analysis_window_mask,
    attach_sranipal_pupil,
    discover_pupil_files,
    infer_viewing_eye,
    load_sranipal_pupil,
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


def test_load_okr_log(tmp_path: Path):
    log = tmp_path / "OKR_Log_test.txt"
    log.write_text(
        "# OKR Condition Log\n"
        "# StimulusName: Comb4 Block3 RE Increment White Dots\n"
        "# StimulusEyePatch: Left\n"
        "eventIndex\teventType\teyePatch\tcontrastBlockIndex\tdotColor\t"
        "direction\tcontrastLevel\tdotStartSize\temissionRate\tusePersistentDots\t"
        "sessionContrastThreshold\tthresholdMultiplier\tisAnchor100\tstartTime\tendTime\n"
        "1\tInitialFixation\tLeft\tNA\tNA\tNA\tNA\tNA\tNA\t0\t0\t0\t0\t66.4\t70.4\n"
        "2\tAnchor100PersistentBlock\tLeft\t0\tWhite\tUp\t1.0\t0.7\t3000\t1\t0.076\tNA\t1\t70.42\t85.4\n"
        "3\tFixationITI\tLeft\t0\tNA\tNA\tNA\tNA\tNA\t0\t0\t0\t0\t85.41\t89.4\n"
        "6\tContrastBlock\tLeft\t2\tWhite\tDown\t0.1525\t0.7\t20000\t0\t0.076\t2\t0\t108.44\t123.4\n"
        "7\tCustomStimulusBlock\tLeft\t3\tWhite\tUp\t0.3\t0.7\t20000\t0\t0.076\t4\t0\t130.0\t145.0\n"
        "8\tAnchor100FlickerBlock\tLeft\t4\tWhite\tUp\t1.0\t0.7\t20000\t0\t0.076\tNA\t1\t150.0\t165.0\n"
    )
    okr = load_okr_log(log)
    assert len(okr.block_markers) == 4
    assert len(okr.fixation_markers) == 2
    assert okr.stimulus_name == "Comb4 Block3 RE Increment White Dots"
    assert okr.stimulus_eye_patch == "Left"
    assert okr.block_markers[0].start_time == pytest.approx(70.42)
    assert okr.block_markers[0].end_time == pytest.approx(85.4)
    assert okr.block_markers[0].label == "B0↑"
    assert okr.block_markers[0].use_persistent_dots is True
    assert okr.block_markers[0].is_anchor100 is True
    assert okr.block_markers[1].label == "B2↓"
    assert okr.block_markers[1].threshold_multiplier == pytest.approx(2.0)
    assert okr.block_markers[2].event_type == "CustomStimulusBlock"
    assert okr.block_markers[2].label == "B3↑"
    assert okr.block_markers[3].event_type == "Anchor100FlickerBlock"
    assert okr.block_markers[3].use_persistent_dots is False
    assert okr.fixation_markers[0].event_type == "InitialFixation"
    assert okr.fixation_markers[1].start_time == pytest.approx(85.41)


def test_condition_at_time(tmp_path: Path):
    from slowphase_okr.okr_log import condition_at_time

    log = tmp_path / "OKR_Log_test.txt"
    log.write_text(
        "# OKR Condition Log\n"
        "# StimulusName: Comb4 Block3 RE Increment White Dots\n"
        "eventIndex\teventType\teyePatch\tcontrastBlockIndex\tdotColor\t"
        "direction\tcontrastLevel\tdotStartSize\temissionRate\tusePersistentDots\t"
        "sessionContrastThreshold\tthresholdMultiplier\tisAnchor100\tstartTime\tendTime\n"
        "1\tInitialFixation\tLeft\tNA\tNA\tNA\tNA\tNA\tNA\t0\t0\t0\t0\t66.4\t70.4\n"
        "2\tAnchor100PersistentBlock\tLeft\t0\tWhite\tUp\t1.0\t0.7\t3000\t1\t0.076\tNA\t1\t70.42\t85.4\n"
        "6\tContrastBlock\tLeft\t2\tWhite\tDown\t0.1525\t0.7\t20000\t0\t0.076\t2\t0\t108.44\t123.4\n"
        "8\tAnchor100FlickerBlock\tLeft\t3\tWhite\tUp\t1.0\t0.7\t20000\t0\t0.076\tNA\t1\t130.0\t145.0\n"
    )
    okr = load_okr_log(log)
    text = condition_at_time(okr, 75.0)
    assert "Session: Increment · White dots" in text
    assert "B0" in text
    assert "Up" in text
    assert "Persistent" in text
    assert "contrast 1" in text

    text_down = condition_at_time(okr, 110.0)
    assert "B2" in text_down
    assert "Down" in text_down
    assert "Flicker" in text_down
    assert "2×T" in text_down

    text_flicker = condition_at_time(okr, 135.0)
    assert "Flicker" in text_flicker
    assert "B3" in text_flicker

    text_fix = condition_at_time(okr, 68.0)
    assert "Initial fixation" in text_fix


def test_median_gain():
    from slowphase_okr.fit import SegmentFit

    segs = [
        SegmentFit(1, 0, 1, 0, 1, 2, 20, 0, 20 / 31, 0.99, True, 31),
        SegmentFit(2, 2, 3, 1, 2, 2, 30, 0, 30 / 31, 0.98, True, 31),
    ]
    assert trial_summary_median_gain(segs) == pytest.approx((20 / 31 + 30 / 31) / 2)


def test_detect_upward_slow_phases():
    from slowphase_okr.detect import DetectParams, detect_slow_phases

    times = np.linspace(0, 2, 400)
    elev = np.zeros_like(times)
    elev[50:120] = 20.0 * (times[50:120] - times[50])
    elev[200:270] = 20.0 * (times[200:270] - times[200])
    valid = np.ones(len(times), dtype=bool)

    params = DetectParams(
        direction="up",
        min_duration_sec=0.05,
        min_r2=0.9,
        max_saccade_velocity_deg_s=200.0,
        window_sec=0.08,
        merge_gap_sec=0.04,
    )
    found = detect_slow_phases(times, elev, valid, 31.0, params)
    assert len(found) >= 2
    assert all(seg.direction_upward for seg in found)
    assert found == sorted(found, key=lambda s: s.t_start)


def test_detect_respects_exclude():
    from slowphase_okr.detect import DetectParams, detect_slow_phases
    from slowphase_okr.fit import SegmentFit

    times = np.linspace(0, 1, 200)
    elev = 25.0 * times
    valid = np.ones(len(times), dtype=bool)
    params = DetectParams(
        direction="up",
        min_duration_sec=0.05,
        min_r2=0.9,
        max_saccade_velocity_deg_s=200.0,
        window_sec=0.05,
    )
    exclude = [
        SegmentFit(1, 0, 150, times[0], times[150], 151, 25.0, 0.0, 25 / 31, 0.99, True, 31.0),
    ]
    found = detect_slow_phases(times, elev, valid, 31.0, params, exclude=exclude)
    assert not any(seg.t_start <= times[150] <= seg.t_end for seg in found)


def test_detect_rejects_saccade_inside_segment():
    from slowphase_okr.detect import DetectParams, detect_slow_phases

    times = np.linspace(0, 1, 300)
    elev = 20.0 * times
    elev[150] += 8.0
    valid = np.ones(len(times), dtype=bool)
    strict = DetectParams(
        direction="up",
        min_duration_sec=0.05,
        min_r2=0.5,
        max_saccade_velocity_deg_s=50.0,
        window_sec=0.05,
    )
    loose = DetectParams(
        direction="up",
        min_duration_sec=0.05,
        min_r2=0.5,
        max_saccade_velocity_deg_s=500.0,
        window_sec=0.05,
    )
    strict_found = detect_slow_phases(times, elev, valid, 31.0, strict)
    loose_found = detect_slow_phases(times, elev, valid, 31.0, loose)
    assert len(loose_found) >= len(strict_found)


def test_collapse_duplicate_timestamps():
    from slowphase_okr.detect import collapse_duplicate_timestamps

    times = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    elev = np.array([0.0, 1.0, 2.0, 4.0, 6.0, 9.0])
    valid = np.ones(len(times), dtype=bool)
    collapsed = collapse_duplicate_timestamps(times, elev, valid)
    assert len(collapsed.times) == 3
    assert collapsed.elevation_deg[0] == pytest.approx(1.0)
    assert collapsed.first_orig_idx[0] == 0
    assert collapsed.last_orig_idx[0] == 2


def test_detect_merge_gap_joins_fragments():
    from slowphase_okr.detect import DetectParams, detect_slow_phases

    times = np.linspace(0, 1, 500)
    elev = np.zeros_like(times)
    elev[50:120] = 20.0 * (times[50:120] - times[50])
    elev[125:195] = elev[119] + 20.0 * (times[125:195] - times[125])
    valid = np.ones(len(times), dtype=bool)
    merged = DetectParams(
        direction="up",
        min_duration_sec=0.05,
        min_r2=0.85,
        max_saccade_velocity_deg_s=200.0,
        window_sec=0.06,
        merge_gap_sec=0.05,
        refine_boundaries=False,
    )
    separate = DetectParams(
        direction="up",
        min_duration_sec=0.05,
        min_r2=0.85,
        max_saccade_velocity_deg_s=200.0,
        window_sec=0.06,
        merge_gap_sec=0.0,
        refine_boundaries=False,
    )
    merged_found = detect_slow_phases(times, elev, valid, 31.0, merged)
    separate_found = detect_slow_phases(times, elev, valid, 31.0, separate)
    assert len(merged_found) <= len(separate_found)


def test_gain_rescales_when_stimulus_velocity_changes():
    times = np.linspace(0, 1, 50)
    elev = 10.0 * times
    seg = fit_segment(times, elev, 0, 49, stimulus_velocity=31.0, segment_id=1)
    assert abs(seg.slope_deg_s - 10.0) < 0.01
    assert abs(seg.gain - 10.0 / 31.0) < 0.01
    rescaled_gain = seg.slope_deg_s / 10.0
    assert abs(rescaled_gain - seg.gain * (31.0 / 10.0)) < 1e-9


def test_load_sranipal_pupil(tmp_path: Path):
    pos = tmp_path / "sranipalRightPupilPositions.txt"
    pos.write_text("(0.5, 0.4)\n(0.51, 0.41)\n(0.52, 0.42)\n")
    timef = tmp_path / "sranipalRightPupilPositionTimes.txt"
    timef.write_text("1.0\n1.01\n1.02\n")
    pupil = load_sranipal_pupil(pos, timef, eye="right")
    assert pupil.eye == "right"
    assert len(pupil.times) == 3
    assert pupil.y[0] == pytest.approx(0.4)
    assert np.all(np.diff(pupil.y) > 0)


def test_discover_pupil_files(tmp_path: Path):
    pos = tmp_path / "sranipalLeftPupilPositions.txt"
    pos.write_text("(0.1, 0.2)\n")
    timef = tmp_path / "sranipalLeftPupilTimes.txt"
    timef.write_text("0.0\n")
    found = discover_pupil_files(tmp_path, viewing_eye="left")
    assert found is not None
    assert found[0] == pos
    assert found[1] == timef
    assert discover_pupil_files(tmp_path, viewing_eye="right") is None


def test_infer_viewing_eye():
    assert infer_viewing_eye(eye_patch="Left") == "right"
    assert infer_viewing_eye(eye_patch="Right") == "left"
    assert infer_viewing_eye(trial_id="Comb4 Block3 RE Increment") == "right"
    assert infer_viewing_eye(trial_id="Comb2 Block1 LE") == "left"


def test_attach_sranipal_pupil(tmp_path: Path):
    gaze = tmp_path / "rotatedGaze.txt"
    gaze.write_text("(0.0, 0.1, 1.0)\n")
    timef = tmp_path / "gazeTime.txt"
    timef.write_text("0.0\n")
    trial = load_ush2a_trial(gaze, timef, trial_id="test RE")
    pos = tmp_path / "sranipalRightPupilPositions.txt"
    pos.write_text("(0.5, 0.4)\n")
    pupil_time = tmp_path / "sranipalRightPupilPositionTimes.txt"
    pupil_time.write_text("0.0\n")
    attached = attach_sranipal_pupil(trial, tmp_path)
    assert attached is not None
    assert trial.pupil is not None
    assert trial.pupil.eye == "right"


def test_refit_segment_by_time():
    from slowphase_okr.fit import refit_segment_by_time

    times = np.linspace(0, 1, 100)
    elev = 25.0 * times + 1.0
    valid = np.ones(len(times), dtype=bool)
    seg = refit_segment_by_time(times, elev, 0.1, 0.8, 31.0, 1, valid)
    assert seg.slope_deg_s == pytest.approx(25.0, rel=1e-2)
    assert seg.gain == pytest.approx(25.0 / 31.0, rel=1e-2)


def test_autosave_roundtrip(tmp_path: Path):
    from slowphase_okr.autosave import (
        autosave_matches_trial,
        autosave_path,
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
        tmp_path / "trial_slowphase_okr_markings.json",
        trial_id="trial",
        gaze_source=str(gaze.resolve()),
        time_source=str(timef.resolve()),
        stimulus_velocity=31.0,
        segments=segments,
        software_version="0.2.6",
    )
    data = load_autosave(out)
    assert data is not None
    assert autosave_matches_trial(data, str(gaze.resolve()), str(timef.resolve()))
    restored = segments_from_autosave(data)
    assert len(restored) == 1
    assert restored[0].gain == pytest.approx(20.0 / 31.0)


def test_autosave_path_uses_directory(tmp_path: Path):
    from slowphase_okr.autosave import autosave_path, markings_id_from_gaze_path

    personal = tmp_path / "my_markings"
    path = autosave_path(personal, "Patient016_LE")
    assert path == personal / "Patient016_LE_slowphase_okr_markings.json"

    gaze = (
        tmp_path
        / "Jonathan Test"
        / "Param Block3 RE Increment 1deg 30dps White Dots"
        / "rotatedGaze.txt"
    )
    gaze.parent.mkdir(parents=True)
    gaze.write_text("(0,0,1)\n")
    stem = markings_id_from_gaze_path(gaze)
    assert stem == (
        "Jonathan Test_Param Block3 RE Increment 1deg 30dps White Dots"
    )
    assert autosave_path(personal, stem).name.endswith("_slowphase_okr_markings.json")


def test_annotations_dir_prefs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from slowphase_okr import prefs

    monkeypatch.setattr(prefs, "prefs_path", lambda: tmp_path / ".slowphase_okr_prefs.json")
    assert prefs.get_annotations_dir() is None
    chosen = prefs.set_annotations_dir(tmp_path / "markings")
    assert chosen == (tmp_path / "markings").resolve()
    assert prefs.get_annotations_dir() == chosen

