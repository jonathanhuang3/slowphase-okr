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
    is_tobii_glasses3_gazedata,
    load_gaze_trial,
    load_sranipal_pupil,
    load_tobii_glasses3_trial,
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
    assert seg.rmse_deg == pytest.approx(0.0, abs=1e-9)
    assert seg.direction_upward is True


def test_fit_segment_rmse_increases_with_noise():
    rng = np.random.default_rng(0)
    times = np.linspace(0, 1, 200)
    elev_clean = 20.0 * times
    elev_noisy = elev_clean + rng.normal(0.0, 0.5, size=times.shape)
    clean = fit_segment(times, elev_clean, 0, 199, stimulus_velocity=20.0, segment_id=1)
    noisy = fit_segment(times, elev_noisy, 0, 199, stimulus_velocity=20.0, segment_id=2)
    assert clean.rmse_deg == pytest.approx(0.0, abs=1e-9)
    assert noisy.rmse_deg > 0.3
    assert noisy.r2 < clean.r2


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
    assert trial.azimuth_deg is not None
    assert len(trial.azimuth_deg) == 3
    assert np.all(np.diff(trial.elevation_deg[~np.isnan(trial.elevation_deg)]) > 0)


def test_load_ush2a_trial_azimuth(tmp_path: Path):
    gaze = tmp_path / "rotatedGaze.txt"
    # Increasing x with fixed z → increasing azimuth (avoid 0,0 which is treated as invalid)
    gaze.write_text("(0.1, 0.05, 1.0)\n(0.3, 0.05, 1.0)\n(0.5, 0.05, 1.0)\n")
    timef = tmp_path / "gazeTime.txt"
    timef.write_text("0.0\n0.01\n0.02\n")
    trial = load_ush2a_trial(gaze, timef, trial_id="az_test")
    assert trial.azimuth_deg is not None
    assert np.all(np.isfinite(trial.azimuth_deg))
    assert np.all(np.diff(trial.azimuth_deg) > 0)


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
    assert trial.azimuth_deg is not None
    assert np.isnan(trial.azimuth_deg[:2]).all()
    assert np.isfinite(trial.azimuth_deg[2:]).all()


def test_load_tobii_glasses3_trial(tmp_path: Path):
    import json

    path = tmp_path / "recording" / "gazedata.json"
    path.parent.mkdir()
    rows = [
        {
            "type": "gaze",
            "timestamp": 0.0,
            "data": {
                "gaze2d": [0.5, 0.5],
                "gaze3d": [0.0, 0.0, 500.0],
                "eyeleft": {
                    "gazedirection": [0.0, 0.0, 1.0],
                    "pupildiameter": 3.0,
                },
                "eyeright": {
                    "gazedirection": [0.0, 0.0, 1.0],
                    "pupildiameter": 3.0,
                },
            },
        },
        {"type": "gaze", "timestamp": 0.02, "data": {}},  # tracking lost
        {
            "type": "gaze",
            "timestamp": 0.04,
            "data": {
                "eyeleft": {"gazedirection": [0.0, np.sin(np.radians(10)), np.cos(np.radians(10))]},
                "eyeright": {"gazedirection": [0.0, np.sin(np.radians(10)), np.cos(np.radians(10))]},
            },
        },
        {
            "type": "gaze",
            "timestamp": 0.06,
            "data": {
                # only gaze3d
                "gaze3d": [0.0, 100.0, 100.0],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assert is_tobii_glasses3_gazedata(path)
    trial = load_tobii_glasses3_trial(path)
    assert trial.source_format == "tobii_glasses3"
    assert trial.source_time == trial.source_gaze
    assert len(trial.times) == 3  # empty data skipped
    assert trial.times[0] == pytest.approx(0.0)
    assert trial.elevation_deg[0] == pytest.approx(0.0, abs=1e-6)
    assert trial.elevation_deg[1] == pytest.approx(10.0, abs=1e-3)
    assert trial.azimuth_deg is not None
    assert np.isfinite(trial.elevation_deg[2])
    assert trial.has_per_eye_gaze()
    assert trial.elevation_left_deg is not None
    assert trial.elevation_right_deg is not None
    assert trial.elevation_left_deg[1] == pytest.approx(10.0, abs=1e-3)
    assert trial.elevation_right_deg[1] == pytest.approx(10.0, abs=1e-3)
    # gaze3d-only sample: binocular finite, per-eye NaN
    assert np.isfinite(trial.elevation_deg[2])
    assert np.isnan(trial.elevation_left_deg[2])
    assert np.isnan(trial.elevation_right_deg[2])

    via_dispatch = load_gaze_trial(path, trial_id="g3")
    assert via_dispatch.source_format == "tobii_glasses3"
    assert via_dispatch.trial_id == "g3"


def test_tobii_left_right_differ(tmp_path: Path):
    import json

    path = tmp_path / "gazedata.json"
    elev_l, elev_r = 5.0, -3.0
    row = {
        "type": "gaze",
        "timestamp": 1.0,
        "data": {
            "eyeleft": {
                "gazedirection": [
                    0.0,
                    float(np.sin(np.radians(elev_l))),
                    float(np.cos(np.radians(elev_l))),
                ]
            },
            "eyeright": {
                "gazedirection": [
                    0.0,
                    float(np.sin(np.radians(elev_r))),
                    float(np.cos(np.radians(elev_r))),
                ]
            },
        },
    }
    path.write_text(json.dumps(row) + "\n")
    trial = load_tobii_glasses3_trial(path)
    assert trial.elevation_left_deg is not None
    assert trial.elevation_right_deg is not None
    assert trial.elevation_left_deg[0] == pytest.approx(elev_l, abs=1e-3)
    assert trial.elevation_right_deg[0] == pytest.approx(elev_r, abs=1e-3)
    assert trial.elevation_deg[0] == pytest.approx(0.5 * (elev_l + elev_r), abs=1e-3)


def test_is_tobii_rejects_unity_gaze(tmp_path: Path):
    gaze = tmp_path / "rotatedGaze.txt"
    gaze.write_text("(0.0, 0.1, 1.0)\n(0.0, 0.2, 1.0)\n")
    assert not is_tobii_glasses3_gazedata(gaze)


def _minimal_eyelink_asc(
    tmp_path: Path,
    samples=None,
    synctime=1000,
):
    """Write a tiny binocular EyeLink ASC with vertical gaze ramp."""
    path = tmp_path / "test.asc"
    if samples is None:
        samples = [
            (1000, 960.0, 540.0, 960.0, 540.0),
            (1001, 960.0, 530.0, 960.0, 530.0),
            (1002, 960.0, 520.0, 960.0, 520.0),
        ]
    lines = [
        "** VERSION: EYELINK 3",
        "MSG\t500 DISPLAY_COORDS 0 0 1919 1079",
        "MSG\t600 VALIDATE LR POINT 0  LEFT  at 960,540  OFFSET 1.00 deg.  0.0,-40.0 pix.",
        "START\t900 \tLEFT\tRIGHT\tSAMPLES\tEVENTS",
        "SAMPLES\tGAZE\tLEFT\tRIGHT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2",
    ]
    if synctime is not None:
        lines.append(f"MSG\t{synctime} SYNCTIME")
    for ts, lx, ly, rx, ry in samples:
        lines.append(
            f"{ts}\t{lx:.1f}\t{ly:.1f}\t59.0\t{rx:.1f}\t{ry:.1f}\t......"
        )
    lines.append("MSG\t2000 TRIAL_RESULT 0")
    lines.append("END\t2001 \tSAMPLES\tEVENTS")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_eyelink_asc_loader(tmp_path: Path):
    from slowphase_okr.eyelink_asc import is_eyelink_asc, load_eyelink_asc_trial

    path = _minimal_eyelink_asc(tmp_path)
    assert is_eyelink_asc(path)
    trial = load_eyelink_asc_trial(path, trial_id="el")
    assert trial.source_format == "eyelink_asc"
    assert trial.trial_id == "el"
    assert len(trial.times) == 3
    assert trial.times[0] == pytest.approx(0.0)
    assert trial.times[-1] == pytest.approx(0.002)
    assert trial.has_per_eye_gaze()
    # 10 px up at ~40 px/deg => ~0.25 deg; center sample is 0
    assert trial.elevation_deg[0] == pytest.approx(0.0, abs=0.05)
    assert trial.elevation_deg[-1] == pytest.approx(0.5, abs=0.1)
    assert trial.azimuth_deg is not None
    assert np.allclose(trial.azimuth_deg, 0.0, atol=0.05, equal_nan=True)

    via = load_gaze_trial(path, trial_id="via")
    assert via.source_format == "eyelink_asc"


def test_eyelink_asc_rejects_non_eyelink(tmp_path: Path):
    from slowphase_okr.eyelink_asc import is_eyelink_asc

    path = tmp_path / "notes.asc"
    path.write_text("random text\n")
    assert not is_eyelink_asc(path)


def test_eyelink_asc_hpose_and_eye_in_head(tmp_path: Path):
    from slowphase_okr.eyelink_asc import load_eyelink_asc_trial

    path = tmp_path / "el3.asc"
    lines = [
        "** VERSION: EYELINK 3",
        "MSG\t500 DISPLAY_COORDS 0 0 1919 1079",
        "MSG\t600 VALIDATE LR POINT 0  LEFT  at 960,540  OFFSET 1.00 deg.  0.0,-40.0 pix.",
        "START\t900 \tLEFT\tRIGHT\tSAMPLES\tEVENTS",
        "SAMPLES\tGAZE\tLEFT\tRIGHT\tHPOSE\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2",
        "MSG\t1000 SYNCTIME",
    ]
    # time Lx Ly Lp Rx Ry Rp roll pitch yaw x y z headx heady + eye cols ... eihLy eihRy
    row = (
        "1000\t960.0\t540.0\t59.0\t960.0\t540.0\t59.0"
        "\t......\t10.0\t2.0\t3.0\t-19.0\t-19.0\t-607.0"
        "\t....\t960.0\t540.0"
        "\t960.0\t960.0\t960.0"
        "\t530.0\t520.0\t525.0"
    )
    lines.append(row)
    lines.append("MSG\t2000 TRIAL_RESULT 0")
    lines.append("END\t2001 \tSAMPLES\tEVENTS")
    path.write_text("\n".join(lines) + "\n")

    trial = load_eyelink_asc_trial(path)
    assert trial.has_heading()
    assert trial.heading is not None
    assert trial.heading.roll_deg[0] == pytest.approx(0.0, abs=1e-6)
    assert trial.heading.pitch_deg[0] == pytest.approx(0.0, abs=1e-6)
    assert trial.has_eye_in_head()
    assert trial.eye_in_head is not None
    assert trial.eye_in_head.elevation_left_deg[0] == pytest.approx(0.25, abs=0.05)


def test_eyelink_asc_monocular_loader(tmp_path: Path):
    from slowphase_okr.eyelink_asc import load_eyelink_asc_trial

    path = tmp_path / "mono.asc"
    lines = [
        "** VERSION: EYELINK 3",
        "MSG\t500 DISPLAY_COORDS 0 0 1919 1079",
        "MSG\t600 VALIDATE LR POINT 0  LEFT  at 960,540  OFFSET 1.00 deg.  0.0,-40.0 pix.",
        "START\t900 \tLEFT\tSAMPLES\tEVENTS",
        "SAMPLES\tGAZE\tLEFT\tHPOSE\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2",
        "MSG\t1000 SYNCTIME",
        # Lx Ly Lp roll pitch yaw x y z headx heady pad eihLy
        "1000\t960.0\t540.0\t59.0\t10.0\t2.0\t3.0\t-19.0\t-19.0\t-607.0"
        "\t960.0\t540.0\t0.0\t530.0",
        "1001\t960.0\t530.0\t59.0\t10.0\t2.0\t3.0\t-19.0\t-19.0\t-607.0"
        "\t960.0\t540.0\t0.0\t520.0",
        "MSG\t2000 TRIAL_RESULT 0",
        "END\t2001 \tSAMPLES\tEVENTS",
    ]
    path.write_text("\n".join(lines) + "\n")

    trial = load_eyelink_asc_trial(path, trial_id="mono")
    assert trial.source_format == "eyelink_asc"
    assert not trial.has_per_eye_gaze()
    assert trial.elevation_right_deg is None
    assert trial.azimuth_right_deg is None
    assert trial.elevation_left_deg is not None
    assert trial.elevation_deg[0] == pytest.approx(0.0, abs=0.05)
    assert trial.elevation_deg[-1] == pytest.approx(0.25, abs=0.1)
    assert trial.has_heading()
    assert trial.has_eye_in_head()
    assert trial.eye_in_head is not None
    assert trial.eye_in_head.elevation_left_deg[0] == pytest.approx(0.25, abs=0.05)
    assert np.all(np.isnan(trial.eye_in_head.elevation_right_deg))


def test_eyelink_blink_intervals(tmp_path: Path):
    from slowphase_okr.eyelink_asc import load_eyelink_asc_trial
    from slowphase_okr.gaze import mask_during_blink_intervals

    path = tmp_path / "blink.asc"
    lines = [
        "** VERSION: EYELINK 3",
        "MSG\t500 DISPLAY_COORDS 0 0 1919 1079",
        "MSG\t600 VALIDATE LR POINT 0  LEFT  at 960,540  OFFSET 1.00 deg.  0.0,-40.0 pix.",
        "START\t900 \tLEFT\tSAMPLES\tEVENTS",
        "SAMPLES\tGAZE\tLEFT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2",
        "MSG\t1000 SYNCTIME",
        "1000\t960.0\t540.0\t59.0",
        "SBLINK L 1001",
        "1001\t960.0\t530.0\t59.0",
        "EBLINK L 1001\t1005\t4",
        "1002\t960.0\t520.0\t59.0",
        "MSG\t2000 TRIAL_RESULT 0",
        "END\t2001 \tSAMPLES\tEVENTS",
    ]
    path.write_text("\n".join(lines) + "\n")

    trial = load_eyelink_asc_trial(path)
    assert trial.has_blink_intervals()
    assert trial.blink_intervals_sec is not None
    assert len(trial.blink_intervals_sec) == 1
    start, end = trial.blink_intervals_sec[0]
    assert start == pytest.approx(0.001)
    assert end == pytest.approx(0.005)

    masked = mask_during_blink_intervals(
        trial.times,
        trial.elevation_deg,
        trial.blink_intervals_sec,
        pad_sec=0.0,
    )
    assert np.isfinite(masked[0])
    assert np.isnan(masked[1])


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
    assert okr.block_markers[0].dot_start_size == pytest.approx(0.7)
    assert okr.block_markers[0].emission_rate == pytest.approx(3000)
    assert okr.block_markers[1].label == "B2↓"
    assert okr.block_markers[1].threshold_multiplier == pytest.approx(2.0)
    assert okr.block_markers[2].event_type == "CustomStimulusBlock"
    assert okr.block_markers[2].label == "B3↑"
    assert okr.block_markers[3].event_type == "Anchor100FlickerBlock"
    assert okr.block_markers[3].use_persistent_dots is False
    assert okr.fixation_markers[0].event_type == "InitialFixation"
    assert okr.fixation_markers[1].start_time == pytest.approx(85.41)


def test_load_legacy_trial_okr_log(tmp_path: Path):
    """Older TrialIndex logs (dot-size / emission-rate schedules)."""
    from slowphase_okr.okr_log import condition_at_time, segment_condition_fields

    log = tmp_path / "OKR_Log_legacy.txt"
    log.write_text(
        "# OKR Condition Log\n"
        "# PatientName: June162026 Test 2\n"
        "# StimulusName: White Dots Up 100% Contrast Left Dot Size Testing\n"
        "# TimeBase: Unity Time.time (seconds since Play; matches gazeTimes.txt)\n"
        "TrialIndex\tdotColor\tdirection\tcontrastLevel\tdotStartSize\t"
        "emissionRate\tstartTime\tendTime\n"
        "1\tWhite\tUp\t1.000000\t2.800000\t7500.00\t10.883830\t30.888290\n"
        "2\tWhite\tUp\t1.000000\t0.700000\t10000.00\t34.908240\t54.911190\n"
        "3\tWhite\tUp\t1.000000\t2.800000\t5000.00\t58.934400\t78.922630\n"
    )
    okr = load_okr_log(log)
    assert len(okr.block_markers) == 3
    assert okr.fixation_markers == []
    assert okr.stimulus_name == "White Dots Up 100% Contrast Left Dot Size Testing"
    assert okr.block_markers[0].event_type == "TrialBlock"
    assert okr.block_markers[0].block_index == 0
    assert okr.block_markers[0].label == "B0↑"
    assert okr.block_markers[0].start_time == pytest.approx(10.883830)
    assert okr.block_markers[0].end_time == pytest.approx(30.888290)
    assert okr.block_markers[0].dot_start_size == pytest.approx(2.8)
    assert okr.block_markers[0].emission_rate == pytest.approx(7500)
    assert okr.block_markers[1].block_index == 1
    assert okr.block_markers[1].dot_start_size == pytest.approx(0.7)
    assert okr.block_markers[1].emission_rate == pytest.approx(10000)

    text = condition_at_time(okr, 20.0)
    assert "Dots → Left eye" in text
    assert "size 2.8" in text
    assert "rate 7500" in text
    assert "Up" in text
    assert "contrast 1" in text

    fields = segment_condition_fields(okr, 40.0, 41.0)
    assert fields["block_label"] == "B1↑"
    assert fields["dot_start_size"] == pytest.approx(0.7)
    assert fields["emission_rate"] == pytest.approx(10000)
    assert fields["event_type"] == "TrialBlock"


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
    assert "Session: Dots → Right eye · Increment · White dots" in text
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
    assert "Dots → Right eye" in text_fix
    assert "Initial fixation" in text_fix


def test_okr_left_right_labels_and_detect_map(tmp_path: Path):
    from slowphase_okr.detect import _okr_direction_to_detect
    from slowphase_okr.okr_log import load_okr_log

    log = tmp_path / "OKR_Log_lr.txt"
    log.write_text(
        "# OKR Condition Log\n"
        "eventIndex\teventType\teyePatch\tcontrastBlockIndex\tdotColor\t"
        "direction\tcontrastLevel\tdotStartSize\temissionRate\tusePersistentDots\t"
        "sessionContrastThreshold\tthresholdMultiplier\tisAnchor100\tstartTime\tendTime\n"
        "1\tContrastBlock\tLeft\t0\tWhite\tLeft\t0.5\t0.7\t20000\t0\t0.1\t2\t0\t10.0\t20.0\n"
        "2\tContrastBlock\tLeft\t1\tWhite\tRight\t0.5\t0.7\t20000\t0\t0.1\t2\t0\t25.0\t35.0\n"
    )
    okr = load_okr_log(log)
    assert okr.block_markers[0].label == "B0←"
    assert okr.block_markers[1].label == "B1→"
    assert _okr_direction_to_detect("Left") == "down"
    assert _okr_direction_to_detect("Right") == "up"


def test_median_gain():
    from slowphase_okr.fit import SegmentFit

    segs = [
        SegmentFit(1, 0, 1, 0, 1, 2, 20, 0, 20 / 31, 0.99, 0.01, True, 31),
        SegmentFit(2, 2, 3, 1, 2, 2, 30, 0, 30 / 31, 0.98, 0.02, True, 31),
    ]
    assert trial_summary_median_gain(segs) == pytest.approx((20 / 31 + 30 / 31) / 2)


def test_segments_meeting_min_r2():
    from slowphase_okr.fit import SegmentFit, segments_meeting_min_r2, trial_summary_gain

    segs = [
        SegmentFit(1, 0, 1, 0.0, 1.0, 2, 10.0, 0.0, 1.0, 0.95, 0.01, True, 10.0),
        SegmentFit(2, 2, 3, 1.0, 2.0, 2, 20.0, 0.0, 2.0, 0.80, 0.02, True, 10.0),
        SegmentFit(3, 4, 5, 2.0, 3.0, 2, 30.0, 0.0, 3.0, float("nan"), 0.03, True, 10.0),
    ]
    assert len(segments_meeting_min_r2(segs, None)) == 3
    kept = segments_meeting_min_r2(segs, 0.9)
    assert [s.segment_id for s in kept] == [1]
    assert trial_summary_gain(kept, "median") == pytest.approx(1.0)
    assert segments_meeting_min_r2(segs, 0.99) == []


def test_summarize_gains_by_block(tmp_path: Path):
    from slowphase_okr.export import export_to_excel
    from slowphase_okr.fit import SegmentFit, summarize_gains_by_block
    from slowphase_okr.okr_log import load_okr_log

    log = tmp_path / "OKR_Log_test.txt"
    log.write_text(
        "# OKR Condition Log\n"
        "# StimulusName: Comb4 Block3 RE Increment White Dots\n"
        "eventIndex\teventType\teyePatch\tcontrastBlockIndex\tdotColor\t"
        "direction\tcontrastLevel\tdotStartSize\temissionRate\tusePersistentDots\t"
        "sessionContrastThreshold\tthresholdMultiplier\tisAnchor100\tstartTime\tendTime\n"
        "2\tContrastBlock\tLeft\t0\tWhite\tUp\t0.5\t0.7\t20000\t0\t0.1\t2\t0\t10.0\t20.0\n"
        "4\tContrastBlock\tLeft\t1\tWhite\tDown\t0.25\t0.7\t20000\t0\t0.1\t4\t0\t30.0\t40.0\n"
    )
    okr = load_okr_log(log)
    segs = [
        SegmentFit(1, 0, 1, 12.0, 13.0, 10, 8.0, 0.0, 0.80, 0.95, 0.05, True, 10.0),
        SegmentFit(2, 2, 3, 14.0, 15.0, 10, 9.0, 0.0, 0.90, 0.96, 0.04, True, 10.0),
        SegmentFit(3, 4, 5, 32.0, 33.0, 10, -7.0, 0.0, -0.70, 0.94, 0.06, False, 10.0),
        SegmentFit(4, 6, 7, 50.0, 51.0, 10, 5.0, 0.0, 0.50, 0.90, 0.08, True, 10.0),
    ]
    summaries = summarize_gains_by_block(segs, okr)
    assert len(summaries) == 3
    by_label = {s.block_label: s for s in summaries}
    assert by_label["B0↑"].n_segments == 2
    assert by_label["B0↑"].median_gain == pytest.approx(0.85)
    assert by_label["B0↑"].direction == "Up"
    assert by_label["B0↑"].contrast_level == pytest.approx(0.5)
    assert by_label["B1↓"].n_segments == 1
    assert by_label["B1↓"].median_gain == pytest.approx(-0.70)
    assert by_label["Outside blocks"].n_segments == 1
    assert "Increment" in by_label["B0↑"].session_tags

    out = export_to_excel(
        segs,
        trial_id="t",
        software_version="0.0",
        output_path=tmp_path / "out.xlsx",
        okr_log=okr,
    )
    import pandas as pd

    by_block = pd.read_excel(out, sheet_name="by_block")
    segments = pd.read_excel(out, sheet_name="segments")
    assert "median_gain" in by_block.columns
    assert "block_label" in segments.columns
    assert "duration_s" in segments.columns
    assert "velocity_deg_s" in segments.columns
    assert segments["duration_s"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert segments["velocity_deg_s"].tolist() == pytest.approx(
        segments["slope_deg_s"].tolist()
    )
    assert len(by_block) == 3


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
        SegmentFit(1, 0, 150, times[0], times[150], 151, 25.0, 0.0, 25 / 31, 0.99, 0.01, True, 31.0),
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


def test_load_sranipal_eye_openness(tmp_path: Path):
    from slowphase_okr.gaze import (
        attach_eye_openness,
        discover_eye_openness_files,
        load_sranipal_eye_openness,
        load_ush2a_trial,
    )

    left_vals = tmp_path / "sranipalLeftEyeOpenness.txt"
    left_times = tmp_path / "sranipalLeftEyeOpennessTimes.txt"
    right_vals = tmp_path / "sranipalRightEyeOpenness.txt"
    right_times = tmp_path / "sranipalRightEyeOpennessTimes.txt"
    left_vals.write_text("1.0\n0.5\n0.2\n")
    left_times.write_text("0.0\n0.1\n0.2\n")
    right_vals.write_text("0.9\n0.8\n0.7\n")
    right_times.write_text("0.0\n0.1\n0.2\n")

    found = discover_eye_openness_files(tmp_path, "left")
    assert found is not None
    left = load_sranipal_eye_openness(found[0], found[1], eye="left")
    assert left.eye == "left"
    assert left.openness[1] == pytest.approx(0.5)
    assert discover_eye_openness_files(tmp_path, "right") is not None

    gaze = tmp_path / "rotatedGaze.txt"
    gaze.write_text("(0,0,1)\n(0,0.1,1)\n(0,0.2,1)\n")
    timef = tmp_path / "gazeTime.txt"
    timef.write_text("0.0\n0.1\n0.2\n")
    trial = load_ush2a_trial(gaze, timef, trial_id="open_test")
    left_tr, right_tr = attach_eye_openness(trial, tmp_path)
    assert left_tr is not None and right_tr is not None
    assert trial.has_eye_openness()
    assert trial.openness_left is not None
    assert trial.openness_right.openness[0] == pytest.approx(0.9)


def test_discover_eye_openness_missing(tmp_path: Path):
    from slowphase_okr.gaze import discover_eye_openness_files

    assert discover_eye_openness_files(tmp_path, "left") is None


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


def test_attach_sranipal_pupils_both_eyes(tmp_path: Path):
    from slowphase_okr.gaze import attach_sranipal_pupils

    gaze = tmp_path / "rotatedGaze.txt"
    gaze.write_text("(0.0, 0.1, 1.0)\n")
    timef = tmp_path / "gazeTime.txt"
    timef.write_text("0.0\n")
    trial = load_ush2a_trial(gaze, timef, trial_id="test RE")
    for eye, prefix in (("left", "sranipalLeft"), ("right", "sranipalRight")):
        (tmp_path / f"{prefix}PupilPositions.txt").write_text("(0.5, 0.5)\n")
        (tmp_path / f"{prefix}PupilPositionTimes.txt").write_text("0.0\n")
    left, right = attach_sranipal_pupils(trial, tmp_path)
    assert left is not None and right is not None
    assert trial.pupil_left is left
    assert trial.pupil_right is right
    assert trial.pupil is right  # RE trial → viewing eye
    assert trial.has_pupil_sensor()


def test_summarize_pupil_sensor_near_edge():
    from slowphase_okr.gaze import PupilTrace, pupil_edge_distance, summarize_pupil_sensor

    pupil = PupilTrace(
        times=np.array([0.0, 0.1, 0.2, 0.3]),
        x=np.array([0.5, 0.05, 0.5, 0.95]),
        y=np.array([0.5, 0.5, 0.5, 0.5]),
        eye="left",
    )
    edge = pupil_edge_distance(pupil.x, pupil.y)
    assert edge[0] == pytest.approx(0.5)
    assert edge[1] == pytest.approx(0.05)
    stats = summarize_pupil_sensor(pupil, margin=0.15)
    assert stats.n_valid == 4
    assert stats.pct_near_edge == pytest.approx(50.0)
    assert stats.mean_x == pytest.approx(0.5)


def test_attach_and_compare_gaze_origins(tmp_path: Path):
    from slowphase_okr.gaze import (
        attach_sranipal_gaze_origins,
        compare_gaze_origins,
        summarize_gaze_origin,
    )

    gaze = tmp_path / "rotatedGaze.txt"
    gaze.write_text("(0.0, 0.1, 1.0)\n")
    timef = tmp_path / "gazeTime.txt"
    timef.write_text("0.0\n")
    trial = load_ush2a_trial(gaze, timef, trial_id="test RE")
    (tmp_path / "sranipalLeftGazeOrigins.txt").write_text(
        "(34.0, 2.0, -38.0)\n(34.1, 2.1, -38.1)\n(34.2, 2.0, -38.0)\n"
    )
    (tmp_path / "sranipalLeftGazeTime.txt").write_text("0.0\n0.01\n0.02\n")
    (tmp_path / "sranipalRightGazeOrigins.txt").write_text(
        "(-28.0, 6.0, -35.0)\n(-28.1, 6.1, -35.1)\n"
    )
    (tmp_path / "sranipalRightGazeTime.txt").write_text("0.0\n0.01\n")
    left, right = attach_sranipal_gaze_origins(trial, tmp_path)
    assert left is not None and right is not None
    assert trial.has_gaze_origins()
    sym = compare_gaze_origins(left, right)
    assert sym.ipd_mm == pytest.approx(62.15, abs=0.2)
    assert sym.delta_y == pytest.approx(-4.0, abs=0.2)
    assert sym.delta_z == pytest.approx(-3.0, abs=0.2)
    left_stats = summarize_gaze_origin(left)
    assert left_stats.jitter_x == pytest.approx(0.1, abs=1e-6)
    assert left_stats.jitter_3d > 0.0


def test_load_heading_trace_stable(tmp_path: Path):
    from slowphase_okr.gaze import (
        attach_heading_trace,
        load_heading_trace,
        summarize_heading_wiggle,
    )

    # Constant Inverse(heading) → recovered heading constant → Δ ≈ 0
    q = "(-0.4, -0.01, -0.002, 0.9)\n"
    rot = tmp_path / "gazeRotations.txt"
    rot.write_text(q * 3)
    timef = tmp_path / "gazeTime.txt"
    timef.write_text("0.0\n0.1\n0.2\n")
    heading = load_heading_trace(rot, timef)
    assert len(heading.times) == 3
    assert heading.roll_deg[0] == pytest.approx(0.0, abs=1e-6)
    assert heading.pitch_deg[0] == pytest.approx(0.0, abs=1e-6)
    assert heading.yaw_deg[0] == pytest.approx(0.0, abs=1e-6)
    assert heading.angle_from_start_deg[0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.abs(heading.angle_from_start_deg) < 1e-6)

    gaze = tmp_path / "rotatedGaze.txt"
    gaze.write_text("(0.0, 0.1, 1.0)\n(0.0, 0.1, 1.0)\n(0.0, 0.1, 1.0)\n")
    trial = load_ush2a_trial(gaze, timef, trial_id="wiggle")
    attached = attach_heading_trace(trial, tmp_path)
    assert attached is not None
    assert trial.has_heading()
    stats = summarize_heading_wiggle(trial.heading)
    assert stats.max_angle_from_start == pytest.approx(0.0, abs=1e-6)
    assert stats.yaw_ptp == pytest.approx(0.0, abs=1e-6)


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
        SegmentFit(1, 0, 1, 0.0, 0.1, 2, 20.0, 0.0, 20.0 / 31.0, 0.99, 0.01, True, 31.0),
    ]
    out = save_autosave(
        tmp_path / "trial_slowphase_okr_markings.json",
        trial_id="trial",
        gaze_source=str(gaze.resolve()),
        time_source=str(timef.resolve()),
        stimulus_velocity=31.0,
        segments=segments,
        software_version="0.2.7",
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


def test_conservative_gain_for_upward_window():
    from slowphase_okr.conservative import compute_conservative_gain_for_window

    times = np.arange(0.0, 1.1, 0.1)
    elevation = 10.0 * times
    elevation[6:] += 10.0  # one large positive saccade frame
    result = compute_conservative_gain_for_window(
        times,
        elevation,
        start_time=0.0,
        end_time=1.0,
        stimulus_velocity_deg_s=20.0,
        saccade_threshold_deg_s=20.0,
        min_slow_frames=1,
    )
    assert result.n_velocity_frames == 10
    assert result.n_slow_frames == 9
    assert result.pct_saccade_frames == pytest.approx(10.0)
    assert result.mean_velocity_deg_s == pytest.approx(10.0)
    assert result.gain == pytest.approx(0.5)
    assert result.gain_sd == pytest.approx(0.0)


def test_conservative_gain_is_direction_matched_for_down_block():
    from slowphase_okr.conservative import compute_conservative_gain_for_window

    times = np.arange(0.0, 1.1, 0.1)
    elevation = -8.0 * times
    result = compute_conservative_gain_for_window(
        times,
        elevation,
        start_time=0.0,
        end_time=1.0,
        stimulus_velocity_deg_s=20.0,
        saccade_threshold_deg_s=20.0,
        direction="Down",
        min_slow_frames=1,
    )
    assert result.mean_velocity_deg_s == pytest.approx(-8.0)
    assert result.gain == pytest.approx(0.4)
    assert result.gain_sd == pytest.approx(0.0)


def test_conservative_gain_sd_from_variable_frame_gains():
    from slowphase_okr.conservative import compute_conservative_gain_for_window

    times = np.arange(0.0, 0.5, 0.1)
    # Frame velocities: 4, 8, 12, 16 deg/s → gains 0.2, 0.4, 0.6, 0.8
    elevation = np.array([0.0, 0.4, 1.2, 2.4, 4.0])
    result = compute_conservative_gain_for_window(
        times,
        elevation,
        start_time=0.0,
        end_time=0.4,
        stimulus_velocity_deg_s=20.0,
        saccade_threshold_deg_s=20.0,
        min_slow_frames=1,
    )
    assert result.gain == pytest.approx(0.5)
    assert result.gain_sd == pytest.approx(np.std([0.2, 0.4, 0.6, 0.8], ddof=1))


def test_conservative_gain_by_okr_block(tmp_path: Path):
    from slowphase_okr.conservative import compute_conservative_gains_by_block

    log = tmp_path / "OKR_Log_blocks.txt"
    log.write_text(
        "eventIndex\teventType\tcontrastBlockIndex\tstartTime\tendTime\tdirection\n"
        "1\tContrastBlock\t0\t0.0\t1.0\tUp\n"
        "2\tContrastBlock\t1\t1.1\t2.0\tDown\n"
    )
    okr = load_okr_log(log)
    times = np.arange(0.0, 2.1, 0.1)
    elevation = np.where(times <= 1.0, 10.0 * times, 10.0 - 5.0 * (times - 1.0))
    results = compute_conservative_gains_by_block(
        times,
        elevation,
        stimulus_velocity_deg_s=20.0,
        saccade_threshold_deg_s=20.0,
        okr_log=okr,
        min_slow_frames=1,
    )
    assert [result.block_label for result in results] == ["B0↑", "B1↓"]
    assert results[0].gain == pytest.approx(0.5)
    assert results[1].gain == pytest.approx(0.25)


def test_sort_conservative_gains_puts_persistent_last():
    from slowphase_okr.conservative import (
        ConservativeBlockGain,
        contrast_plot_x_label,
        sort_conservative_gains_for_contrast_plot,
    )

    def _block(
        label: str,
        contrast: float | None,
        persistent: bool | None,
        gain: float,
        start: float,
    ) -> ConservativeBlockGain:
        return ConservativeBlockGain(
            block_label=label,
            condition=label,
            direction="Up",
            start_time=start,
            end_time=start + 1.0,
            n_velocity_frames=10,
            n_slow_frames=10,
            pct_saccade_frames=0.0,
            mean_velocity_deg_s=10.0,
            gain=gain,
            gain_sd=0.0,
            contrast_level=contrast,
            is_persistent=persistent,
        )

    blocks = [
        _block("P", 1.0, True, 0.9, 4.0),
        _block("C_high", 0.8, False, 0.7, 2.0),
        _block("C_low", 0.2, False, 0.3, 0.0),
        _block("C_mid", 0.5, False, 0.5, 1.0),
    ]
    ordered = sort_conservative_gains_for_contrast_plot(blocks)
    assert [b.block_label for b in ordered] == ["C_low", "C_mid", "C_high", "P"]
    assert contrast_plot_x_label(ordered[0]) == "0.2"
    assert contrast_plot_x_label(ordered[-1]) == "Persistent\n(1)"


def test_conservative_gain_keeps_contrast_and_persistent_from_okr_log(tmp_path: Path):
    from slowphase_okr.conservative import (
        compute_conservative_gains_by_block,
        sort_conservative_gains_for_contrast_plot,
    )

    log = tmp_path / "OKR_Log_contrast.txt"
    log.write_text(
        "eventIndex\teventType\teyePatch\tcontrastBlockIndex\tdotColor\t"
        "direction\tcontrastLevel\tdotStartSize\temissionRate\tusePersistentDots\t"
        "sessionContrastThreshold\tthresholdMultiplier\tisAnchor100\tstartTime\tendTime\n"
        "1\tContrastBlock\tRight\t0\tWhite\tUp\t0.25\t1\t10\t0\t0.1\t1\t0\t0.0\t1.0\n"
        "2\tAnchor100PersistentBlock\tRight\t1\tWhite\tUp\t1.0\t1\t10\t1\t0.1\tNA\t1\t1.1\t2.0\n"
    )
    okr = load_okr_log(log)
    times = np.arange(0.0, 2.1, 0.1)
    elevation = 10.0 * times
    results = compute_conservative_gains_by_block(
        times,
        elevation,
        stimulus_velocity_deg_s=20.0,
        saccade_threshold_deg_s=40.0,
        okr_log=okr,
        min_slow_frames=1,
    )
    assert results[0].contrast_level == pytest.approx(0.25)
    assert results[0].is_persistent is False
    assert results[1].contrast_level == pytest.approx(1.0)
    assert results[1].is_persistent is True
    ordered = sort_conservative_gains_for_contrast_plot(results)
    assert [r.block_label for r in ordered] == [results[0].block_label, results[1].block_label]


def test_mlx_net_displacement_recovers_steady_upward_ramp():
    from slowphase_okr.net_displacement import compute_mlx_net_displacement_for_window

    fs = 100.0
    times = np.arange(0.0, 2.0 + 1e-9, 1.0 / fs)
    elevation = 10.0 * times  # 10 deg/s upward
    azimuth = np.zeros_like(times)
    result = compute_mlx_net_displacement_for_window(
        times,
        elevation,
        azimuth,
        start_time=0.0,
        end_time=2.0,
        max_speed_deg_s=40.0,
        min_samples=20,
        direction="Up",
    )
    assert result.error == ""
    assert result.axis == "elevation"
    assert result.net_disp_deg == pytest.approx(20.0, abs=1.0)
    assert result.norm_disp_deg == pytest.approx(result.net_disp_deg, abs=1e-6)


def test_mlx_net_displacement_zeros_fast_saccades():
    from slowphase_okr.net_displacement import compute_mlx_net_displacement_for_window

    fs = 100.0
    times = np.arange(0.0, 1.0 + 1e-9, 1.0 / fs)
    elevation = 5.0 * times
    # Inject one large jump (~80 deg in one frame → >>40 deg/s)
    elevation = elevation.copy()
    elevation[50:] += 80.0
    azimuth = np.zeros_like(times)
    result = compute_mlx_net_displacement_for_window(
        times,
        elevation,
        azimuth,
        start_time=0.0,
        end_time=1.0,
        max_speed_deg_s=40.0,
        min_samples=20,
        direction="Up",
    )
    assert result.error == ""
    # Without zeroing, net would be ~85°; with saccade zeroing stay near slow ramp (~5°)
    assert result.net_disp_deg == pytest.approx(5.0, abs=2.0)
    assert result.pct_saccade_frames > 0


def test_mlx_net_displacement_by_block_uses_azimuth_for_right(tmp_path: Path):
    from slowphase_okr.net_displacement import compute_mlx_net_displacements_by_block

    log = tmp_path / "OKR_Log_right.txt"
    log.write_text(
        "eventIndex\teventType\tcontrastBlockIndex\tstartTime\tendTime\tdirection\n"
        "1\tContrastBlock\t0\t0.0\t1.0\tRight\n"
    )
    okr = load_okr_log(log)
    times = np.arange(0.0, 1.0 + 1e-9, 0.01)
    elevation = np.zeros_like(times)
    azimuth = 8.0 * times
    results = compute_mlx_net_displacements_by_block(
        times,
        elevation,
        azimuth,
        okr_log=okr,
        max_speed_deg_s=40.0,
        min_samples=20,
    )
    assert len(results) == 1
    assert results[0].axis == "azimuth"
    assert results[0].error == ""
    assert results[0].net_disp_deg == pytest.approx(8.0, abs=1.0)


def test_sort_net_displacements_puts_persistent_last():
    from slowphase_okr.net_displacement import (
        NetDisplacementBlockResult,
        net_disp_contrast_plot_x_label,
        sort_net_displacements_for_contrast_plot,
    )

    def _block(
        label: str,
        contrast: float | None,
        persistent: bool | None,
        start: float,
    ) -> NetDisplacementBlockResult:
        return NetDisplacementBlockResult(
            block_label=label,
            condition=label,
            direction="Up",
            start_time=start,
            end_time=start + 1.0,
            axis="elevation",
            n_samples=100,
            pct_saccade_frames=0.0,
            fraction_failures=0.0,
            net_disp_deg=10.0,
            norm_disp_deg=10.0,
            contrast_level=contrast,
            is_persistent=persistent,
        )

    blocks = [
        _block("P", 1.0, True, 4.0),
        _block("C_high", 0.8, False, 2.0),
        _block("C_low", 0.2, False, 0.0),
        _block("C_mid", 0.5, False, 1.0),
    ]
    ordered = sort_net_displacements_for_contrast_plot(blocks)
    assert [b.block_label for b in ordered] == ["C_low", "C_mid", "C_high", "P"]
    assert net_disp_contrast_plot_x_label(ordered[0]) == "0.2"
    assert net_disp_contrast_plot_x_label(ordered[-1]) == "Persistent\n(1)"

