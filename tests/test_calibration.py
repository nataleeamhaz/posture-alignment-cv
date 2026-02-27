"""Unit tests for src/calibration.py.

Tests cover:
- CalibrationManager state machine transitions
- Frame filtering (low-visibility frames are dropped)
- Landmark averaging across captured frames
- Derived metric computation (neck angle, torso centroid, etc.)
- Save / load round-trip via JSON
- Edge case: capture window closes with no usable frames → FAILED state
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.calibration import (
    CAPTURE_DURATION_SECONDS,
    KEY_LANDMARKS,
    LEFT_EAR,
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    CalibrationManager,
    CalibrationState,
    LandmarkPoint,
    PostureBaseline,
    _average_frames,
    _baseline_from_dict,
    _baseline_to_dict,
    _compute_baseline,
    _neck_angle,
)


# ---------------------------------------------------------------------------
# Helpers


def _make_landmark(x=0.5, y=0.5, z=0.0, visibility=0.9):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _make_landmarks(overrides: dict | None = None, visibility=0.9):
    """Build a fake MediaPipe NormalizedLandmarkList with 33 landmarks."""
    lms = [_make_landmark(visibility=visibility) for _ in range(33)]
    if overrides:
        for idx, kwargs in overrides.items():
            lm = _make_landmark(**kwargs)
            lms[idx] = lm
    return SimpleNamespace(landmark=lms)


def _make_manager(**kwargs) -> CalibrationManager:
    return CalibrationManager(**kwargs)


# ---------------------------------------------------------------------------
# State machine


class TestCalibrationState:
    def test_initial_state_is_idle(self):
        mgr = _make_manager()
        assert mgr.state == CalibrationState.IDLE

    def test_start_transitions_to_capturing(self):
        mgr = _make_manager()
        mgr.start()
        assert mgr.state == CalibrationState.CAPTURING

    def test_restart_resets_frames_and_baseline(self):
        mgr = _make_manager()
        mgr.start()
        mgr._frames.append({"dummy": True})
        mgr.start()
        assert mgr._frames == []
        assert mgr.baseline is None
        assert mgr.state == CalibrationState.CAPTURING

    def test_add_frame_before_start_is_noop(self):
        mgr = _make_manager()
        lms = _make_landmarks()
        state = mgr.add_frame(lms)
        assert state == CalibrationState.IDLE
        assert mgr._frames == []

    def test_add_frame_after_complete_is_noop(self):
        mgr = _make_manager(capture_duration=0.0)
        mgr.start()
        mgr.add_frame(_make_landmarks())
        assert mgr.state == CalibrationState.COMPLETE
        prev_baseline = mgr.baseline
        mgr.add_frame(_make_landmarks())
        assert mgr.baseline is prev_baseline  # unchanged


# ---------------------------------------------------------------------------
# Progress


class TestProgress:
    def test_progress_idle_is_zero(self):
        mgr = _make_manager()
        assert mgr.progress == 0.0

    def test_progress_complete_is_one(self):
        mgr = _make_manager(capture_duration=0.0)
        mgr.start()
        mgr.add_frame(_make_landmarks())
        assert mgr.progress == 1.0

    def test_progress_increases_over_time(self):
        mgr = _make_manager(capture_duration=10.0)
        mgr.start()
        # Patch monotonic to simulate 5 seconds elapsed
        with patch("src.calibration.time.monotonic", return_value=mgr._start_time + 5.0):
            assert abs(mgr.progress - 0.5) < 0.01

    def test_progress_capped_at_one(self):
        mgr = _make_manager(capture_duration=1.0)
        mgr.start()
        with patch("src.calibration.time.monotonic", return_value=mgr._start_time + 999.0):
            assert mgr.progress == 1.0


# ---------------------------------------------------------------------------
# Frame filtering


class TestFrameFiltering:
    def test_none_landmark_is_skipped(self):
        mgr = _make_manager()
        mgr.start()
        mgr.add_frame(None)
        assert mgr._frames == []

    def test_low_visibility_frame_is_dropped(self):
        mgr = _make_manager(min_visibility=0.5)
        mgr.start()
        # All key landmarks have visibility 0.3 — below threshold
        lms = _make_landmarks(visibility=0.3)
        mgr.add_frame(lms)
        assert mgr._frames == []

    def test_high_visibility_frame_is_kept(self):
        mgr = _make_manager(min_visibility=0.5)
        mgr.start()
        lms = _make_landmarks(visibility=0.9)
        mgr.add_frame(lms)
        assert len(mgr._frames) == 1

    def test_partial_visibility_failure_drops_frame(self):
        """If just one key landmark is below threshold, the whole frame is dropped."""
        mgr = _make_manager(min_visibility=0.5)
        mgr.start()
        lms = _make_landmarks(visibility=0.9)
        # Override one key landmark to be below threshold
        lms.landmark[NOSE] = _make_landmark(visibility=0.1)
        mgr.add_frame(lms)
        assert mgr._frames == []

    def test_no_usable_frames_results_in_failed(self):
        mgr = _make_manager(capture_duration=0.0, min_visibility=0.9)
        mgr.start()
        mgr.add_frame(_make_landmarks(visibility=0.1))  # dropped
        assert mgr.state == CalibrationState.FAILED
        assert mgr.baseline is None


# ---------------------------------------------------------------------------
# Averaging


class TestAveraging:
    def test_single_frame_averages_to_itself(self):
        frame = {idx: {"x": 0.1 * idx, "y": 0.2 * idx, "z": 0.0, "visibility": 0.9}
                 for idx in KEY_LANDMARKS}
        result = _average_frames([frame])
        for idx in KEY_LANDMARKS:
            assert result[idx]["x"] == pytest.approx(frame[idx]["x"])
            assert result[idx]["y"] == pytest.approx(frame[idx]["y"])

    def test_two_frames_averaged_correctly(self):
        frame_a = {idx: {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 1.0}
                   for idx in KEY_LANDMARKS}
        frame_b = {idx: {"x": 1.0, "y": 1.0, "z": 0.0, "visibility": 1.0}
                   for idx in KEY_LANDMARKS}
        result = _average_frames([frame_a, frame_b])
        for idx in KEY_LANDMARKS:
            assert result[idx]["x"] == pytest.approx(0.5)
            assert result[idx]["y"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Baseline computation


class TestBaselineComputation:
    def _make_avg(self, x=0.5, y=0.5) -> dict:
        return {idx: {"x": x, "y": y, "z": 0.0, "visibility": 0.9}
                for idx in KEY_LANDMARKS}

    def test_returns_posture_baseline(self):
        avg = self._make_avg()
        baseline = _compute_baseline(avg)
        assert isinstance(baseline, PostureBaseline)

    def test_shoulder_y_avg_is_correct(self):
        avg = self._make_avg()
        avg[LEFT_SHOULDER]["y"] = 0.4
        avg[RIGHT_SHOULDER]["y"] = 0.6
        baseline = _compute_baseline(avg)
        assert baseline.shoulder_y_avg == pytest.approx(0.5)

    def test_shoulder_width_is_correct(self):
        avg = self._make_avg()
        avg[LEFT_SHOULDER]["x"] = 0.3
        avg[RIGHT_SHOULDER]["x"] = 0.7
        baseline = _compute_baseline(avg)
        assert baseline.shoulder_width == pytest.approx(0.4)

    def test_torso_centroid_is_midpoint_of_shoulders_and_hips(self):
        avg = self._make_avg()
        avg[LEFT_SHOULDER]["x"] = 0.2
        avg[RIGHT_SHOULDER]["x"] = 0.8
        avg[LEFT_HIP]["x"] = 0.3
        avg[RIGHT_HIP]["x"] = 0.7
        baseline = _compute_baseline(avg)
        assert baseline.torso_centroid_x == pytest.approx(0.5)

    def test_captured_at_is_recent(self):
        avg = self._make_avg()
        before = time.time()
        baseline = _compute_baseline(avg)
        after = time.time()
        assert before <= baseline.captured_at <= after


# ---------------------------------------------------------------------------
# Neck angle


class TestNeckAngle:
    def test_vertical_alignment_is_zero_degrees(self):
        ear = LandmarkPoint(x=0.5, y=0.2, z=0.0, visibility=1.0)
        shoulder = LandmarkPoint(x=0.5, y=0.5, z=0.0, visibility=1.0)
        # Ear directly above shoulder (same x) → 0 degrees forward lean
        angle = _neck_angle(ear, shoulder)
        assert angle == pytest.approx(0.0, abs=1e-6)

    def test_forward_lean_is_positive(self):
        # Ear is shifted forward (larger x in camera frame = forward lean)
        ear = LandmarkPoint(x=0.6, y=0.2, z=0.0, visibility=1.0)
        shoulder = LandmarkPoint(x=0.5, y=0.5, z=0.0, visibility=1.0)
        angle = _neck_angle(ear, shoulder)
        assert angle > 0.0

    def test_known_angle(self):
        # 45-degree forward lean: ear 1 unit right, 1 unit up from shoulder
        ear = LandmarkPoint(x=1.0, y=0.0, z=0.0, visibility=1.0)
        shoulder = LandmarkPoint(x=0.0, y=1.0, z=0.0, visibility=1.0)
        angle = _neck_angle(ear, shoulder)
        assert angle == pytest.approx(45.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Save / load round-trip


class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        mgr = _make_manager(capture_duration=0.0)
        mgr.start()
        mgr.add_frame(_make_landmarks())
        assert mgr.state == CalibrationState.COMPLETE
        out = tmp_path / "calibration.json"
        mgr.save(out)
        assert out.exists()

    def test_load_restores_baseline(self, tmp_path):
        mgr = _make_manager(capture_duration=0.0)
        mgr.start()
        mgr.add_frame(_make_landmarks())
        out = tmp_path / "calibration.json"
        mgr.save(out)

        loaded = CalibrationManager.load(out)
        assert isinstance(loaded, PostureBaseline)
        assert loaded.neck_angle == pytest.approx(mgr.baseline.neck_angle)
        assert loaded.shoulder_y_avg == pytest.approx(mgr.baseline.shoulder_y_avg)
        assert loaded.torso_centroid_x == pytest.approx(mgr.baseline.torso_centroid_x)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CalibrationManager.load(tmp_path / "nonexistent.json")

    def test_save_before_calibration_raises(self, tmp_path):
        mgr = _make_manager()
        with pytest.raises(RuntimeError, match="No baseline"):
            mgr.save(tmp_path / "out.json")

    def test_save_creates_parent_dirs(self, tmp_path):
        mgr = _make_manager(capture_duration=0.0)
        mgr.start()
        mgr.add_frame(_make_landmarks())
        nested = tmp_path / "a" / "b" / "calibration.json"
        mgr.save(nested)
        assert nested.exists()

    def test_serialized_json_has_expected_keys(self, tmp_path):
        mgr = _make_manager(capture_duration=0.0)
        mgr.start()
        mgr.add_frame(_make_landmarks())
        out = tmp_path / "calibration.json"
        mgr.save(out)
        data = json.loads(out.read_text())
        expected_keys = {
            "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip", "neck_angle", "shoulder_y_avg",
            "shoulder_width", "torso_centroid_x", "torso_centroid_y", "captured_at",
        }
        assert expected_keys == set(data.keys())


# ---------------------------------------------------------------------------
# Integration: full capture flow with mocked time


class TestFullCaptureFlow:
    def test_complete_flow_produces_baseline(self):
        mgr = _make_manager(capture_duration=0.0)
        mgr.start()
        lms = _make_landmarks()
        state = mgr.add_frame(lms)
        assert state == CalibrationState.COMPLETE
        assert mgr.baseline is not None

    def test_multiple_frames_are_averaged(self):
        """Baseline should reflect the average across all captured frames."""
        mgr = _make_manager(capture_duration=0.0)
        mgr.start()

        # Feed two frames with different shoulder positions
        lms_a = _make_landmarks(overrides={LEFT_SHOULDER: {"x": 0.3, "y": 0.4, "z": 0.0, "visibility": 0.9}})
        lms_b = _make_landmarks(overrides={LEFT_SHOULDER: {"x": 0.5, "y": 0.6, "z": 0.0, "visibility": 0.9}})

        # Capture window is 0 seconds → finalize on first frame, so use capture_duration > 0
        mgr2 = _make_manager(capture_duration=999.0)
        mgr2.start()
        mgr2.add_frame(lms_a)
        mgr2.add_frame(lms_b)

        # Force finalize
        mgr2._manager_finalize = mgr2._finalize
        mgr2.capture_duration = 0.0
        mgr2.add_frame(_make_landmarks())  # this frame triggers finalize

        assert mgr2.state == CalibrationState.COMPLETE
        # Left shoulder x should be average of 0.3, 0.5, and 0.5 (third frame)
        assert mgr2.baseline.left_shoulder.x == pytest.approx((0.3 + 0.5 + 0.5) / 3, abs=0.01)
