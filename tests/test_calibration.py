"""Unit tests for calibration.py — state machine and persistence."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

# Allow importing src modules without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from calibration import (
    CalibrationBaseline,
    CalibrationSession,
    CalibrationState,
    load_baseline,
    save_baseline,
)
from config import AppConfig
from posture_analyzer import LandmarkPoint, PostureFrame


def _make_config(duration=0.2, min_frames=3) -> AppConfig:
    cfg = AppConfig()
    cfg.calibration_duration_seconds = duration
    cfg.calibration_min_frames = min_frames
    return cfg


def _make_frame(angle_approx=10.0) -> PostureFrame:
    """Create a PostureFrame with an approximate neck angle."""
    import math
    rad = math.radians(angle_approx)
    ear = LandmarkPoint(0.5 + math.sin(rad) * 0.1, 0.0, 1.0)
    shoulder = LandmarkPoint(0.5, math.cos(rad) * 0.1, 1.0)
    return PostureFrame(ear=ear, shoulder=shoulder)


# --------------------------------------------------------------------------- #
# CalibrationSession state machine
# --------------------------------------------------------------------------- #

class TestCalibrationSession:
    def test_initial_state_is_idle(self):
        session = CalibrationSession(_make_config())
        assert session.state == CalibrationState.IDLE

    def test_start_transitions_to_running(self):
        session = CalibrationSession(_make_config())
        session.start()
        assert session.state == CalibrationState.RUNNING

    def test_progress_zero_before_start(self):
        session = CalibrationSession(_make_config())
        assert session.progress() == pytest.approx(0.0)

    def test_progress_increases_over_time(self):
        session = CalibrationSession(_make_config(duration=1.0))
        session.start()
        time.sleep(0.1)
        assert session.progress() > 0.0

    def test_progress_capped_at_one(self):
        session = CalibrationSession(_make_config(duration=0.01))
        session.start()
        time.sleep(0.05)
        assert session.progress() == pytest.approx(1.0)

    def test_is_done_after_duration(self):
        session = CalibrationSession(_make_config(duration=0.05))
        session.start()
        time.sleep(0.1)
        assert session.is_done()

    def test_is_not_done_before_duration(self):
        session = CalibrationSession(_make_config(duration=5.0))
        session.start()
        assert not session.is_done()

    def test_get_baseline_returns_none_before_done(self):
        session = CalibrationSession(_make_config(duration=5.0))
        session.start()
        session.add_frame(_make_frame())
        assert session.get_baseline() is None

    def test_successful_calibration(self):
        session = CalibrationSession(_make_config(duration=0.05, min_frames=2))
        session.start()
        for _ in range(5):
            session.add_frame(_make_frame(10.0))
        time.sleep(0.1)
        baseline = session.get_baseline()
        assert baseline is not None
        assert isinstance(baseline.neck_angle, float)
        assert baseline.frames_count >= 2
        assert session.state == CalibrationState.COMPLETE

    def test_failed_calibration_too_few_frames(self):
        session = CalibrationSession(_make_config(duration=0.05, min_frames=10))
        session.start()
        session.add_frame(_make_frame())  # only 1 frame, need 10
        time.sleep(0.1)
        baseline = session.get_baseline()
        assert baseline is None
        assert session.state == CalibrationState.FAILED

    def test_add_frame_ignored_when_not_running(self):
        session = CalibrationSession(_make_config())
        # Should not raise; frames just ignored
        session.add_frame(_make_frame())


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #

class TestBaselinePersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        baseline = CalibrationBaseline(
            neck_angle=12.5,
            frames_count=15,
            captured_at=1000.0,
        )
        calib_file = tmp_path / "calibration.json"

        with patch("calibration.CALIBRATION_FILE", calib_file):
            save_baseline(baseline)
            loaded = load_baseline()

        assert loaded is not None
        assert loaded.neck_angle == pytest.approx(12.5)
        assert loaded.frames_count == 15
        assert loaded.captured_at == pytest.approx(1000.0)

    def test_load_returns_none_when_file_missing(self, tmp_path):
        calib_file = tmp_path / "nonexistent.json"
        with patch("calibration.CALIBRATION_FILE", calib_file):
            result = load_baseline()
        assert result is None

    def test_load_returns_none_on_corrupt_file(self, tmp_path):
        calib_file = tmp_path / "calibration.json"
        calib_file.write_text("not valid json {{")
        with patch("calibration.CALIBRATION_FILE", calib_file):
            result = load_baseline()
        assert result is None

    def test_average_angle_is_correct(self):
        import math
        session = CalibrationSession(_make_config(duration=0.05, min_frames=1))
        session.start()
        # Add 3 frames with the same known angle
        for _ in range(3):
            session.add_frame(_make_frame(10.0))
        time.sleep(0.1)
        baseline = session.get_baseline()
        assert baseline is not None
        # All frames at ~10 degrees, so average should be ~10
        assert baseline.neck_angle == pytest.approx(10.0, abs=0.5)
