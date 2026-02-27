"""Calibration flow and baseline storage.

Captures the user's "good posture" baseline over a configurable window (default
5 seconds), averages the landmark positions across all valid frames, and persists
the result to a JSON file. All subsequent posture analysis compares live landmarks
against this personal baseline.

Usage:
    manager = CalibrationManager()
    manager.start()

    # In your frame loop:
    state = manager.add_frame(mediapipe_result.pose_landmarks)
    if state == CalibrationState.COMPLETE:
        manager.save("data/calibration.json")
        baseline = manager.baseline

    # Later, to reload:
    baseline = CalibrationManager.load("data/calibration.json")
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# MediaPipe landmark indices used for posture analysis (Section 3.2)

NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

KEY_LANDMARKS = [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

CAPTURE_DURATION_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Data classes


@dataclass
class LandmarkPoint:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PostureBaseline:
    """Averaged landmark positions representing the user's good-posture reference.

    Derived metrics are precomputed here so the posture analysis engine can
    compare against them without recomputing every frame.
    """

    nose: LandmarkPoint
    left_ear: LandmarkPoint
    right_ear: LandmarkPoint
    left_shoulder: LandmarkPoint
    right_shoulder: LandmarkPoint
    left_hip: LandmarkPoint
    right_hip: LandmarkPoint

    # Precomputed metrics (see Section 3.3 for definitions)
    neck_angle: float          # Forward head angle at right ear-shoulder-vertical (degrees)
    shoulder_y_avg: float      # Average normalised y of both shoulders (slouch reference)
    shoulder_width: float      # Normalised distance between shoulders
    torso_centroid_x: float    # Hip-shoulder midpoint x (background-person anchor, §5.2)
    torso_centroid_y: float    # Hip-shoulder midpoint y

    captured_at: float         # Unix timestamp


# ---------------------------------------------------------------------------
# State machine


class CalibrationState(Enum):
    IDLE = auto()       # Not yet started
    CAPTURING = auto()  # Actively accumulating frames
    COMPLETE = auto()   # Baseline successfully computed
    FAILED = auto()     # Completed window but no usable frames were collected


# ---------------------------------------------------------------------------
# Manager


class CalibrationManager:
    """Orchestrates the calibration capture flow.

    Thread-safety note: add_frame() is expected to be called from a single
    background thread (the capture loop). start()/save()/load() may be called
    from the UI thread before/after capture, but should not overlap with
    add_frame() calls.
    """

    def __init__(
        self,
        capture_duration: float = CAPTURE_DURATION_SECONDS,
        min_visibility: float = 0.5,
    ) -> None:
        self.capture_duration = capture_duration
        self.min_visibility = min_visibility

        self.state: CalibrationState = CalibrationState.IDLE
        self.baseline: Optional[PostureBaseline] = None

        self._frames: list[dict[int, dict]] = []
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API

    def start(self) -> None:
        """Begin (or restart) the calibration capture window."""
        self._frames = []
        self._start_time = time.monotonic()
        self.baseline = None
        self.state = CalibrationState.CAPTURING

    def add_frame(self, landmarks) -> CalibrationState:
        """Feed one MediaPipe pose-landmark result into the capture buffer.

        Args:
            landmarks: The ``pose_landmarks`` attribute of a MediaPipe Pose
                       result (``NormalizedLandmarkList``), or ``None`` if no
                       pose was detected in this frame.

        Returns:
            The current CalibrationState after processing.
        """
        if self.state != CalibrationState.CAPTURING:
            return self.state

        if landmarks is not None and self._frame_is_usable(landmarks):
            self._frames.append(self._extract_frame(landmarks))

        # Check whether the capture window has elapsed
        elapsed = time.monotonic() - self._start_time  # type: ignore[operator]
        if elapsed >= self.capture_duration:
            self._finalize()

        return self.state

    @property
    def progress(self) -> float:
        """Capture progress in [0.0, 1.0].  1.0 once complete."""
        if self.state == CalibrationState.IDLE:
            return 0.0
        if self.state in (CalibrationState.COMPLETE, CalibrationState.FAILED):
            return 1.0
        if self._start_time is None:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        return min(elapsed / self.capture_duration, 1.0)

    # ------------------------------------------------------------------
    # Persistence

    def save(self, path: Path | str) -> None:
        """Persist the calibration baseline to a JSON file.

        Raises RuntimeError if calibration has not completed successfully.
        """
        if self.baseline is None:
            raise RuntimeError("No baseline to save — run calibration first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(_baseline_to_dict(self.baseline), fh, indent=2)

    @staticmethod
    def load(path: Path | str) -> PostureBaseline:
        """Load a calibration baseline from a JSON file produced by save().

        Raises FileNotFoundError if the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        with open(path) as fh:
            data = json.load(fh)
        return _baseline_from_dict(data)

    # ------------------------------------------------------------------
    # Internal helpers

    def _frame_is_usable(self, landmarks) -> bool:
        """Return True only if all key landmarks meet the visibility threshold."""
        return all(
            landmarks.landmark[idx].visibility >= self.min_visibility
            for idx in KEY_LANDMARKS
        )

    def _extract_frame(self, landmarks) -> dict[int, dict]:
        return {
            idx: {
                "x": landmarks.landmark[idx].x,
                "y": landmarks.landmark[idx].y,
                "z": landmarks.landmark[idx].z,
                "visibility": landmarks.landmark[idx].visibility,
            }
            for idx in KEY_LANDMARKS
        }

    def _finalize(self) -> None:
        if not self._frames:
            self.state = CalibrationState.FAILED
            return
        averaged = _average_frames(self._frames)
        self.baseline = _compute_baseline(averaged)
        self.state = CalibrationState.COMPLETE


# ---------------------------------------------------------------------------
# Frame averaging


def _average_frames(frames: list[dict[int, dict]]) -> dict[int, dict]:
    """Return per-landmark averages across all captured frames."""
    n = len(frames)
    result: dict[int, dict] = {}
    for idx in KEY_LANDMARKS:
        result[idx] = {
            field: sum(f[idx][field] for f in frames) / n
            for field in ("x", "y", "z", "visibility")
        }
    return result


# ---------------------------------------------------------------------------
# Baseline computation


def _compute_baseline(avg: dict[int, dict]) -> PostureBaseline:
    def pt(idx: int) -> LandmarkPoint:
        d = avg[idx]
        return LandmarkPoint(d["x"], d["y"], d["z"], d["visibility"])

    l_shoulder = pt(LEFT_SHOULDER)
    r_shoulder = pt(RIGHT_SHOULDER)
    l_hip = pt(LEFT_HIP)
    r_hip = pt(RIGHT_HIP)
    r_ear = pt(RIGHT_EAR)

    return PostureBaseline(
        nose=pt(NOSE),
        left_ear=pt(LEFT_EAR),
        right_ear=r_ear,
        left_shoulder=l_shoulder,
        right_shoulder=r_shoulder,
        left_hip=l_hip,
        right_hip=r_hip,
        neck_angle=_neck_angle(r_ear, r_shoulder),
        shoulder_y_avg=(l_shoulder.y + r_shoulder.y) / 2.0,
        shoulder_width=abs(r_shoulder.x - l_shoulder.x),
        torso_centroid_x=(l_shoulder.x + r_shoulder.x + l_hip.x + r_hip.x) / 4.0,
        torso_centroid_y=(l_shoulder.y + r_shoulder.y + l_hip.y + r_hip.y) / 4.0,
        captured_at=time.time(),
    )


# ---------------------------------------------------------------------------
# Geometry


def _neck_angle(ear: LandmarkPoint, shoulder: LandmarkPoint) -> float:
    """Forward head angle from vertical (degrees).

    Positive = head forward of shoulder. Uses the same formula as
    calculate_neck_inclination() from the design doc (Section 4.1).
    MediaPipe y-coordinates increase downward, so dy is inverted.
    """
    dx = ear.x - shoulder.x
    dy = shoulder.y - ear.y  # inverted: y increases downward in MediaPipe
    return math.degrees(math.atan2(dx, dy))


# ---------------------------------------------------------------------------
# Serialization


def _point_to_dict(pt: LandmarkPoint) -> dict:
    return {"x": pt.x, "y": pt.y, "z": pt.z, "visibility": pt.visibility}


def _point_from_dict(d: dict) -> LandmarkPoint:
    return LandmarkPoint(d["x"], d["y"], d["z"], d["visibility"])


def _baseline_to_dict(b: PostureBaseline) -> dict:
    return {
        "nose": _point_to_dict(b.nose),
        "left_ear": _point_to_dict(b.left_ear),
        "right_ear": _point_to_dict(b.right_ear),
        "left_shoulder": _point_to_dict(b.left_shoulder),
        "right_shoulder": _point_to_dict(b.right_shoulder),
        "left_hip": _point_to_dict(b.left_hip),
        "right_hip": _point_to_dict(b.right_hip),
        "neck_angle": b.neck_angle,
        "shoulder_y_avg": b.shoulder_y_avg,
        "shoulder_width": b.shoulder_width,
        "torso_centroid_x": b.torso_centroid_x,
        "torso_centroid_y": b.torso_centroid_y,
        "captured_at": b.captured_at,
    }


def _baseline_from_dict(d: dict) -> PostureBaseline:
    return PostureBaseline(
        nose=_point_from_dict(d["nose"]),
        left_ear=_point_from_dict(d["left_ear"]),
        right_ear=_point_from_dict(d["right_ear"]),
        left_shoulder=_point_from_dict(d["left_shoulder"]),
        right_shoulder=_point_from_dict(d["right_shoulder"]),
        left_hip=_point_from_dict(d["left_hip"]),
        right_hip=_point_from_dict(d["right_hip"]),
        neck_angle=d["neck_angle"],
        shoulder_y_avg=d["shoulder_y_avg"],
        shoulder_width=d["shoulder_width"],
        torso_centroid_x=d["torso_centroid_x"],
        torso_centroid_y=d["torso_centroid_y"],
        captured_at=d["captured_at"],
    )
