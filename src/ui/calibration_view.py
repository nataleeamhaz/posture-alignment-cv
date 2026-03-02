"""Calibration UI — live camera feed with pose landmark overlay.

Shows the actual camera frame so the user can confirm they are in frame and
that the skeleton is tracking correctly before starting calibration.

Emits:
    calibration_complete(PostureBaseline) — baseline captured and ready.
    calibration_cancelled()              — user dismissed without calibrating.
"""

from __future__ import annotations

import math

import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.calibration import (
    CalibrationManager,
    CalibrationState,
    LANDMARK_GROUPS,
    NOSE,
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    check_landmark_groups,
)

# ---------------------------------------------------------------------------
# MediaPipe helpers

_mp_pose = mp.solutions.pose
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles

# ---------------------------------------------------------------------------
# Constants

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480
FRAME_INTERVAL_MS = 33     # ~30 FPS

# BGR colors and labels for the posture-critical landmarks
_KEY_LANDMARK_STYLE: dict[int, tuple[tuple[int, int, int], str]] = {
    NOSE:           ((0, 255, 255), "Nose"),
    LEFT_EAR:       ((0, 165, 255), "L. Ear"),
    RIGHT_EAR:      ((0, 165, 255), "R. Ear"),
    LEFT_SHOULDER:  ((0, 255, 0),   "L. Shoulder"),
    RIGHT_SHOULDER: ((0, 255, 0),   "R. Shoulder"),
    LEFT_HIP:       ((255, 0, 255), "L. Hip"),
    RIGHT_HIP:      ((255, 0, 255), "R. Hip"),
}

# Positioning tips keyed by the frozenset of group names that are missing
_POSITIONING_TIPS: dict[frozenset, str] = {
    frozenset(["Head"]):
        "Make sure your face is well-lit and facing the camera directly.",
    frozenset(["Shoulders"]):
        "Move back from the camera so your shoulders are fully in frame.",
    frozenset(["Hips"]):
        "Move further back so your hips are visible.",
    frozenset(["Shoulders", "Hips"]):
        "Move back so both your shoulders and hips are in frame.",
    frozenset(["Head", "Hips"]):
        "Face the camera directly and move back so your hips are visible.",
    frozenset(["Head", "Shoulders"]):
        "Improve the lighting and move back so your shoulders are in frame.",
    frozenset(["Head", "Shoulders", "Hips"]):
        "Stand in front of the camera in a well-lit area.",
}

# Qt stylesheet strings for readiness-label states
_STYLE_MISSING = (
    "padding: 6px 14px; border-radius: 4px;"
    "background-color: #f7c5c5; color: #721515;"
)
_STYLE_DETECTED = (
    "padding: 6px 14px; border-radius: 4px;"
    "background-color: #c8f7c5; color: #155724; font-weight: bold;"
)
_STYLE_SAVED = (
    "padding: 6px 14px; border-radius: 4px;"
    "background-color: #c8f7c5; color: #155724; font-weight: bold;"
)


# ---------------------------------------------------------------------------
# Widget


class CalibrationView(QWidget):
    """Guides the user through the calibration flow.

    Instantiate with a CalibrationManager and connect to the two signals:

        view = CalibrationView(manager)
        view.calibration_complete.connect(on_baseline_ready)
        view.calibration_cancelled.connect(on_cancel)
        view.show()
    """

    calibration_complete = pyqtSignal(object)   # PostureBaseline
    calibration_cancelled = pyqtSignal()

    def __init__(self, manager: CalibrationManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._cap: cv2.VideoCapture | None = None
        self._pose = _mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._process_frame)
        self._last_landmark_groups: dict[str, bool] = {g: False for g in LANDMARK_GROUPS}
        self._readiness_labels: dict[str, QLabel] = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        self._instruction_label = QLabel(
            "Position yourself so your <b>head</b>, <b>shoulders</b>, and <b>hips</b> "
            "are all visible in the camera, then click <b>Start Calibration</b>."
        )
        self._instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._instruction_label.setWordWrap(True)
        layout.addWidget(self._instruction_label)

        self._preview = QLabel()
        self._preview.setFixedSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self._preview.setStyleSheet("background-color: black;")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._preview, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Live readiness checklist — one pill label per landmark group
        self._readiness_row = QWidget()
        readiness_layout = QHBoxLayout(self._readiness_row)
        readiness_layout.setContentsMargins(0, 0, 0, 0)
        readiness_layout.setSpacing(8)
        for group in LANDMARK_GROUPS:
            lbl = QLabel(f"  {group}: detecting…  ")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(_STYLE_MISSING)
            self._readiness_labels[group] = lbl
            readiness_layout.addWidget(lbl)
        layout.addWidget(self._readiness_row, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Positioning tip — updated live in IDLE/FAILED states
        self._tip_label = QLabel("")
        self._tip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tip_label.setWordWrap(True)
        layout.addWidget(self._tip_label)

        # Countdown shown during capture
        self._countdown_label = QLabel("")
        self._countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._countdown_label.hide()
        layout.addWidget(self._countdown_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Capturing… %p%")
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._start_btn = QPushButton("Start Calibration")
        self._start_btn.setEnabled(False)   # enabled once all groups are visible
        self._start_btn.clicked.connect(self._on_start)
        layout.addWidget(self._start_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._cancel_btn)

    # ------------------------------------------------------------------
    # Lifecycle — open/close camera with the widget's visibility

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._open_camera()
        self._timer.start(FRAME_INTERVAL_MS)

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self._timer.stop()
        self._close_camera()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        self._close_camera()
        self._pose.close()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Camera helpers

    def _open_camera(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)
            if not self._cap.isOpened():
                self._status_label.setText(
                    "Camera not available. Grant camera access to Terminal in\n"
                    "System Settings → Privacy & Security → Camera, then relaunch."
                )
                self._start_btn.setEnabled(False)

    def _close_camera(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    # ------------------------------------------------------------------
    # Frame loop

    def _process_frame(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            return
        ret, frame = self._cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        canvas = _render_frame(frame, result)
        self._update_preview(canvas)

        state = self._manager.state

        if state in (CalibrationState.IDLE, CalibrationState.FAILED):
            groups = check_landmark_groups(result.pose_landmarks)
            self._last_landmark_groups = groups
            self._update_readiness(groups)
            all_ready = all(groups.values())
            self._start_btn.setEnabled(all_ready)
            if all_ready:
                self._tip_label.setText(
                    "All body parts detected. Sit up straight, then click Start Calibration."
                )
            else:
                missing = frozenset(g for g, ok in groups.items() if not ok)
                self._tip_label.setText(
                    _POSITIONING_TIPS.get(
                        missing, "Adjust your position so all body parts are visible."
                    )
                )

        elif state == CalibrationState.CAPTURING:
            self._last_landmark_groups = check_landmark_groups(result.pose_landmarks)
            self._manager.add_frame(result.pose_landmarks)
            self._progress_bar.setValue(int(self._manager.progress * 100))
            remaining = math.ceil(
                (1.0 - self._manager.progress) * self._manager.capture_duration
            )
            self._countdown_label.setText(f"Hold still… {max(remaining, 1)}s remaining")

            if self._manager.state == CalibrationState.COMPLETE:
                self._on_capture_complete()
            elif self._manager.state == CalibrationState.FAILED:
                self._on_capture_failed()

    def _update_readiness(self, groups: dict[str, bool]) -> None:
        """Refresh the pill labels to reflect current detection state."""
        for group, visible in groups.items():
            lbl = self._readiness_labels[group]
            if visible:
                lbl.setText(f"  {group}: Detected  ")
                lbl.setStyleSheet(_STYLE_DETECTED)
            else:
                lbl.setText(f"  {group}: Not visible  ")
                lbl.setStyleSheet(_STYLE_MISSING)

    def _update_preview(self, canvas: np.ndarray) -> None:
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._preview.setPixmap(QPixmap.fromImage(qimg))

    # ------------------------------------------------------------------
    # Button / event handlers

    def _on_start(self) -> None:
        self._manager.start()
        self._start_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._readiness_row.hide()
        self._tip_label.hide()
        self._status_label.setText("")
        self._countdown_label.setText(
            f"Hold still… {int(self._manager.capture_duration)}s remaining"
        )
        self._countdown_label.show()
        self._instruction_label.setText(
            "Hold your best sitting posture and stay still while we capture your baseline."
        )

    def _on_cancel(self) -> None:
        self._timer.stop()
        self._close_camera()
        self.calibration_cancelled.emit()

    def _on_capture_complete(self) -> None:
        self._progress_bar.setValue(100)
        self._countdown_label.hide()
        self._status_label.setText("Calibration complete!")
        self._start_btn.setText("Recalibrate")
        self._start_btn.setEnabled(True)
        # Show per-group confirmation in the readiness row
        for group in LANDMARK_GROUPS:
            lbl = self._readiness_labels[group]
            lbl.setText(f"  {group}: Saved  ")
            lbl.setStyleSheet(_STYLE_SAVED)
        self._readiness_row.show()
        self._tip_label.hide()
        self._instruction_label.setText(
            "Your baseline has been saved. Monitoring will now use this as your reference posture."
        )
        self.calibration_complete.emit(self._manager.baseline)

    def _on_capture_failed(self) -> None:
        self._progress_bar.hide()
        self._countdown_label.hide()
        # Diagnose which groups had visibility problems
        missing = [g for g, ok in self._last_landmark_groups.items() if not ok]
        if missing:
            missing_set = frozenset(missing)
            tip = _POSITIONING_TIPS.get(
                missing_set, "Adjust your position so all body parts are visible."
            )
            self._status_label.setText(
                f"Could not detect: {', '.join(missing)}.\n{tip}"
            )
        else:
            self._status_label.setText(
                "Calibration failed — couldn't detect your posture reliably.\n"
                "Try improving the lighting or adjusting your camera angle."
            )
        # Show readiness row so live feedback resumes
        self._readiness_row.show()
        self._tip_label.show()
        self._start_btn.setEnabled(all(self._last_landmark_groups.values()))
        self._instruction_label.setText("Adjust your position and try again.")


# ---------------------------------------------------------------------------
# Frame rendering (module-level so it can be tested independently)


def _render_frame(frame: np.ndarray, result) -> np.ndarray:
    """Draw the camera feed with pose skeleton and highlighted posture landmarks.

    Args:
        frame:  Raw BGR frame from the webcam.
        result: A MediaPipe Pose process() result.

    Returns:
        A (PREVIEW_HEIGHT, PREVIEW_WIDTH, 3) uint8 BGR array ready to display.
    """
    canvas = frame.copy()

    if not result.pose_landmarks:
        return canvas

    # Draw all connections and generic landmarks subtly
    _mp_drawing.draw_landmarks(
        canvas,
        result.pose_landmarks,
        _mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=_mp_drawing.DrawingSpec(
            color=(200, 200, 200), thickness=1, circle_radius=2
        ),
        connection_drawing_spec=_mp_drawing.DrawingSpec(
            color=(180, 180, 180), thickness=1
        ),
    )

    # Highlight the posture-critical landmarks with colored dots and labels
    lm = result.pose_landmarks.landmark
    for idx, (color, label) in _KEY_LANDMARK_STYLE.items():
        pt = lm[idx]
        if pt.visibility < 0.5:
            continue
        cx = int(pt.x * PREVIEW_WIDTH)
        cy = int(pt.y * PREVIEW_HEIGHT)
        cv2.circle(canvas, (cx, cy), 8, color, -1)
        cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)  # white outline
        cv2.putText(
            canvas,
            label,
            (cx + 10, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas
