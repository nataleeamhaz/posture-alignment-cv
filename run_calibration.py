"""Live test launcher for the calibration flow.

Runs the CalibrationView widget end-to-end:
  - Opens your webcam
  - Shows the skeleton-only privacy preview
  - Runs the 5-second capture on "Start Calibration"
  - Prints and saves the resulting baseline to data/calibration.json

Usage:
    source .venv/bin/activate
    python run_calibration.py
"""

import json
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow, QStatusBar

from src.calibration import CalibrationManager, PostureBaseline
from src.ui.calibration_view import CalibrationView

BASELINE_PATH = Path("data/calibration.json")


class MainWindow(QMainWindow):
    def __init__(self, manager: CalibrationManager) -> None:
        super().__init__()
        self.setWindowTitle("Posture Calibration — Live Test")

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready. Click 'Start Calibration' to begin.")

        view = CalibrationView(manager, parent=self)
        view.calibration_complete.connect(self._on_complete)
        view.calibration_cancelled.connect(self._on_cancelled)
        self.setCentralWidget(view)
        self.adjustSize()

    def _on_complete(self, baseline: PostureBaseline) -> None:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        manager.save(BASELINE_PATH)

        self._status_bar.showMessage(
            f"Baseline saved to {BASELINE_PATH}  |  "
            f"neck angle: {baseline.neck_angle:.1f}°  |  "
            f"shoulder y: {baseline.shoulder_y_avg:.3f}  |  "
            f"torso centroid: ({baseline.torso_centroid_x:.3f}, {baseline.torso_centroid_y:.3f})"
        )

        print("\n=== Calibration complete ===")
        print(f"Saved to: {BASELINE_PATH}")
        print(f"  neck_angle:        {baseline.neck_angle:.2f}°")
        print(f"  shoulder_y_avg:    {baseline.shoulder_y_avg:.4f}")
        print(f"  shoulder_width:    {baseline.shoulder_width:.4f}")
        print(f"  torso_centroid:    ({baseline.torso_centroid_x:.4f}, {baseline.torso_centroid_y:.4f})")
        print(f"  captured_at:       {baseline.captured_at:.0f}")
        print("\nFull baseline JSON:")
        print(json.dumps(json.loads(BASELINE_PATH.read_text()), indent=2))

    def _on_cancelled(self) -> None:
        print("Calibration cancelled.")
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    manager = CalibrationManager()
    window = MainWindow(manager)
    window.show()
    sys.exit(app.exec())
