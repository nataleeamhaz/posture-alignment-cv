# Calibration Module

This branch implements the calibration flow for posture-alignment-cv — the step where the user establishes their personal "good posture" baseline that all subsequent analysis is measured against.

---

## What's implemented

### `src/calibration.py`

Core calibration logic with no UI or hardware dependencies.

- **`CalibrationManager`** — orchestrates the capture flow via a state machine: `IDLE → CAPTURING → COMPLETE / FAILED`
  - `start()` — begins the capture window
  - `add_frame(landmarks)` — accepts a MediaPipe pose landmark result each frame; filters out frames where any key landmark falls below the visibility threshold
  - `progress` — float in `[0.0, 1.0]` for driving a progress bar
  - `save(path)` / `load(path)` — persists the baseline to/from JSON
- **`PostureBaseline`** — dataclass holding averaged landmark positions plus precomputed metrics used by the analysis engine:
  - `neck_angle` — forward head angle (ear-shoulder-vertical, degrees)
  - `shoulder_y_avg` — average normalised shoulder y position (slouch reference)
  - `shoulder_width` — normalised distance between shoulders
  - `torso_centroid_x/y` — hip-shoulder midpoint used as the background-person anchor (§5.2 of the design doc)

### `src/ui/calibration_view.py`

PyQt6 widget that guides the user through calibration.

- Opens the webcam and runs MediaPipe Pose at ~5 FPS
- **Never shows raw camera pixels** — only a skeleton overlay drawn on a neutral grey canvas (privacy guarantee per design doc §5.2)
- Progress bar fills over the 5-second capture window
- Emits `calibration_complete(PostureBaseline)` on success and `calibration_cancelled()` if dismissed
- Handles the failure state (poor lighting / camera angle) with a user-facing message

### `tests/test_calibration.py`

32 unit tests covering the pure-logic layer (no webcam or UI required):
- State machine transitions
- Frame filtering (low-visibility frames, `None` poses, partial landmark failures)
- Landmark averaging math
- Derived metric computation (neck angle, torso centroid, shoulder width)
- Save / load JSON round-trip
- `FAILED` path when the capture window closes with no usable frames

### `run_calibration.py`

Live test launcher — runs the full calibration UI end-to-end with your webcam.

---

## Setup

### 1. Install Python 3.11+

```bash
brew install python@3.11
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

Full install (required for live testing):

```bash
pip install mediapipe opencv-python PyQt6 numpy
```

Unit tests only (no webcam or CV deps needed):

```bash
pip install pytest
```

---

## Running

### Unit tests

```bash
source .venv/bin/activate
pytest -v
```

All 32 tests should pass in under a second with no webcam required.

### Live test with webcam

```bash
source .venv/bin/activate
python run_calibration.py
```

A window will open with a skeleton-only preview of your webcam feed. Click **Start Calibration**, sit up straight, and hold still for 5 seconds. On completion:
- Key metrics (neck angle, shoulder position, torso centroid) appear in the status bar
- The full baseline is printed to the terminal
- The baseline is saved to `data/calibration.json`

Click **Recalibrate** to redo it, or **Cancel** to close.

---

## Files added on this branch

```
posture-alignment-cv/
├── src/
│   ├── calibration.py
│   └── ui/
│       └── calibration_view.py
├── tests/
│   └── test_calibration.py
├── run_calibration.py
├── requirements.txt
├── pyproject.toml
└── .gitignore
```
