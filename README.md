# posture-alignment-cv

> **Work in progress.** This project is under active development and not yet ready for general use.

A desktop app that uses your webcam to monitor your sitting posture in real time. When it detects slouching, forward head posture, or lateral lean, it alerts you with a notification and tells you specifically what to fix. Everything runs locally — no video or data leaves your machine.

---

## How it works

The app captures frames from your webcam at a low rate (2–5 FPS), runs them through [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) to extract body landmarks, and compares those landmarks against a personal baseline you set during a one-time calibration step. If bad posture is sustained for more than ~10–15 seconds, you get a desktop notification with specific guidance.

Key design decisions:
- **Privacy-first** — raw frames are never stored or displayed. The UI shows only a skeleton overlay on a blank background.
- **Low CPU** — MediaPipe Lite model at 2 FPS keeps usage under 10% on modern hardware.
- **Personal baseline** — all analysis is relative to your calibrated "good posture", so camera angle and body type don't matter.

---

## Requirements

- Python 3.11+
- A webcam
- macOS, Windows, or Linux

---

## Setup

### 1. Install Python 3.11+

```bash
# macOS (Homebrew)
brew install python@3.11
```

### 2. Clone the repo and create a virtual environment

```bash
git clone https://github.com/nataleeamhaz/posture-alignment-cv.git
cd posture-alignment-cv
python3.11 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

---

## Status

| Module | Status |
|--------|--------|
| Calibration flow | In progress — [`calibration` branch](../../tree/calibration) |
| Pose estimation | Not started |
| Posture analysis engine | Not started |
| Alert system | Not started |
| Session tracking | Not started |
| UI (system tray + dashboard) | Not started |

See [`POSTURE_DESIGN_DOC.md`](POSTURE_DESIGN_DOC.md) for the full technical design.
