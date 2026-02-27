# Posture Correction Tool - Design Document

## 1. Project Overview

### 1.1 Purpose
A desktop application that uses the computer's webcam to continuously monitor a user's sitting posture while they work. When poor posture is detected, the app alerts the user with visual and audio notifications and provides actionable guidance on how to correct it.

### 1.2 Problem Statement
Prolonged poor posture while working at a desk leads to chronic neck pain, back pain, shoulder tension, and long-term spinal issues. Most people don't notice they're slouching until discomfort sets in. A real-time monitoring tool can catch bad habits early and build healthier posture over time.

### 1.3 Target Users
- Office workers and remote employees who sit at a desk for extended periods
- Students studying or attending online classes
- Developers, designers, and anyone doing long sessions at a computer
- People recovering from back/neck injuries who need posture awareness

### 1.4 Core Value Proposition
- **Real-time monitoring** - Continuous posture tracking via webcam
- **Privacy-first** - All processing happens locally; no video data leaves the machine
- **Actionable corrections** - Not just "fix your posture" but specific guidance (e.g., "raise your chin," "pull shoulders back")
- **Non-intrusive** - Alerts that inform without disrupting deep work

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Desktop Application                    │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  Webcam   │──▶│ Pose         │──▶│ Posture        │  │
│  │  Capture  │   │ Estimation   │   │ Analysis       │  │
│  │  Module   │   │ (MediaPipe)  │   │ Engine         │  │
│  └──────────┘   └──────────────┘   └───────┬────────┘  │
│                                             │           │
│                                    ┌────────▼────────┐  │
│  ┌──────────┐   ┌──────────────┐  │ Alert &         │  │
│  │  Stats   │◀──│ Session      │◀─│ Notification    │  │
│  │  Dashboard│   │ Tracker      │  │ Manager         │  │
│  └──────────┘   └──────────────┘  └─────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

**Application Framework:**
- Language: Python 3.11+
- Desktop UI: PyQt6
- Alternative: Tauri (Rust + web frontend) for a lightweight native app

**Computer Vision / Pose Estimation:**
- Primary: MediaPipe Pose (Google) - lightweight, runs on CPU, 33 body landmarks
- Alternative: OpenPose or MoveNet (TensorFlow)
- Webcam capture: OpenCV (`cv2.VideoCapture`)

**Audio Notifications:**
- Library: `pygame.mixer` or `playsound` for audio alerts
- Text-to-speech (optional): `pyttsx3` for spoken correction guidance

**Data Storage (local):**
- SQLite for session history and posture statistics
- JSON config file for user preferences and calibration data

**Packaging & Distribution:**
- PyInstaller or cx_Freeze for standalone executables
- Platform support: macOS, Windows, Linux

---

## 3. Core Features

### 3.1 Webcam Capture Module

**Responsibilities:**
- Access the default webcam (or user-selected camera)
- Capture frames at a configurable rate (default: 2-5 FPS for posture analysis — no need for full 30 FPS)
- Handle camera permissions, disconnection, and reconnection gracefully

**Key Considerations:**
- Low frame rate to minimize CPU usage (posture doesn't change frame-to-frame)
- Downscale frames before processing (e.g., 640x480) to reduce computation
- Provide a camera preview toggle so users can see what the app sees during calibration

### 3.2 Pose Estimation

**Model: MediaPipe Pose**

MediaPipe Pose provides 33 body landmarks in real-time on CPU. The key landmarks for seated posture analysis:

| Landmark | Index | Use |
|----------|-------|-----|
| Nose | 0 | Head position / forward lean |
| Left/Right Ear | 7, 8 | Head tilt detection |
| Left/Right Shoulder | 11, 12 | Shoulder alignment, slouching |
| Left/Right Hip | 23, 24 | Torso lean, base reference |
| Left/Right Elbow | 13, 14 | Arm position (optional) |
| Left/Right Eye | 2, 5 | Head rotation (supplementary) |

**Processing Pipeline:**
```
Webcam Frame
    │
    ▼
Resize to 640x480
    │
    ▼
MediaPipe Pose Detection
    │
    ▼
Extract Key Landmarks (shoulders, ears, nose, hips)
    │
    ▼
Calculate Posture Metrics (angles, distances, ratios)
    │
    ▼
Compare Against Calibrated Baseline
    │
    ▼
Classify: Good / Needs Correction / Poor
```

### 3.3 Posture Analysis Engine

The core logic that determines whether posture is good or bad based on landmark positions.

**Metrics Tracked:**

1. **Forward Head Posture (Neck Angle)**
   - Measure the angle between the ear, shoulder, and hip (sagittal plane)
   - Good posture: ear roughly aligned above shoulder (~0-10 degree forward lean)
   - Bad posture: ear significantly forward of shoulder (>15-20 degrees)

2. **Shoulder Slouch (Vertical Drop)**
   - Track the vertical position of shoulders relative to the calibrated baseline
   - Detect when shoulders drop (slouching) or rise (tension)

3. **Shoulder Asymmetry (Lateral Tilt)**
   - Compare left vs. right shoulder height
   - Detect lateral leaning or one-sided slouching

4. **Head Tilt**
   - Compare left and right ear positions
   - Detect head tilting to one side (common when resting head on hand)

5. **Torso Lean**
   - Angle of the shoulder-hip line relative to vertical
   - Detect excessive forward or lateral leaning

**Calibration Process:**
1. On first launch (or when user clicks "Calibrate"), prompt user to sit in their best posture
2. Capture baseline measurements for 5 seconds
3. Average the landmark positions to establish "good posture" reference
4. All subsequent analysis compares against this personal baseline
5. Allow re-calibration at any time

**Classification Logic:**
```python
def classify_posture(current_metrics, baseline_metrics, thresholds):
    issues = []

    # Forward head
    neck_angle_delta = current_metrics.neck_angle - baseline_metrics.neck_angle
    if neck_angle_delta > thresholds.neck_warning:      # e.g., 15 degrees
        issues.append(PostureIssue.FORWARD_HEAD)

    # Shoulder slouch
    shoulder_drop = baseline_metrics.shoulder_y - current_metrics.shoulder_y
    if shoulder_drop > thresholds.slouch_warning:        # e.g., 30px relative
        issues.append(PostureIssue.SLOUCHING)

    # Shoulder asymmetry
    shoulder_diff = abs(current_metrics.left_shoulder_y - current_metrics.right_shoulder_y)
    if shoulder_diff > thresholds.asymmetry_warning:     # e.g., 20px
        issues.append(PostureIssue.LATERAL_LEAN)

    # Head tilt
    ear_diff = abs(current_metrics.left_ear_y - current_metrics.right_ear_y)
    if ear_diff > thresholds.tilt_warning:               # e.g., 15px
        issues.append(PostureIssue.HEAD_TILT)

    if not issues:
        return PostureStatus.GOOD, []
    elif len(issues) <= 1:
        return PostureStatus.NEEDS_CORRECTION, issues
    else:
        return PostureStatus.POOR, issues
```

**Temporal Smoothing:**
- Don't alert on momentary shifts (reaching for coffee, turning to talk)
- Use a rolling window (e.g., 10-15 seconds) and only trigger alerts when bad posture is sustained
- Require bad posture in >70% of frames within the window before alerting

### 3.4 Alert & Notification System

**Visual Alerts:**
- System tray icon changes color: green (good) / yellow (warning) / red (poor)
- Pop-up notification with specific correction advice
- Optional: small floating overlay widget showing posture status

**Audio Alerts:**
- Gentle chime or tone when bad posture is first detected
- Escalating: soft chime at first, more noticeable sound if uncorrected after 30-60 seconds
- Optional voice announcement: "Try pulling your shoulders back" (text-to-speech)

**Correction Guidance Messages:**

| Issue | Notification Text |
|-------|------------------|
| Forward head | "Your head is leaning forward. Try tucking your chin and aligning your ears over your shoulders." |
| Slouching | "You're slouching. Sit up tall and imagine a string pulling the top of your head toward the ceiling." |
| Lateral lean | "You're leaning to one side. Center your weight evenly on both sit bones." |
| Head tilt | "Your head is tilted. Level your head so both ears are at the same height." |
| Shoulder tension | "Your shoulders are raised. Drop them down and back, away from your ears." |

**Alert Cooldown:**
- After an alert is shown, wait a configurable cooldown period (default: 5 minutes) before alerting for the same issue again
- Prevents notification fatigue
- Reset cooldown if posture is corrected and then deteriorates again

### 3.5 Session Tracking & Statistics

**Per-Session Data:**
- Session start/end time
- Total time in good vs. bad posture (percentage)
- Number of alerts triggered
- Most common posture issues
- Posture score (0-100, weighted by time in good posture)

**Historical Data:**
- Daily/weekly/monthly posture trends
- Improvement tracking over time
- Streak tracking (consecutive days with >80% good posture)

**Storage Schema (SQLite):**
```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    good_posture_pct REAL,
    alerts_count INTEGER,
    posture_score INTEGER
);

CREATE TABLE posture_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES sessions(id),
    timestamp DATETIME NOT NULL,
    status TEXT NOT NULL,           -- 'good', 'warning', 'poor'
    issues TEXT,                    -- JSON array: ["forward_head", "slouching"]
    neck_angle REAL,
    shoulder_drop REAL,
    shoulder_asymmetry REAL,
    head_tilt REAL
);

CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES sessions(id),
    timestamp DATETIME NOT NULL,
    issue_type TEXT NOT NULL,
    message TEXT NOT NULL,
    corrected_within_seconds INTEGER  -- NULL if not corrected
);
```

### 3.6 User Interface

**System Tray Application:**
- Runs primarily in the system tray (menu bar on macOS)
- Tray icon color indicates current posture status
- Right-click menu: Start/Stop Monitoring, Calibrate, Dashboard, Settings, Quit

**Main Window (Dashboard):**
- Current session summary (time monitored, posture score, alerts today)
- Real-time posture status indicator
- Historical charts (posture score over time, issue frequency breakdown)
- Calibration button with camera preview

**Settings Panel:**
- Alert preferences (sound on/off, volume, cooldown interval)
- Sensitivity thresholds (strict / normal / relaxed)
- Camera selection (if multiple webcams)
- Startup behavior (launch on login, start monitoring automatically)
- Break reminders (optional: remind to stand/stretch every N minutes)

---

## 4. Posture Detection Details

### 4.1 Coordinate System & Angle Calculations

MediaPipe returns normalized coordinates (0-1 range). For posture analysis:

```python
import math

def calculate_angle(a, b, c):
    """Calculate angle at point b given three landmark points."""
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(math.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

def calculate_neck_inclination(ear, shoulder):
    """Calculate forward head angle from vertical."""
    dx = ear.x - shoulder.x
    dy = shoulder.y - ear.y  # inverted because y increases downward
    angle = math.degrees(math.atan2(dx, dy))
    return angle
```

### 4.2 Handling Edge Cases

- **User leaves frame:** Detect when key landmarks have low confidence or are missing. Trigger smart pause (see Section 5.3) — do not alert.
- **Multiple people in frame:** MediaPipe returns the most prominent person, but the app must detect and handle background presence (see Section 5.2).
- **Poor lighting:** Monitor landmark detection confidence. If consistently low, notify user to adjust lighting or camera angle.
- **Camera at an angle:** Calibration accounts for camera position since everything is measured relative to the user's personal baseline.

---

## 5. Privacy & Security

### 5.1 Privacy Guarantees
- **No video storage:** Frames are processed in memory and immediately discarded. No images are saved to disk.
- **No network access:** The app makes zero network calls. All pose estimation runs locally via MediaPipe.
- **No telemetry:** No usage data, analytics, or crash reports are sent anywhere.
- **Only derived metrics are stored:** The SQLite database contains only numerical posture metrics (angles, scores), never images or video.

### 5.2 Background Person Detection & Privacy

When other people enter the camera's field of view (e.g., a coworker walking by, a family member in the background), the app must not analyze or store any information about them.

**Detection Strategy:**

MediaPipe Pose returns a single "primary" pose (most prominent person). However, multiple people can sometimes be partially detected. The app uses a **proximity heuristic** to filter non-primary persons: the calibrated user's hip-shoulder midpoint is anchored during calibration. Any detected pose whose torso centroid deviates significantly from that anchor is treated as a background person.

```
On each frame:
  1. Run MediaPipe Pose
  2. Compute torso centroid of detected pose
  3. Compare against anchored user centroid from calibration
  4. If deviation > background_threshold (e.g., 25% of frame width):
       → Discard frame silently — no posture event recorded
  5. If deviation is within range:
       → Proceed with normal posture analysis
```

**Auto-Pause on Crowd Detection:**

If the scene changes dramatically (e.g., a meeting starts and multiple people fill the frame), confidence scores for MediaPipe landmarks tend to drop or fluctuate. If average landmark visibility drops below 0.6 for 3+ consecutive seconds, the app enters a **Privacy Pause** — it stops all analysis and shows a subtle "monitoring paused — too many people in frame" indicator. It resumes automatically once the scene clears.

**Privacy Mode (Camera Preview):**

During calibration and in the settings panel, the user can enable a live camera preview to adjust their position. This preview **never shows raw video**. It renders only the skeleton overlay (MediaPipe landmark connections) on a neutral grey background — no pixel data from the webcam is ever shown in the UI.

```
Privacy Preview Rendering:
  Raw frame → MediaPipe → Extract landmarks only
  Draw skeleton on blank canvas (no background pixels)
  Display skeleton-only canvas in UI
```

**Settings Options:**
- `background_detection_sensitivity`: Low / Medium (default) / High — controls how aggressively the app pauses when background motion is detected
- `privacy_preview_only`: Always show skeleton-only preview (on by default)
- `auto_pause_on_crowd`: Toggle automatic Privacy Pause (on by default)

### 5.3 Smart Monitoring Pause (User Absence)

Users frequently step away momentarily — to grab coffee, take a phone call, or move to another room. The app must handle these absences gracefully: no false alerts, no lost calibration, and automatic resumption.

**Absence Detection:**

The app continuously monitors landmark visibility. When the primary user's shoulders and/or head landmarks fall below a visibility threshold, a timer starts.

```
Absence State Machine:

  [MONITORING]
      │
      │ landmarks missing or low confidence
      ▼
  [GRACE_PERIOD] ── countdown: 8 seconds ──────────────────────┐
      │                                                         │
      │ user returns (landmarks restored)                       │ grace period expires
      ▼                                                         ▼
  [MONITORING]                                           [AWAY_MODE]
                                                               │
                                                               │ landmarks restored
                                                               ▼
                                                        [RESUMING]
                                                               │
                                                               │ confirm stable detection (3s)
                                                               ▼
                                                        [MONITORING]
```

**Grace Period (8 seconds):** Brief absences (leaning to pick something up, turning to talk to someone) are absorbed silently. No state change is logged, no notification is shown.

**Away Mode:** After the grace period, the app enters Away Mode:
- Pose estimation drops to 0.5 FPS (from default 2-5 FPS) to save CPU
- Posture alerts are fully suppressed
- Session timer pauses — away time does not count against the posture score
- Tray icon shows a grey "paused" state instead of green/yellow/red

**Return & Re-verification:** When landmarks are detected again after Away Mode:
- A 3-second re-verification window confirms the user is stably back
- If the user was away for **< 10 minutes**: resume immediately with existing calibration baseline
- If the user was away for **≥ 10 minutes**: prompt with "Welcome back — recalibrate?" because the user may have returned to a different chair position or adjusted their setup

**Manual Quick-Pause:**
- Global keyboard shortcut (default: `Ctrl+Shift+P`) to manually trigger Away Mode instantly
- Useful when stepping away intentionally (the user knows they won't be back for a while)
- System tray right-click → "Pause monitoring" accomplishes the same

**Configurable Thresholds:**
```json
{
  "grace_period_seconds": 8,
  "long_absence_minutes": 10,
  "away_mode_fps": 0.5,
  "landmark_visibility_threshold": 0.5,
  "return_verification_seconds": 3
}
```

### 5.4 Camera Access
- Request camera permission through OS-level prompts (macOS, Windows)
- Clearly explain why camera access is needed on first launch
- Provide visual indicator (LED-style dot in UI) when camera is active
- Allow user to pause/stop monitoring at any time with one click

---

## 6. Performance Requirements

### 6.1 Resource Usage Targets
- **CPU usage:** <10% average on a modern machine (M1/i5 or better)
- **Memory:** <200MB RAM
- **Frame processing:** 2-5 FPS is sufficient (configurable)
- **Alert latency:** <500ms from posture change detection to notification

### 6.2 Optimization Strategies
- Process frames at 2 FPS by default (posture changes slowly)
- Use MediaPipe's "lite" model variant for lower CPU usage
- Skip frames when system is under load (degrade gracefully)
- Run pose estimation in a background thread to keep UI responsive
- Batch SQLite writes (write posture events every 30 seconds, not every frame)

---

## 7. MVP Scope

### 7.1 Phase 1 - MVP

**Must Have:**
- Webcam capture and MediaPipe pose estimation
- Calibration flow (sit up straight, capture baseline)
- Forward head and slouch detection
- System tray icon with color status (green/yellow/red)
- Desktop notification with correction guidance text
<<<<<<< HEAD
=======
- Audio chime on bad posture detection
>>>>>>> 1d246479bdc8a5ed84841206ec71d3036f51546b
- Temporal smoothing (don't alert on momentary movements)
- Alert cooldown to prevent notification fatigue
- Basic settings (sensitivity, sound on/off)
- Start/stop monitoring

**Not in MVP:**
- Session history and statistics dashboard
- Break reminders
<<<<<<< HEAD
- Audio chime on bad posture detection
=======
>>>>>>> 1d246479bdc8a5ed84841206ec71d3036f51546b
- Posture score and streak tracking
- Multiple camera support
- Custom notification sounds

### 7.2 Phase 2 - Post-MVP Enhancements

- Session tracking with SQLite persistence
- Dashboard with posture score trends and charts
- Break/stretch reminders with timer
- Detailed analytics (most common issues, time-of-day patterns)
- Export posture reports (CSV/PDF)
- Custom alert sounds and notification styles
- Auto-start on login
- Keyboard shortcut to pause/resume

### 7.3 Phase 3 - Future Ideas

- Stretch/exercise suggestions tailored to detected issues
- Integration with health apps (Apple Health, Google Fit)
- Multi-monitor awareness (detect which screen user is looking at)
- Ergonomic setup advisor (camera angle analysis for monitor height, chair adjustment)
- Gamification (daily posture challenges, achievements)

---

## 8. Project Structure

```
posture-corrector/
├── src/
│   ├── main.py                  # Entry point, app lifecycle
│   ├── capture.py               # Webcam capture module
│   ├── pose_estimator.py        # MediaPipe pose detection wrapper
│   ├── posture_analyzer.py      # Posture analysis engine (angles, classification)
│   ├── calibration.py           # Calibration flow and baseline storage
│   ├── alert_manager.py         # Notification and sound alert logic
│   ├── session_tracker.py       # Session data recording
│   ├── config.py                # User settings and thresholds
│   ├── ui/
│   │   ├── tray.py              # System tray icon and menu
│   │   ├── dashboard.py         # Main window / stats dashboard
│   │   ├── calibration_view.py  # Calibration UI with camera preview
│   │   └── settings_view.py     # Settings panel
│   └── assets/
│       ├── sounds/
│       │   ├── gentle_chime.wav
│       │   └── alert.wav
│       └── icons/
│           ├── green.png
│           ├── yellow.png
│           └── red.png
├── tests/
│   ├── test_posture_analyzer.py
│   ├── test_calibration.py
│   └── test_alert_manager.py
├── data/
│   └── posture.db               # SQLite database (created at runtime)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 9. Dependencies

```
# requirements.txt
mediapipe>=0.10.0
opencv-python>=4.8.0
PyQt6>=6.5.0
playsound>=1.3.0
pyttsx3>=2.90          # optional: text-to-speech
```

All dependencies run locally. No cloud services or API keys required.

---

## 10. Technical Challenges & Mitigations

### 10.1 Challenge: Webcam Angle Variability
**Problem:** Users have webcams at different heights and angles (laptop built-in, external monitor-mounted, etc.), which affects landmark positions.

**Mitigation:** The calibration step captures the user's "good posture" from their specific camera angle. All analysis is relative to this baseline, so absolute camera position doesn't matter.

### 10.2 Challenge: False Positives
**Problem:** Momentary movements (reaching for a drink, turning to talk) could trigger unnecessary alerts.

**Mitigation:**
- Temporal smoothing with a rolling window (10-15 seconds)
- Require sustained bad posture (>70% of frames in window) before alerting
- Alert cooldown period prevents repeated notifications

### 10.3 Challenge: User Fatigue with Notifications
**Problem:** Too many alerts lead users to ignore or disable the app.

**Mitigation:**
- Configurable sensitivity (relaxed / normal / strict)
- Alert cooldown (default 5 minutes between same-issue alerts)
- Subtle system tray color change as passive indicator (alerts are the escalation)
- Track and show improvement to keep users motivated

### 10.4 Challenge: Lighting and Detection Reliability
**Problem:** MediaPipe accuracy drops in poor lighting or with complex backgrounds.

**Mitigation:**
- Monitor landmark detection confidence scores
- If confidence drops below threshold, pause analysis rather than producing false results
- Notify user: "Can't detect posture reliably - try improving lighting"

### 10.5 Challenge: CPU Usage on Older Machines
**Problem:** Continuous pose estimation could consume noticeable CPU on lower-end hardware.

**Mitigation:**
- Default to 2 FPS (sufficient for posture tracking)
- Use MediaPipe Pose Lite model
- Allow users to reduce frame rate further in settings
- Skip processing when system is under heavy load
- Away Mode automatically drops to 0.5 FPS during absences, reducing background load significantly

### 10.6 Challenge: Reliable Background Person Exclusion
**Problem:** The torso-centroid heuristic for background detection could fail if a background person walks close to the user's anchor point, causing incorrect pausing.

**Mitigation:**
- Use a weighted composite: centroid deviation + landmark confidence delta + depth cues (face size as a proxy for distance)
- Require the condition to persist for 2+ seconds before triggering Privacy Pause (avoids pausing on quick walk-bys)
- Allow users to adjust `background_detection_sensitivity` to match their environment
- Log Privacy Pause events (timestamp + duration only) so users can review if the app pauses unexpectedly

### 10.7 Challenge: False Absence Detection
**Problem:** Sustained forward lean or looking down at desk could cause the head/shoulder landmarks to temporarily drop in confidence, incorrectly triggering Away Mode.

**Mitigation:**
- Require *all* primary landmarks (both shoulders + nose/ears) to drop below threshold, not just one
- Check hip landmarks as a secondary anchor — hips remain visible even when leaning forward
- The 8-second grace period absorbs most transient dips before Away Mode is triggered
- Provide a "false pause" feedback button so users can report and tune their threshold

---

## 11. Testing Strategy

### 11.1 Unit Tests
- Angle calculation functions (given known landmark positions, verify correct angles)
- Posture classification logic (given metrics and thresholds, verify correct status)
- Temporal smoothing (verify alerts only fire after sustained bad posture)
- Cooldown logic (verify alerts respect cooldown intervals)

### 11.2 Integration Tests
- End-to-end: feed pre-recorded landmark data through the pipeline, verify correct alerts
- Calibration flow: verify baseline is correctly computed and stored
- Settings persistence: verify config changes survive app restart

### 11.3 Manual Testing
- Test with various webcam positions (laptop, external, different heights)
- Test in different lighting conditions
- Test with different body types and sitting positions
- Verify notification appearance on macOS, Windows, Linux

---

## 12. Success Metrics

### 12.1 Usability
- User can complete calibration in under 30 seconds
- First alert appears within 15 seconds of sustained bad posture
- False positive rate < 10% during normal use

### 12.2 Engagement
- Users keep the app running for >4 hours/day on average
- Posture score improvement after 2 weeks of use
- <20% uninstall rate in first week

### 12.3 Technical
- CPU usage stays under 10% on target hardware
- Memory usage stays under 200MB
- No crashes during 8+ hour continuous sessions

---

## 13. Open Questions

1. **UI Framework Decision:**
   - PyQt6 (native feel, heavier dependency) vs. Tauri (modern web UI, Rust backend, smaller binary) vs. Electron (web UI, larger binary)?
   - Recommendation: Start with PyQt6 for MVP simplicity, consider Tauri for v2 if a more polished UI is needed.

2. **Sensitivity Defaults:**
   - How strict should default thresholds be? Too strict = annoying false positives. Too lenient = misses real issues.
   - Plan: Start with lenient defaults and let users tighten. Collect feedback to tune.

3. **Break Reminders:**
   - Should break/stretch reminders be part of MVP or post-MVP?
   - Recommendation: Post-MVP. Keep MVP focused on posture detection and correction.

4. **Distribution:**
   - Distribute as a standalone binary (PyInstaller) or require Python installed?
   - Recommendation: PyInstaller for end users, pip install for developers.

---

## 14. Getting Started (Next Steps)

1. Set up Python project with `pyproject.toml` and virtual environment
2. Implement webcam capture + MediaPipe pose estimation proof of concept
3. Build posture analysis functions (angle calculations, classification)
4. Implement calibration flow
5. Add system tray with color-coded status icon
6. Add desktop notifications with correction messages
7. Add audio chime alerts
8. Implement temporal smoothing and cooldown logic
9. Build settings panel
10. Package with PyInstaller and test on macOS/Windows
