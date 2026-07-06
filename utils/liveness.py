"""
liveness.py
-----------
Lightweight challenge-response liveness check to prevent photo/screen
spoofing during real-time attendance.

Why challenge-response instead of blink/EAR detection:
    YuNet only returns 5 facial landmarks (right eye, left eye, nose,
    right mouth corner, left mouth corner) - not the full eye-contour
    points needed for reliable Eye Aspect Ratio (EAR) blink detection.
    Adding a separate landmark model would violate the project rule of
    "no unnecessary AI models".

How it works:
    1. Once a face is recognized with high confidence, capture a short
       BASELINE of that face's landmark geometry (assumes neutral pose).
    2. Issue a random challenge: "Turn LEFT", "Turn RIGHT", "Look UP", or
       "Look DOWN".
    3. Track the person's landmark geometry over the next few seconds and
       check it moves in the requested direction by a large enough margin
       relative to their own baseline (not a fixed absolute threshold,
       since head size/camera distance varies per person/session).
    4. Attendance is marked ONLY if the challenge is completed in time.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# YuNet landmark order: 0=right eye, 1=left eye, 2=nose, 3=right mouth, 4=left mouth
RIGHT_EYE, LEFT_EYE, NOSE, RIGHT_MOUTH, LEFT_MOUTH = 0, 1, 2, 3, 4

CHALLENGES = ["LEFT", "RIGHT", "UP", "DOWN"]

BASELINE_DURATION_SECONDS = 0.8
CHALLENGE_TIMEOUT_SECONDS = 5.0

YAW_MOVEMENT_THRESHOLD = 0.18
PITCH_MOVEMENT_THRESHOLD = 0.18


def _pose_features(landmarks: np.ndarray) -> tuple[float, float, float]:
    eye_mid = (landmarks[RIGHT_EYE] + landmarks[LEFT_EYE]) / 2.0
    mouth_mid = (landmarks[RIGHT_MOUTH] + landmarks[LEFT_MOUTH]) / 2.0
    inter_eye_dist = float(np.linalg.norm(landmarks[LEFT_EYE] - landmarks[RIGHT_EYE]))

    if inter_eye_dist < 1e-3:
        return 0.0, 0.5, 0.0

    yaw_signal = float(landmarks[NOSE][0] - eye_mid[0]) / inter_eye_dist

    vertical_span = float(mouth_mid[1] - eye_mid[1])
    if abs(vertical_span) < 1e-3:
        pitch_signal = 0.5
    else:
        pitch_signal = float(landmarks[NOSE][1] - eye_mid[1]) / vertical_span

    return yaw_signal, pitch_signal, inter_eye_dist


@dataclass
class LivenessChallenge:
    direction: str = ""
    state: str = "idle"
    baseline_started_at: float = 0.0
    challenge_started_at: float = 0.0
    baseline_samples: list = field(default_factory=list)
    baseline_yaw: float = 0.0
    baseline_pitch: float = 0.0
    baseline_scale: float = 0.0

    def start(self) -> None:
        self.direction = random.choice(CHALLENGES)
        self.state = "baseline"
        self.baseline_started_at = time.time()
        self.baseline_samples = []

    def update(self, landmarks: Optional[np.ndarray]) -> str:
        if self.state == "idle":
            return "idle"

        if landmarks is None:
            self.state = "failed"
            return "failed"

        yaw, pitch, scale = _pose_features(landmarks)

        if self.state == "baseline":
            self.baseline_samples.append((yaw, pitch, scale))
            if time.time() - self.baseline_started_at >= BASELINE_DURATION_SECONDS:
                arr = np.array(self.baseline_samples)
                self.baseline_yaw = float(np.median(arr[:, 0]))
                self.baseline_pitch = float(np.median(arr[:, 1]))
                self.baseline_scale = float(np.median(arr[:, 2]))
                self.state = "waiting"
                self.challenge_started_at = time.time()
            return "collecting_baseline"

        if self.state == "waiting":
            if time.time() - self.challenge_started_at > CHALLENGE_TIMEOUT_SECONDS:
                self.state = "failed"
                return "failed"

            if self.baseline_scale > 0 and scale / self.baseline_scale < 0.5:
                return "waiting"

            yaw_delta = yaw - self.baseline_yaw
            pitch_delta = pitch - self.baseline_pitch

            passed = False
            if self.direction == "LEFT" and yaw_delta <= -YAW_MOVEMENT_THRESHOLD:
                passed = True
            elif self.direction == "RIGHT" and yaw_delta >= YAW_MOVEMENT_THRESHOLD:
                passed = True
            elif self.direction == "UP" and pitch_delta <= -PITCH_MOVEMENT_THRESHOLD:
                passed = True
            elif self.direction == "DOWN" and pitch_delta >= PITCH_MOVEMENT_THRESHOLD:
                passed = True

            if passed:
                self.state = "passed"
                return "passed"
            return "waiting"

        return self.state

    def time_remaining(self) -> float:
        if self.state != "waiting":
            return CHALLENGE_TIMEOUT_SECONDS
        return max(0.0, CHALLENGE_TIMEOUT_SECONDS - (time.time() - self.challenge_started_at))

    def reset(self) -> None:
        self.state = "idle"
        self.direction = ""
        self.baseline_samples = []