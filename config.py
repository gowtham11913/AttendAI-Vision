"""
config.py
---------
Central validated runtime configuration for AttendAI Vision.

Default values live in this module.
User-editable overrides are persisted in database/settings.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = BASE_DIR / "database" / "settings.json"


DEFAULT_SETTINGS = {
    "recognition_confidence_threshold": 0.90,
    "liveness_score_threshold": 0.70,
    "attendance_once_per_day": True,
    "camera_index": 0,

    # Provisional, configurable starting value only.
    # This is NOT a calibrated universal ArcFace threshold.
    # Calibrate from genuine and impostor cosine-similarity distributions
    # collected with this project's actual camera, preprocessing pipeline,
    # ArcFace model, registration samples, poses, and operating conditions.
    "arcface_verification_threshold": 0.45,
}


def _validate_settings(settings: dict[str, Any]) -> None:
    recognition_threshold = settings["recognition_confidence_threshold"]
    liveness_threshold = settings["liveness_score_threshold"]
    attendance_once_per_day = settings["attendance_once_per_day"]
    camera_index = settings["camera_index"]
    arcface_verification_threshold = settings[
        "arcface_verification_threshold"
    ]

    if (
        isinstance(recognition_threshold, bool)
        or not isinstance(recognition_threshold, (int, float))
        or not 0.0 <= float(recognition_threshold) <= 1.0
    ):
        raise ValueError(
            "recognition_confidence_threshold must be between 0.0 and 1.0"
        )

    if (
        isinstance(liveness_threshold, bool)
        or not isinstance(liveness_threshold, (int, float))
        or not 0.0 <= float(liveness_threshold) <= 1.0
    ):
        raise ValueError(
            "liveness_score_threshold must be between 0.0 and 1.0"
        )

    if not isinstance(attendance_once_per_day, bool):
        raise ValueError(
            "attendance_once_per_day must be True or False"
        )

    if (
        isinstance(camera_index, bool)
        or not isinstance(camera_index, int)
        or camera_index < 0
    ):
        raise ValueError(
            "camera_index must be a non-negative integer"
        )

    if (
        isinstance(arcface_verification_threshold, bool)
        or not isinstance(arcface_verification_threshold, (int, float))
        or not 0.0 <= float(arcface_verification_threshold) <= 1.0
    ):
        raise ValueError(
            "arcface_verification_threshold must be between 0.0 and 1.0"
        )


def load_settings() -> dict[str, Any]:
    """
    Load persisted settings and merge them with defaults.

    Missing settings.json:
        returns defaults.

    Missing individual keys:
        uses defaults for those keys.

    Invalid JSON or invalid values:
        raises an error instead of silently using unsafe values.
    """
    settings = DEFAULT_SETTINGS.copy()

    if SETTINGS_FILE.is_file():
        with SETTINGS_FILE.open("r", encoding="utf-8") as file:
            saved_settings = json.load(file)

        if not isinstance(saved_settings, dict):
            raise ValueError(
                "settings.json must contain a JSON object"
            )

        for key in DEFAULT_SETTINGS:
            if key in saved_settings:
                settings[key] = saved_settings[key]

    _validate_settings(settings)
    return settings


def save_settings(
    recognition_confidence_threshold: float,
    liveness_score_threshold: float,
    attendance_once_per_day: bool,
    camera_index: int,
    arcface_verification_threshold: float,
) -> dict[str, Any]:
    """Validate and persist runtime-editable settings."""
    settings = {
        "recognition_confidence_threshold": float(
            recognition_confidence_threshold
        ),
        "liveness_score_threshold": float(
            liveness_score_threshold
        ),
        "attendance_once_per_day": attendance_once_per_day,
        "camera_index": camera_index,
        "arcface_verification_threshold": float(
            arcface_verification_threshold
        ),
    }

    _validate_settings(settings)

    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = SETTINGS_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)

    temporary_file.replace(SETTINGS_FILE)
    return settings


def validate_config() -> None:
    """Validate the complete effective configuration."""
    load_settings()


# Backward-compatible process-start effective values.
_effective_settings = load_settings()

RECOGNITION_CONFIDENCE_THRESHOLD = float(
    _effective_settings["recognition_confidence_threshold"]
)

LIVENESS_SCORE_THRESHOLD = float(
    _effective_settings["liveness_score_threshold"]
)

ATTENDANCE_ONCE_PER_DAY = bool(
    _effective_settings["attendance_once_per_day"]
)

CAMERA_INDEX = int(
    _effective_settings["camera_index"]
)

ARCFACE_VERIFICATION_THRESHOLD = float(
    _effective_settings["arcface_verification_threshold"]
)
