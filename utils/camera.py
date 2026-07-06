"""
camera.py
---------
Camera capture and YuNet-based face detection utilities.

Provides:
    - Camera: thin wrapper around cv2.VideoCapture with safe open/close.
    - FaceDetector: wraps OpenCV's YuNet (cv2.FaceDetectorYN) model and
      exposes a single-face detection helper used across all phases.

Design notes:
    - YuNet is the ONLY face detector used in this project (per project rules).
    - FaceDetector.detect_single_face() enforces "exactly one face" policy
      and validates face size, since registration/attendance both require
      a single, well-framed face.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Standard face crop size used everywhere in this project (per spec: 112x112).
FACE_CROP_SIZE: Tuple[int, int] = (112, 112)

# Minimum / maximum face bounding-box edge length (in pixels, relative to the
# captured frame) to reject faces that are too small (too far) or too large
# (too close) for a usable embedding.
MIN_FACE_FRACTION = 0.10   # face width must be >= 10% of frame width
MAX_FACE_FRACTION = 0.85   # face width must be <= 85% of frame width


@dataclass
class FaceBox:
    """Result of a single detected face from YuNet."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    landmarks: np.ndarray  # shape (5, 2): right eye, left eye, nose, right mouth, left mouth

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class Camera:
    """Safe wrapper around cv2.VideoCapture."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> "Camera":
        self._cap = cv2.VideoCapture(self.index)
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Unable to open camera at index {self.index}. "
                "Check that a webcam is connected and not in use by another app."
            )
        return self

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class FaceDetector:
    """
    Wraps OpenCV's YuNet face detector (cv2.FaceDetectorYN).

    Model file must be present at:
        models/face_detection_yunet_2023mar.onnx
    Download from the OpenCV Zoo if missing:
        https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
    """

    def __init__(self, model_path: str, input_size: Tuple[int, int] = (320, 320),
                 score_threshold: float = 0.85, nms_threshold: float = 0.3,
                 top_k: int = 50):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"YuNet model not found at '{model_path}'. "
                "Download 'face_detection_yunet_2023mar.onnx' from the OpenCV Zoo "
                "and place it in the models/ directory."
            )
        self.model_path = model_path
        self.input_size = input_size
        self._detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def _set_input_size_for_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """Run YuNet on a BGR frame and return all detected faces."""
        self._set_input_size_for_frame(frame)
        _, faces = self._detector.detect(frame)
        results: List[FaceBox] = []
        
        if faces is None:
            return results

        for f in faces:
            x = int(f[0])
            y = int(f[1])
            w = int(f[2])
            h = int(f[3])

            landmarks = (
                f[4:14]
                .reshape(5, 2)
                .astype(np.float32)
            )

            confidence = float(f[14])

            results.append(
                FaceBox(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    confidence=confidence,
                    landmarks=landmarks,
                )
            )


        return results

    def detect_single_face(
        self, frame: np.ndarray
    ) -> Tuple[Optional[FaceBox], str]:
        """
        Enforce the "exactly one usable face" policy.

        Returns:
            (FaceBox or None, status_message)

        status_message is one of:
            "ok"              -> exactly one valid face found
            "no_face"         -> no face detected
            "multiple_faces"  -> more than one face detected
            "face_too_small"  -> face bounding box below MIN_FACE_FRACTION
            "face_too_large"  -> face bounding box above MAX_FACE_FRACTION
        """
        faces = self.detect(frame)

        if len(faces) == 0:
            return None, "no_face"
        if len(faces) > 1:
            return None, "multiple_faces"

        face = faces[0]
        frame_w = frame.shape[1]
        face_fraction = face.w / float(frame_w)

        if face_fraction < MIN_FACE_FRACTION:
            return None, "face_too_small"
        if face_fraction > MAX_FACE_FRACTION:
            return None, "face_too_large"

        return face, "ok"


def crop_and_resize_face(
    frame: np.ndarray, face: FaceBox, output_size: Tuple[int, int] = FACE_CROP_SIZE,
    margin: float = 0.2
) -> Optional[np.ndarray]:
    """
    Crop the face region (with a small margin) from the frame and resize it
    to the standard output_size (default 112x112), as required for storage.

    Returns None if the crop would fall entirely outside the frame.
    """
    h_frame, w_frame = frame.shape[:2]
    x, y, w, h = face.box

    # Expand box by margin on each side.
    mx = int(w * margin)
    my = int(h * margin)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_frame, x + w + mx)
    y2 = min(h_frame, y + h + my)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    resized = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
    return resized
