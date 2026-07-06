"""
antispoof.py
------------
Passive, single-frame face anti-spoofing using the Silent-Face-Anti-Spoofing
(MiniFASNet) models, converted from the official PyTorch weights to ONNX so
this project only ever needs onnxruntime at runtime (already a dependency
for ArcFace/YuNet), never PyTorch.

Source: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
License: Apache 2.0 (see THIRD_PARTY_NOTICES.md)

How it works:
    Two small MiniFASNet models (V1SE and V2) are run as an ensemble on a
    scale-expanded crop around the detected face, each producing a 3-class
    softmax score. The scores are summed and the argmax class is taken:
    class 1 = real face, classes 0/2 = fake (print / replay). This exactly
    mirrors the official reference inference logic (src/test.py in the
    upstream repo), just re-implemented against ONNX Runtime + our own
    YuNet bounding box instead of their bundled RetinaFace detector.

No challenge, no waiting state, no retry loop: one decision per frame,
in milliseconds on CPU.

Known limitations (documented honestly):
    - This model was trained primarily on print/replay attacks; very
      high-quality 3D masks or extremely high-res screens held far enough
      away may still occasionally pass. No anti-spoofing model is 100%.
    - Accuracy depends on face crop quality/lighting just like the
      recognition pipeline; very poor lighting can lower confidence for
      genuine live faces too. If real users are being flagged as fake,
      check `LIVENESS_SCORE_THRESHOLD` first before assuming the model
      is wrong - and improve lighting/camera framing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

# Real-face class index in the 3-class softmax output (1 = real, 0/2 = fake),
# matching the official Silent-Face-Anti-Spoofing convention.
REAL_CLASS_INDEX = 1

# Minimum ensemble-averaged probability assigned to the "real" class to be
# accepted as a live face. The official demo app exposes this as a
# user-adjustable threshold; 0.7 is a reasonable, fairly strict default.
LIVENESS_SCORE_THRESHOLD = 0.70


@dataclass
class _SubModel:
    session: ort.InferenceSession
    input_name: str
    crop_size: Tuple[int, int]  # (h, w)
    scale: float                # bbox expansion factor used at training time


@dataclass
class LivenessResult:
    score: float           # ensemble-averaged probability of "real"
    is_live: bool
    per_model_scores: List[float]


class AntiSpoofDetector:
    """
    Loads the MiniFASNetV1SE + MiniFASNetV2 ONNX ensemble from
    models/anti_spoof_models/ and runs passive liveness checks against a
    YuNet face bounding box on the *original, uncropped* frame (it needs
    surrounding context beyond just the tight face crop, since it expands
    the box by a scale factor itself - this is different from the
    112x112 crop used for ArcFace).
    """

    def __init__(self, models_dir: str, threshold: float = LIVENESS_SCORE_THRESHOLD):
        if not os.path.isdir(models_dir):
            raise FileNotFoundError(
                f"Anti-spoofing models directory not found: {models_dir}"
            )

        onnx_files = sorted(f for f in os.listdir(models_dir) if f.endswith(".onnx"))
        if not onnx_files:
            raise FileNotFoundError(
                f"No .onnx anti-spoofing models found in {models_dir}. "
                "Expected '2.7_80x80_MiniFASNetV2.onnx' and "
                "'4_0_0_80x80_MiniFASNetV1SE.onnx'."
            )

        self.threshold = threshold
        self.sub_models: List[_SubModel] = []

        for filename in onnx_files:
            path = os.path.join(models_dir, filename)
            h_input, w_input, scale = self._parse_model_filename(filename)

            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name

            self.sub_models.append(_SubModel(
                session=session,
                input_name=input_name,
                crop_size=(h_input, w_input),
                scale=scale,
            ))

    @staticmethod
    def _parse_model_filename(filename: str) -> Tuple[int, int, float]:
        """
        Parse filenames using the exact upstream naming convention from
        Silent-Face-Anti-Spoofing's parse_model_name(), e.g.:
            '2.7_80x80_MiniFASNetV2.onnx'      -> (h=80, w=80, scale=2.7)
            '4_0_0_80x80_MiniFASNetV1SE.onnx'  -> (h=80, w=80, scale=4.0)
        Only the FIRST underscore-separated token is the scale value; any
        extra tokens before the WxH part (as in the second example above)
        are naming artifacts from upstream and are ignored, exactly as the
        original parse_model_name() does.
        """
        pth_style_name = filename.replace(".onnx", ".pth")
        info = pth_style_name.split("_")[0:-1]  # drop trailing 'ModelType.pth'
        h_str, w_str = info[-1].split("x")
        scale = float(info[0])
        return int(h_str), int(w_str), scale

    @staticmethod
    def _scale_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                     scale: float, out_size: Tuple[int, int]) -> np.ndarray:
        """
        Reproduce the official CropImage.crop() logic: expand the face
        bbox about its center by `scale`, clip to image bounds, then
        resize to out_size (h, w).
        """
        src_h, src_w = frame.shape[:2]
        x, y, box_w, box_h = bbox

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
        new_w = box_w * scale
        new_h = box_h * scale
        center_x, center_y = box_w / 2 + x, box_h / 2 + y

        left = center_x - new_w / 2
        top = center_y - new_h / 2
        right = center_x + new_w / 2
        bottom = center_y + new_h / 2

        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > src_w - 1:
            left -= (right - src_w + 1)
            right = src_w - 1
        if bottom > src_h - 1:
            top -= (bottom - src_h + 1)
            bottom = src_h - 1

        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        crop = frame[top:bottom + 1, left:right + 1]
        if crop.size == 0:
            return None

        h, w = out_size
        return cv2.resize(crop, (w, h))

    def check(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> LivenessResult:
        """
        Run the passive liveness ensemble.

        Args:
            frame: the full original BGR frame (not a pre-cropped face) -
                   the model needs surrounding context to expand the crop
                   itself, matching how it was trained.
            bbox: (x, y, w, h) face bounding box from YuNet, in frame
                  pixel coordinates.

        Returns:
            LivenessResult with the combined "real" probability and a
            boolean decision against `self.threshold`.
        """
        summed_probs = np.zeros(3, dtype=np.float32)
        per_model_scores = []
        valid_models = 0

        for sub in self.sub_models:
            crop = self._scale_crop(frame, bbox, sub.scale, sub.crop_size)
            if crop is None:
                continue

            # Exact upstream preprocessing: BGR, HWC->CHW, float32, NO /255
            # normalization (verified against the original ToTensor()).
            tensor = crop.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]

            logits = sub.session.run(None, {sub.input_name: tensor})[0][0]
            probs = self._softmax(logits)
            summed_probs += probs
            per_model_scores.append(float(probs[REAL_CLASS_INDEX]))
            valid_models += 1

        if valid_models == 0:
            return LivenessResult(score=0.0, is_live=False, per_model_scores=[])

        avg_probs = summed_probs / valid_models
        real_score = float(avg_probs[REAL_CLASS_INDEX])

        return LivenessResult(
            score=real_score,
            is_live=real_score >= self.threshold,
            per_model_scores=per_model_scores,
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits))
        return exp / np.sum(exp)
