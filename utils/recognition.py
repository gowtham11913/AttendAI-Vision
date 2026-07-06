"""
Predict the student ID (ROLLNO_NAME) for a given 512-D embedding.

Returns:
    (student_id_or_"Unknown", confidence_float_0_to_1)
"""

from __future__ import annotations

import pickle
from typing import Tuple

import numpy as np


class FaceRecognizer:
    """Wraps a trained SVM classifier + LabelEncoder for face recognition."""

    def __init__(self, svm_model_path: str, label_encoder_path: str,
                 confidence_threshold: float = 0.85):
        """
        Args:
            svm_model_path: path to svm_model.pkl (must be an SVC trained
                             with probability=True).
            label_encoder_path: path to label_encoder.pkl.
            confidence_threshold: minimum prediction probability required to
                                   accept a match; below this, classified as
                                   "Unknown".
        """
        with open(svm_model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.confidence_threshold = confidence_threshold

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(
                "Loaded SVM model does not support predict_proba(). "
                "Retrain with SVC(kernel='linear', probability=True)."
            )

    def predict(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Predict the roll number for a given 512-D embedding.

        Returns:
            (roll_no_or_"Unknown", confidence_float_0_to_1)
        """
        embedding = embedding.reshape(1, -1)
        probabilities = self.model.predict_proba(embedding)[0]

        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])

        if confidence < self.confidence_threshold:
            return "Unknown", confidence

        student_id = self.label_encoder.inverse_transform([best_idx])[0]
        return str(student_id), confidence