"""
train_model.py
---------------
Phase 3: Model Training.

Loads all per-student embedding arrays from embeddings/students/*.npy
(each of shape (N, 512)), builds the (X, y) training set, encodes roll
numbers as labels, trains a linear SVM classifier with probability
estimates enabled (required for confidence scoring in Phase 4), and saves:

    models/svm_model.pkl
    models/label_encoder.pkl

Run:
    python train_model.py
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.logger import get_logger

log = get_logger("train_model")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings", "students")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


def load_dataset():
    """
    Load all embeddings/students/<ROLLNO_NAME>.npy files.

    Returns:
        X: np.ndarray of shape (total_samples, 512)
        y: list[str] of student IDs (ROLLNO_NAME), one per sample
        per_student_counts: dict[roll_no -> num_images]
    """
    npy_files = sorted(f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy"))
    if not npy_files:
        log.error("No embeddings found in embeddings/students/. Run generate_embeddings.py first.")
        sys.exit(1)

    X_list = []
    y_list = []
    per_student_counts = {}

    for filename in npy_files:
        student_id = os.path.splitext(filename)[0]
        path = os.path.join(EMBEDDINGS_DIR, filename)
        arr = np.load(path)

        # Support both (512,) single embeddings and (N, 512) stacked arrays.
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        X_list.append(arr)
        y_list.extend([student_id] * arr.shape[0])

        per_student_counts[student_id] = arr.shape[0]

    X = np.vstack(X_list).astype(np.float32)
    y = y_list
    return X, y, per_student_counts


def main():
    log.info("Loading embeddings...")
    X, y, per_student_counts = load_dataset()

    num_students = len(per_student_counts)
    num_images = X.shape[0]

    if num_students < 2:
        log.error("At least 2 registered students are required to train a classifier.")
        sys.exit(1)

    log.info(f"Students: {num_students}")
    log.info(f"Total images/embeddings: {num_images}")

    log.info("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Hold out a small validation split if there is enough data per class;
    # otherwise train and report on the full set (small registration sizes
    # in a classroom setting may not support a stratified split per class).
    min_class_count = min(per_student_counts.values())
    use_split = min_class_count >= 3

    log.info("Training Linear SVM classifier...")
    model = SVC(kernel="linear", probability=True, random_state=42)

    if use_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # Refit on full dataset for the final deployed model.
        model.fit(X, y_encoded)
    else:
        model.fit(X, y_encoded)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y_encoded, y_pred)
        log.warning("Not enough samples per class for a held-out validation "
                    "split; accuracy below is computed on the training set itself.")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    log.info("=" * 50)
    log.info("MODEL TRAINING COMPLETE")
    log.info(f"Students          : {num_students}")
    log.info(f"Images/Embeddings : {num_images}")
    log.info(f"Accuracy          : {accuracy * 100:.2f}%")
    log.info(f"Model saved to    : {SVM_MODEL_PATH}")
    log.info(f"Encoder saved to  : {LABEL_ENCODER_PATH}")
    log.info("=" * 50)
    log.info("Next step: run 'python attendance_system.py' to start real-time attendance.")


if __name__ == "__main__":
    main()