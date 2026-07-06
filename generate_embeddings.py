"""
generate_embeddings.py
-----------------------
Phase 2: Embedding Generation.

Reads every registered student's images from dataset/students/<ROLL_NO>/,
detects the face with YuNet (re-validated for safety), generates a 512-D
ArcFace embedding per image, and stores ALL per-image embeddings for that
student stacked into a single array of shape (N, 512) at
embeddings/students/<ROLL_NO>.npy (one file per student, per spec).
Keeping every sample, rather than collapsing to a single average vector,
gives the SVM classifier in Phase 3 multiple training examples per class.

Run:
    python generate_embeddings.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from utils.camera import FaceDetector, crop_and_resize_face
from utils.embedding import EmbeddingExtractor
from utils.logger import get_logger

log = get_logger("generate_embeddings")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "students")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings", "students")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def process_student(roll_no: str, detector: FaceDetector, extractor: EmbeddingExtractor) -> int:
    """
    Generate and save the averaged embedding for a single student.

    Returns the number of images successfully used to build the embedding.
    """
    student_dir = os.path.join(DATASET_DIR, roll_no)
    image_files = sorted(
        f for f in os.listdir(student_dir) if f.lower().endswith(VALID_EXTENSIONS)
    )

    if not image_files:
        log.warning(f"[SKIP] {roll_no}: no images found.")
        return 0

    embeddings = []
    for filename in image_files:
        filepath = os.path.join(student_dir, filename)
        image = cv2.imread(filepath)
        if image is None:
            log.warning(f"[WARN] {roll_no}/{filename}: unreadable image, skipping.")
            continue

        # Images saved during registration are already 112x112 face crops.
        # If an image is not exactly that size (e.g. manually added photos),
        # re-detect and re-crop with YuNet to keep the pipeline consistent.
        if image.shape[:2] != (112, 112):
            face, status = detector.detect_single_face(image)
            if face is None:
                log.warning(f"[WARN] {roll_no}/{filename}: {status}, skipping.")
                continue
            image = crop_and_resize_face(image, face)
            if image is None:
                log.warning(f"[WARN] {roll_no}/{filename}: crop failed, skipping.")
                continue

        embedding = extractor.get_embedding(image)
        if embedding is None:
            log.warning(f"[WARN] {roll_no}/{filename}: embedding generation failed, skipping.")
            continue

        embeddings.append(embedding)

    if not embeddings:
        log.warning(f"[SKIP] {roll_no}: no valid embeddings could be generated.")
        return 0

    # Stack all per-image embeddings into one (N, 512) array -- one file
    # per student, but every sample preserved for SVM training.
    stacked = np.vstack(embeddings).astype(np.float32)

    out_path = os.path.join(
    EMBEDDINGS_DIR,
    f"{os.path.basename(student_dir)}.npy"
    )
    EmbeddingExtractor.save_embedding(stacked, out_path)
    MASTER_EMBEDDINGS = os.path.join(
        BASE_DIR,
        "embeddings",
        "master_embeddings.npy"
    )

    MASTER_LABELS = os.path.join(
        BASE_DIR,
        "embeddings",
        "master_labels.npy"
    )

    labels = np.array(
        [os.path.basename(student_dir)] * len(stacked)
    )


    if os.path.exists(MASTER_EMBEDDINGS):

        old_embeddings = np.load(MASTER_EMBEDDINGS)

        old_labels = np.load(
            MASTER_LABELS,
            allow_pickle=True
        )

        new_embeddings = np.vstack(
            [
                old_embeddings,
                stacked
            ]
        )

        new_labels = np.concatenate(
            [
                old_labels,
                labels
            ]
        )

    else:

        new_embeddings = stacked

        new_labels = labels

    np.save(
        MASTER_EMBEDDINGS,
        new_embeddings
    )

    np.save(
        MASTER_LABELS,
        new_labels
    )


    log.info(f"[OK] {roll_no}: {stacked.shape[0]}/{len(image_files)} embeddings saved.")
    return stacked.shape[0]


def main():

    if not os.path.isdir(DATASET_DIR):
        log.error("No students found.")
        sys.exit(1)

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    log.info(f"Loading YuNet: {MODEL_PATH}")
    detector = FaceDetector(MODEL_PATH)

    log.info("Loading ArcFace...")
    extractor = EmbeddingExtractor()

    student_folders = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )

    generated = 0
    skipped = 0

    for folder_name in student_folders:

        embedding_file = os.path.join(
            EMBEDDINGS_DIR,
            f"{folder_name}.npy"
        )

        if os.path.exists(embedding_file):
            log.info(f"[SKIP] {folder_name} already has embeddings.")
            skipped += 1
            continue

        log.info(f"[NEW] Generating embeddings for {folder_name}")

        if process_student(folder_name, detector, extractor) > 0:
            generated += 1

        
    log.info("=" * 50)
    log.info("EMBEDDING GENERATION COMPLETE")
    log.info(f"Generated : {generated}")
    log.info(f"Skipped   : {skipped}")
    log.info("=" * 50)

    log.info("Next step: run 'python train_model.py' to train the SVM classifier.")



if __name__ == "__main__":
    main()