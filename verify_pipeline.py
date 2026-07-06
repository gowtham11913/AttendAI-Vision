"""
verify_pipeline.py
-------------------
Standalone sanity-checker for the AI-Based Smart Attendance System.

Runs no models and opens no camera -- it only inspects files already on
disk (CSVs, .npy embeddings, .pkl models, dataset images) and checks that
everything is internally consistent. Safe to run after any pipeline stage,
or any time you want a health check.

Run:
    python verify_pipeline.py
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "students")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings", "students")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STUDENTS_CSV = os.path.join(BASE_DIR, "database", "students.csv")
ATTENDANCE_CSV = os.path.join(BASE_DIR, "database", "attendance.csv")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

POSES = ["front", "left", "right", "up", "down"]
IMAGES_PER_POSE = 10
EMBEDDING_DIM = 512

errors = []
warnings = []
oks = []


def ok(msg):
    oks.append(msg)


def warn(msg):
    warnings.append(msg)


def err(msg):
    errors.append(msg)


# ---------------------------------------------------------------------------
# 1. Required model files present
# ---------------------------------------------------------------------------
def check_required_models():
    print("\n[1] Required model files")
    yunet = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    antispoof_dir = os.path.join(MODELS_DIR, "anti_spoof_models")

    if os.path.isfile(yunet):
        ok(f"YuNet model found: {yunet}")
    else:
        err(f"YuNet model MISSING: {yunet}")

    if os.path.isdir(antispoof_dir) and os.listdir(antispoof_dir):
        ok(f"Anti-spoof models found: {antispoof_dir}")
    else:
        err(f"Anti-spoof models MISSING or empty: {antispoof_dir}")


# ---------------------------------------------------------------------------
# 2. students.csv structure + roll_no uniqueness
# ---------------------------------------------------------------------------
def check_students_csv():
    print("[2] students.csv")
    if not os.path.isfile(STUDENTS_CSV):
        err("database/students.csv does not exist yet (no one registered).")
        return None

    df = pd.read_csv(STUDENTS_CSV, dtype={"roll_no": str})
    expected_cols = ["roll_no", "name", "department", "image_count", "registered_date"]
    if list(df.columns) != expected_cols:
        err(f"students.csv columns mismatch. Expected {expected_cols}, got {list(df.columns)}")
    else:
        ok("students.csv columns correct.")

    dupes = df[df.duplicated("roll_no", keep=False)]
    if not dupes.empty:
        err(f"Duplicate roll_no rows found in students.csv: {sorted(dupes['roll_no'].unique())}")
    else:
        ok(f"No duplicate roll numbers ({len(df)} students registered).")

    return df


# ---------------------------------------------------------------------------
# 3. dataset/students/<ROLL>_<Name>/ folders: image counts per pose
# ---------------------------------------------------------------------------
def check_dataset_images(students_df):
    print("[3] dataset/students/ images")
    if not os.path.isdir(DATASET_DIR):
        err("dataset/students/ directory does not exist yet.")
        return

    folders = sorted(
        d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    if not folders:
        warn("No student folders found in dataset/students/.")
        return

    known_rolls = set(students_df["roll_no"].astype(str)) if students_df is not None else set()

    for folder in folders:
        folder_dir = os.path.join(DATASET_DIR, folder)
        roll_no = folder.split("_", 1)[0]

        if roll_no not in known_rolls:
            warn(f"{folder}: roll_no '{roll_no}' has no matching row in students.csv "
                 f"(registration may have been interrupted before add_student() ran).")

        for pose in POSES:
            count = len([
                f for f in os.listdir(folder_dir)
                if f.startswith(f"{pose}_") and f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            if count != IMAGES_PER_POSE:
                warn(f"{folder}: pose '{pose}' has {count}/{IMAGES_PER_POSE} images "
                     f"(expected exactly {IMAGES_PER_POSE}).")

        ok(f"{folder}: checked ({len(os.listdir(folder_dir))} files total).")


# ---------------------------------------------------------------------------
# 4. embeddings/students/*.npy: shape + naming consistency with dataset
# ---------------------------------------------------------------------------
def check_embeddings():
    print("[4] embeddings/students/*.npy")
    if not os.path.isdir(EMBEDDINGS_DIR):
        err("embeddings/students/ directory does not exist yet (run generate_embeddings.py).")
        return

    npy_files = sorted(f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy"))
    if not npy_files:
        warn("No .npy embedding files found yet.")
        return

    dataset_folders = set(
        d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))
    ) if os.path.isdir(DATASET_DIR) else set()

    for filename in npy_files:
        student_id = os.path.splitext(filename)[0]
        path = os.path.join(EMBEDDINGS_DIR, filename)

        try:
            arr = np.load(path)
        except Exception as e:
            err(f"{filename}: failed to load ({e})")
            continue

        if arr.ndim != 2 or arr.shape[1] != EMBEDDING_DIM:
            err(f"{filename}: unexpected shape {arr.shape}, expected (N, {EMBEDDING_DIM}).")
            continue

        # Every row should be L2-normalized (norm ~= 1.0) per embedding.py's contract.
        norms = np.linalg.norm(arr, axis=1)
        bad_norms = np.sum(np.abs(norms - 1.0) > 1e-3)
        if bad_norms > 0:
            warn(f"{filename}: {bad_norms}/{len(norms)} rows are not L2-normalized "
                 f"(expected norm ~1.0).")

        if student_id not in dataset_folders:
            warn(f"{filename}: no matching folder in dataset/students/ "
                 f"(orphaned embedding, or dataset was moved/renamed after embedding).")

        ok(f"{filename}: shape {arr.shape}, OK.")

    # master_embeddings.npy / master_labels.npy are currently unused by
    # train_model.py -- just a heads-up, not an error.
    master_path = os.path.join(BASE_DIR, "embeddings", "master_embeddings.npy")
    if os.path.isfile(master_path):
        warn("master_embeddings.npy exists but train_model.py does not read it "
             "-- this is expected/harmless, not a bug.")


# ---------------------------------------------------------------------------
# 5. Trained model + label encoder: existence, loadability, class alignment
# ---------------------------------------------------------------------------
def check_trained_model():
    print("[5] Trained SVM model + label encoder")
    if not os.path.isfile(SVM_MODEL_PATH) or not os.path.isfile(LABEL_ENCODER_PATH):
        warn("No trained model yet (run train_model.py after registering >= 2 students).")
        return

    try:
        with open(SVM_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        err(f"Failed to load model/encoder: {e}")
        return

    if not hasattr(model, "predict_proba"):
        err("Loaded SVM does not support predict_proba() -- was it trained with "
            "probability=True? attendance_system.py will crash without this.")
    else:
        ok("SVM model supports predict_proba().")

    classes = set(label_encoder.classes_)
    embedded_students = set()
    if os.path.isdir(EMBEDDINGS_DIR):
        embedded_students = set(
            os.path.splitext(f)[0] for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy")
        )

    if classes != embedded_students:
        missing_from_model = embedded_students - classes
        extra_in_model = classes - embedded_students
        if missing_from_model:
            warn(f"Students with embeddings but NOT in the trained model "
                 f"(stale model, retrain needed): {sorted(missing_from_model)}")
        if extra_in_model:
            warn(f"Classes in the trained model with no current embedding file "
                 f"(embeddings deleted since last training?): {sorted(extra_in_model)}")
    else:
        ok(f"Label encoder classes match embedded students exactly ({len(classes)} students).")


# ---------------------------------------------------------------------------
# 6. attendance.csv structure + one-mark-per-day enforcement
# ---------------------------------------------------------------------------
def check_attendance_csv():
    print("[6] attendance.csv")
    if not os.path.isfile(ATTENDANCE_CSV):
        warn("database/attendance.csv does not exist yet (no attendance marked).")
        return

    df = pd.read_csv(ATTENDANCE_CSV, dtype={"roll_no": str})
    expected_cols = ["roll_no", "name", "date", "time", "status"]
    if list(df.columns) != expected_cols:
        err(f"attendance.csv columns mismatch. Expected {expected_cols}, got {list(df.columns)}")
    else:
        ok("attendance.csv columns correct.")

    dupes = df[df.duplicated(subset=["roll_no", "date"], keep=False)]
    if not dupes.empty:
        err(f"Multiple attendance rows for the same student on the same day "
            f"(one-per-day rule violated): "
            f"{sorted(dupes[['roll_no', 'date']].apply(tuple, axis=1).unique())}")
    else:
        ok(f"One-mark-per-day rule holds across {len(df)} attendance records.")

    # Underscore-name bug check: flag any name field that still contains an
    # underscore, which would indicate the attendance_system.py fix isn't
    # applied / isn't working.
    if "name" in df.columns:
        bad_names = df[df["name"].astype(str).str.contains("_", na=False)]
        if not bad_names.empty:
            err(f"Found underscores in attendance name field (name-lookup bug not "
                f"fixed or old data pre-dates the fix): "
                f"{sorted(bad_names['name'].unique())}")
        else:
            ok("No underscores found in attendance name field.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("SMART ATTENDANCE SYSTEM - VERIFICATION REPORT")
    print("=" * 60)

    check_required_models()
    students_df = check_students_csv()
    check_dataset_images(students_df)
    check_embeddings()
    check_trained_model()
    check_attendance_csv()

    print("\n" + "=" * 60)
    print(f"RESULT: {len(oks)} OK, {len(warnings)} warning(s), {len(errors)} error(s)")
    print("=" * 60)

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  ! {w}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  X {e}")

    if not errors and not warnings:
        print("\nEverything checks out.")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
