"""
register_student.py
--------------------
Phase 1: Student Registration.

Flow:
    1. Prompt for Roll Number, Name, Department.
    2. Reject if roll number already exists in database/students.csv.
    3. Create dataset/students/<ROLL_NO>/ directory.
    4. Guide the user through 5 poses (FRONT, LEFT, RIGHT, UP, DOWN), one at
       a time. For each pose:
         - Wait for SPACE to start.
         - 3-2-1 countdown.
         - Auto-capture 10 images, every 300ms, requiring exactly one valid
           face per frame (rejecting no-face / multi-face / bad-size frames).
         - Crop each face to 112x112 and save as <pose>_01.jpg ... _10.jpg.
    5. Append the new student record to database/students.csv.

Run:
    python register_student.py
"""

from __future__ import annotations
import shutil
import numpy as np
from utils.embedding import EmbeddingExtractor, find_duplicate_face
import os
import sys
import time

import cv2
from utils.logger import get_logger
log = get_logger("register_student")
from utils.camera import Camera, FaceDetector, crop_and_resize_face
from utils.attendance import StudentDatabase

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "students")
STUDENTS_CSV = os.path.join(BASE_DIR, "database", "students.csv")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings", "students")
DUPLICATE_FACE_THRESHOLD = 0.55

POSES = ["front", "left", "right", "up", "down"]
IMAGES_PER_POSE = 10
CAPTURE_INTERVAL_MS = 300
COUNTDOWN_SECONDS = 3

WINDOW_NAME = "Smart Attendance - Student Registration"

POSE_INSTRUCTIONS = {
    "front": "Look straight at the camera",
    "left": "Turn your head to the LEFT",
    "right": "Turn your head to the RIGHT",
    "up": "Tilt your head UP",
    "down": "Tilt your head DOWN",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prompt_student_details() -> tuple[str, str, str]:
    log.info("=" * 50)
    log.info("STUDENT REGISTRATION")
    log.info("=" * 50)

    # Roll Number
    roll_no = input("Enter Roll Number: ").strip().upper()

    # Name
    name = input("Enter Name: ").strip().title()

    # Department
    department = input("Enter Department: ").strip().upper()

    if not roll_no or not name or not department:
        log.error("Roll Number, Name and Department are all required. Registration aborted.")
        sys.exit(1)

    return roll_no, name, department


def draw_overlay(frame, name: str, roll_no: str, pose: str, captured: int,
                  total: int, status: str, status_color=(0, 255, 0)):
    """Draw the registration HUD onto the frame (in place)."""
    h, w = frame.shape[:2]
    overlay_h = 130
    cv2.rectangle(frame, (0, 0), (w, overlay_h), (30, 30, 30), -1)

    cv2.putText(frame, f"Name: {name}   Roll No: {roll_no}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Pose: {pose.upper()}  -  {POSE_INSTRUCTIONS[pose]}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    cv2.putText(frame, f"Captured: {captured}/{total}", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (15, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)


def draw_face_box(frame, face):
    x, y, w, h = face.box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


STATUS_MESSAGES = {
    "ok": ("Face OK", (0, 255, 0)),
    "no_face": ("No face detected", (0, 0, 255)),
    "multiple_faces": ("Multiple faces detected - only one allowed", (0, 0, 255)),
    "face_too_small": ("Face too small - move closer", (0, 165, 255)),
    "face_too_large": ("Face too large - move back", (0, 165, 255)),
}


def wait_for_space_or_quit(frame_provider, detector, name, roll_no, pose):
    """
    Show live preview until the user presses SPACE (proceed) or 'q' (quit).
    Returns True to proceed, False to quit.
    """
    while True:
        frame = frame_provider()
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        face, status = detector.detect_single_face(frame)
        if face is not None:
            draw_face_box(display, face)
        msg, color = STATUS_MESSAGES[status]

        draw_overlay(display, name, roll_no, pose, 0, IMAGES_PER_POSE, msg, color)
        cv2.putText(display, "Press SPACE to capture | Q to quit", (15, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        if key == ord('q'):
            return False


def run_countdown(frame_provider, name, roll_no, pose):
    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        start = time.time()
        while time.time() - start < 1.0:
            frame = frame_provider()
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            draw_overlay(display, name, roll_no, pose, 0, IMAGES_PER_POSE, "Get ready...")
            h, w = display.shape[:2]
            cv2.putText(display, str(remaining), (w // 2 - 30, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 220, 255), 6)
            cv2.imshow(WINDOW_NAME, display)
            cv2.waitKey(1)


def capture_pose_images(frame_provider, detector, name, roll_no, pose, save_dir) -> int:
    """
    Auto-capture IMAGES_PER_POSE valid face images for the given pose,
    one every CAPTURE_INTERVAL_MS, saving 112x112 crops to save_dir.

    Returns the number of images successfully saved.
    """
    saved = 0
    last_capture_time = 0.0

    while saved < IMAGES_PER_POSE:
        frame = frame_provider()
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        face, status = detector.detect_single_face(frame)
        msg, color = STATUS_MESSAGES[status]
        if face is not None:
            draw_face_box(display, face)

        draw_overlay(display, name, roll_no, pose, saved, IMAGES_PER_POSE, msg, color)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            log.warning(f"Registration cancelled by user during pose '{pose}' capture.")
            sys.exit(0)

        now = time.time() * 1000  # ms
        if face is not None and (now - last_capture_time) >= CAPTURE_INTERVAL_MS:
            face_crop = crop_and_resize_face(frame, face)
            if face_crop is not None:
                saved += 1
                filename = f"{pose}_{saved:02d}.jpg"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, face_crop)
                last_capture_time = now
                log.debug(f"Saved {filename}")

    return saved

def check_for_duplicate_face(student_dir, extractor):
    front_images = sorted(
        f for f in os.listdir(student_dir) if f.startswith("front_") and f.endswith(".jpg")
    )
    embeddings = []
    for filename in front_images:
        image = cv2.imread(os.path.join(student_dir, filename))
        if image is None:
            continue
        embedding = extractor.get_embedding(image)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        return None, 0.0

    candidate = np.vstack(embeddings)
    return find_duplicate_face(candidate, EMBEDDINGS_DIR, similarity_threshold=DUPLICATE_FACE_THRESHOLD)
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("Application started")

    roll_no, name, department = prompt_student_details()
    log.info(f"Student registration started - Roll Number: {roll_no}, Name: {name}, Department: {department}")

    db = StudentDatabase(STUDENTS_CSV)
    if db.exists(roll_no):
        log.warning(f"Duplicate roll number: '{roll_no}' is already registered. Registration stopped.")
        sys.exit(1)

    safe_name = "_".join(name.split())

    student_folder = f"{roll_no}_{safe_name}"

    student_dir = os.path.join(
        DATASET_DIR,
        student_folder
    )

    os.makedirs(student_dir, exist_ok=True)
    log.info(f"Folder created: {student_dir}")

    log.info(f"Loading YuNet face detector from: {MODEL_PATH}")
    detector = FaceDetector(MODEL_PATH)

    log.info("Loading ArcFace embedding model (for duplicate-face check)...")
    extractor = EmbeddingExtractor()


    total_saved = 0
    with Camera(index=0) as camera:
        log.info("Camera initialized")
        cv2.namedWindow(WINDOW_NAME)

        for pose in POSES:
            log.info(f"Pose started: {pose.upper()}")
            proceed = wait_for_space_or_quit(camera.read, detector, name, roll_no, pose)
            if not proceed:
                log.warning("Registration cancelled by user.")
                cv2.destroyAllWindows()
                sys.exit(0)

            run_countdown(camera.read, name, roll_no, pose)
            saved = capture_pose_images(camera.read, detector, name, roll_no, pose, student_dir)
            total_saved += saved
            log.info(f"Pose '{pose}' complete: {saved}/{IMAGES_PER_POSE} images saved.")

            # FIX: this block was previously dedented to sit AFTER the `for`
            # loop, so it only ran once with `pose` stuck at its final value
            # ("down"), meaning the duplicate-face check never fired. It must
            # run inside the loop, right after the "front" pose is captured.
            if pose == "front":
                log.info("Checking this face against already-registered students...")
                matched_roll_no, similarity = check_for_duplicate_face(student_dir, extractor)
                if matched_roll_no is not None:
                    existing = db.get_student(matched_roll_no)
                    existing_name = existing["name"] if existing else "Unknown"
                    cv2.destroyAllWindows()
                    log.warning(f"Duplicate face detected: matches Roll No {matched_roll_no} ({existing_name}), similarity {similarity*100:.1f}%. Registration blocked.")
                    shutil.rmtree(student_dir, ignore_errors=True)
                    sys.exit(1)
                else:
                    log.info(f"No duplicate found (best similarity: {similarity*100:.1f}%). Continuing...")

    cv2.destroyAllWindows()

    db.add_student(roll_no, name, department, total_saved)
    log.info("Database updated: students.csv")

    log.info("=" * 50)
    log.info("REGISTRATION COMPLETE")
    log.info(f"Roll No   : {roll_no}")
    log.info(f"Name      : {name}")
    log.info(f"Department: {department}")
    log.info(f"Images    : {total_saved}")
    log.info("=" * 50)
    log.info("Next step: run 'python generate_embeddings.py' to generate face embeddings.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        log.warning("Registration cancelled by user (KeyboardInterrupt).")
        sys.exit(0)
    except Exception as e:
        log.critical(f"Unhandled exception during registration: {e}", exc_info=True)
        raise