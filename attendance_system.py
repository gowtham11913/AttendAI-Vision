"""
attendance_system.py
---------------------
Phase 4: Real-Time Attendance.

Opens the camera, detects a face with YuNet, generates its ArcFace
embedding, predicts the student's identity using the trained SVM, displays
roll number / name / confidence / attendance status on screen, and marks
attendance in database/attendance.csv (once per day per student).

Run:
    python attendance_system.py
"""

from __future__ import annotations

import os
import sys
import time

import cv2

from utils.camera import Camera, FaceDetector, crop_and_resize_face
from utils.embedding import EmbeddingExtractor
from utils.recognition import FaceRecognizer
from utils.attendance import AttendanceLog, StudentDatabase
from utils.antispoof import AntiSpoofDetector
from utils.logger import get_logger
from config import (
    CAMERA_INDEX,
    LIVENESS_SCORE_THRESHOLD,
    RECOGNITION_CONFIDENCE_THRESHOLD,
    validate_config,
)
log = get_logger("attendance_system")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
ANTISPOOF_MODELS_DIR = os.path.join(BASE_DIR, "models", "anti_spoof_models")
SVM_MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
STUDENTS_CSV = os.path.join(BASE_DIR, "database", "students.csv")
ATTENDANCE_CSV = os.path.join(BASE_DIR, "database", "attendance.csv")

WINDOW_NAME = "Smart Attendance - Real-Time Recognition"

# How long to keep showing a "Liveness Passed / Failed" message before
# resetting and allowing a fresh attempt.
RESULT_HOLD_SECONDS = 2.0

STATUS_MESSAGES = {
    "ok": "Face OK",
    "no_face": "No face detected",
    "multiple_faces": "Multiple faces detected",
    "face_too_small": "Face too small - move closer",
    "face_too_large": "Face too large - move back",
}


def draw_hud(frame, lines, origin=(15, 30), color=(255, 255, 255), line_height=28):
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (origin[0], origin[1] + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def main():
    validate_config()
    if not os.path.isfile(SVM_MODEL_PATH) or not os.path.isfile(LABEL_ENCODER_PATH):
        log.error("Trained model not found. Run train_model.py first.")
        sys.exit(1)

    log.info("Loading YuNet face detector...")
    detector = FaceDetector(MODEL_PATH)

    log.info("Loading ArcFace embedding model...")
    extractor = EmbeddingExtractor()

    log.info("Loading SVM recognizer...")
    recognizer = FaceRecognizer(
        SVM_MODEL_PATH,
        LABEL_ENCODER_PATH,
        confidence_threshold=RECOGNITION_CONFIDENCE_THRESHOLD,
    )

    log.info("Loading anti-spoofing model (Silent-Face-Anti-Spoofing)...")
    antispoof = AntiSpoofDetector(
        ANTISPOOF_MODELS_DIR,
        threshold=LIVENESS_SCORE_THRESHOLD,
    )

    student_db = StudentDatabase(STUDENTS_CSV)
    attendance_log = AttendanceLog(ATTENDANCE_CSV)

    # Holds a just-marked result on screen briefly so the person gets
    # visible feedback before the loop moves on to the next frame.
    result_message: str | None = None
    result_hold_until: float = 0.0

    log.info("Starting real-time attendance. Press 'q' to quit.")

    with Camera(index=CAMERA_INDEX) as camera:
        cv2.namedWindow(WINDOW_NAME)

        while True:
            frame = camera.read()
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            face, status = detector.detect_single_face(frame)

            # --- Holding a just-finished result on screen briefly ---
            if result_message is not None:
                if time.time() < result_hold_until:
                    color = (0, 255, 0) if "Marked Present" in result_message else (0, 0, 255)
                    draw_hud(display, [result_message], color=color)
                    if face is not None:
                        x, y, w, h = face.box
                        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display, "Press Q to quit", (15, display.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    cv2.imshow(WINDOW_NAME, display)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                    continue
                else:
                    result_message = None

            if face is None:
                draw_hud(display, [STATUS_MESSAGES[status]], color=(0, 0, 255))
            else:
                x, y, w, h = face.box

                # --- Passive anti-spoofing check FIRST, before recognition ---
                liveness = antispoof.check(frame, face.box)

                if not liveness.is_live:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    draw_hud(
                        display,
                        ["SPOOF DETECTED - not a live face",
                         f"Liveness score: {liveness.score * 100:.1f}%"],
                        color=(0, 0, 255),
                    )
                else:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    face_crop = crop_and_resize_face(frame, face)
                    if face_crop is None:
                        draw_hud(display, ["Face crop failed"], color=(0, 0, 255))
                    else:
                        embedding = extractor.get_embedding(face_crop)
                        if embedding is None:
                            draw_hud(display, ["Embedding generation failed"], color=(0, 0, 255))
                        else:
                            student_id, confidence = recognizer.predict(embedding)

                            if student_id == "Unknown":
                                draw_hud(
                                    display,
                                    ["Unknown face", f"Confidence: {confidence * 100:.1f}%"],
                                    color=(0, 0, 255),
                                )
                            else:
                                # The SVM label is the folder name "<ROLL>_<Safe_Name>",
                                # where Safe_Name replaces spaces with underscores and
                                # may itself contain multiple underscores for multi-word
                                # names (e.g. "101_John_Doe_Smith"). Splitting on the
                                # first underscore alone would leave "name" as
                                # "Doe_Smith" instead of "Doe Smith".
                                #
                                # FIX: only take the roll number from the label, then
                                # look up the canonical, correctly-spaced name from
                                # students.csv (the source of truth) instead of trying
                                # to reconstruct it from the label string.
                                roll_no = student_id.split("_", 1)[0]
                                record = student_db.get_student(roll_no)
                                name = record["name"] if record else student_id.split("_", 1)[1].replace("_", " ")
                                student_id = roll_no

                                if attendance_log.already_marked_today(student_id):
                                    draw_hud(
                                        display,
                                        [f"{name} ({student_id})", "Attendance: Marked Present"],
                                        color=(0, 255, 0),
                                    )
                                else:
                                    marked_now = attendance_log.mark_attendance(student_id, name)
                                    status_text = "Marked Present" if marked_now else "Marked P"
                                    result_message = f"{name} ({student_id}) - {status_text}"
                                    result_hold_until = time.time() + RESULT_HOLD_SECONDS
                                    log.info(f"{name} ({student_id}) - {status_text} "
                                             f"[confidence={confidence * 100:.1f}%, "
                                             f"liveness={liveness.score * 100:.1f}%]")
                                    draw_hud(
                                        display,
                                        [f"{name} ({student_id})",
                                         f"Confidence: {confidence * 100:.1f}%",
                                         f"Liveness: {liveness.score * 100:.1f}%",
                                         f"Attendance: {status_text}"],
                                        color=(0, 255, 0),
                                    )

            cv2.putText(display, "Press Q to quit", (15, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()
    log.info("Attendance session ended.")


if __name__ == "__main__":
    main()