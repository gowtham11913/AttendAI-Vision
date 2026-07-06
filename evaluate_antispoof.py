"""
evaluate_antispoof.py
-----------------------
Accuracy report for the anti-spoofing (liveness) model, separate from
face-recognition accuracy.

This CANNOT be auto-generated the way evaluate_model.py is, because it
needs real test images of: (a) real, live faces, and (b) spoofing
attempts (printed photos, phone/monitor screens showing a face). There's
no way to manufacture honest test data for this - you have to capture it
yourself with your actual camera and lighting setup.

Setup (one-time):
    Create this folder structure and drop test images into it:

        reports/antispoof_test_set/
            real/      <- photos of actual live people in front of the camera
            fake/      <- photos/screenshots of printed photos or screens
                          held up to the camera

    Aim for at least 10-20 images in each folder, captured under the
    same lighting/camera conditions you'll actually use for attendance,
    for a meaningful number. Images can be .jpg or .png.

Run:
    python evaluate_antispoof.py

Output:
    - Overall accuracy, plus separate "real face acceptance rate" and
      "fake face rejection rate" (these matter more individually than
      one blended accuracy number - a security system that rejects 100%
      of fakes but also rejects 50% of real users is not "75% accurate"
      in any useful sense).
    - A text report saved to reports/antispoof_accuracy_report.txt
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from utils.camera import FaceDetector
from utils.antispoof import AntiSpoofDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
ANTISPOOF_MODELS_DIR = os.path.join(BASE_DIR, "models", "anti_spoof_models")
TEST_SET_DIR = os.path.join(BASE_DIR, "reports", "antispoof_test_set")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def evaluate_folder(folder: str, expected_live: bool, detector: FaceDetector,
                     antispoof: AntiSpoofDetector) -> dict:
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTENSIONS))
    results = {"correct": 0, "incorrect": 0, "no_face": 0, "details": []}

    for filename in files:
        path = os.path.join(folder, filename)
        image = cv2.imread(path)
        if image is None:
            continue

        face, status = detector.detect_single_face(image)
        if face is None:
            results["no_face"] += 1
            results["details"].append((filename, "NO FACE DETECTED", status))
            continue

        liveness = antispoof.check(image, face.box)
        predicted_live = liveness.is_live
        correct = predicted_live == expected_live

        if correct:
            results["correct"] += 1
        else:
            results["incorrect"] += 1

        results["details"].append((
            filename,
            "LIVE" if predicted_live else "SPOOF",
            f"score={liveness.score:.3f}",
        ))

    return results


def main():
    real_dir = os.path.join(TEST_SET_DIR, "real")
    fake_dir = os.path.join(TEST_SET_DIR, "fake")

    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)
        print("Test set folders were missing, so they've just been created at:")
        print(f"  {real_dir}")
        print(f"  {fake_dir}")
        print("\nAdd at least 10-20 images to each (real live-face photos in "
              "'real/', printed-photo or screen-replay photos in 'fake/'), "
              "then run this script again.")
        sys.exit(0)

    real_count = len([f for f in os.listdir(real_dir) if f.lower().endswith(VALID_EXTENSIONS)])
    fake_count = len([f for f in os.listdir(fake_dir) if f.lower().endswith(VALID_EXTENSIONS)])

    if real_count == 0 or fake_count == 0:
        print(f"Found {real_count} image(s) in real/ and {fake_count} in fake/.")
        print("Add test images to both folders before running this evaluation.")
        sys.exit(1)

    print("Loading YuNet face detector...")
    detector = FaceDetector(MODEL_PATH)

    print("Loading anti-spoofing model...")
    antispoof = AntiSpoofDetector(ANTISPOOF_MODELS_DIR)

    print(f"\nEvaluating {real_count} real-face image(s)...")
    real_results = evaluate_folder(real_dir, expected_live=True, detector=detector, antispoof=antispoof)

    print(f"Evaluating {fake_count} fake/spoof image(s)...")
    fake_results = evaluate_folder(fake_dir, expected_live=False, detector=detector, antispoof=antispoof)

    real_tested = real_results["correct"] + real_results["incorrect"]
    fake_tested = fake_results["correct"] + fake_results["incorrect"]

    real_acceptance_rate = (real_results["correct"] / real_tested * 100) if real_tested else 0.0
    fake_rejection_rate = (fake_results["correct"] / fake_tested * 100) if fake_tested else 0.0

    total_correct = real_results["correct"] + fake_results["correct"]
    total_tested = real_tested + fake_tested
    overall_accuracy = (total_correct / total_tested * 100) if total_tested else 0.0

    print("\n" + "=" * 60)
    print("ANTI-SPOOFING ACCURACY REPORT")
    print("=" * 60)
    print(f"Real-face acceptance rate : {real_acceptance_rate:.1f}%  "
          f"({real_results['correct']}/{real_tested} real faces correctly accepted)")
    print(f"Fake-face rejection rate  : {fake_rejection_rate:.1f}%  "
          f"({fake_results['correct']}/{fake_tested} spoof attempts correctly rejected)")
    print(f"Overall accuracy          : {overall_accuracy:.1f}%")
    if real_results["no_face"] or fake_results["no_face"]:
        print(f"\nNOTE: {real_results['no_face']} real + {fake_results['no_face']} fake "
              "image(s) had no face detected and were excluded from scoring - "
              "check those images' lighting/framing if this number is high.")
    print("=" * 60)

    print("\nWhat to do with these numbers:")
    print("- Low real-face acceptance rate -> genuine users getting rejected.")
    print("  Try LOWERING LIVENESS_SCORE_THRESHOLD in utils/antispoof.py.")
    print("- Low fake-face rejection rate -> spoofing attempts getting through.")
    print("  Try RAISING LIVENESS_SCORE_THRESHOLD in utils/antispoof.py.")
    print("- Both low at once usually means a lighting/camera quality issue")
    print("  rather than a threshold issue - recapture the test images with")
    print("  better lighting before retuning the threshold.")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "antispoof_accuracy_report.txt")
    with open(report_path, "w") as f:
        f.write("ANTI-SPOOFING ACCURACY REPORT\n")
        f.write(f"Real-face acceptance rate: {real_acceptance_rate:.1f}% "
                f"({real_results['correct']}/{real_tested})\n")
        f.write(f"Fake-face rejection rate: {fake_rejection_rate:.1f}% "
                f"({fake_results['correct']}/{fake_tested})\n")
        f.write(f"Overall accuracy: {overall_accuracy:.1f}%\n\n")
        f.write("Per-image details (real/):\n")
        for name, verdict, extra in real_results["details"]:
            f.write(f"  {name}: {verdict}  {extra}\n")
        f.write("\nPer-image details (fake/):\n")
        for name, verdict, extra in fake_results["details"]:
            f.write(f"  {name}: {verdict}  {extra}\n")
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
