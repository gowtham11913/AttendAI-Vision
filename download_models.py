"""
download_models.py
-------------------
One-time setup helper: downloads the YuNet face detection ONNX model
from the OpenCV Zoo into models/face_detection_yunet_2023mar.onnx.

This script requires internet access. If it fails in a restricted
environment, manually download the file from:
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
and place it at: models/face_detection_yunet_2023mar.onnx

Run:
    python download_models.py
"""

import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TARGET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.isfile(TARGET_PATH):
        print(f"Model already exists at: {TARGET_PATH}")
        return

    print(f"Downloading YuNet model from:\n  {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, TARGET_PATH)
    except Exception as exc:
        print(f"\nERROR: Download failed ({exc}).")
        print("Please download the file manually from the URL above and place it at:")
        print(f"  {TARGET_PATH}")
        return

    print(f"Model saved to: {TARGET_PATH}")


if __name__ == "__main__":
    main()
