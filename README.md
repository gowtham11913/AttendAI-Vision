# Smart Attendance AI

AI-based attendance system using YuNet face detection, ArcFace (InsightFace
buffalo_l) embeddings, and an SVM classifier for real-time recognition.

## Setup

```bash
cd Smart_Attendance_AI
pip install -r requirements.txt
python download_models.py   # downloads models/face_detection_yunet_2023mar.onnx
```

If `download_models.py` cannot reach the internet from your machine, manually
download the file from the OpenCV Zoo and place it at:
`models/face_detection_yunet_2023mar.onnx`

> https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

The first run of any script that uses `EmbeddingExtractor` (generate_embeddings.py,
attendance_system.py) will automatically download the InsightFace `buffalo_l`
model pack (ArcFace) to `~/.insightface/models/` if it isn't already cached.

## Usage — run in order

### Phase 1: Register a student
```bash
python register_student.py
```
Enter Roll Number, Name, Department. Follow on-screen pose prompts
(FRONT, LEFT, RIGHT, UP, DOWN). Press SPACE to start each pose; a 3-2-1
countdown runs, then 10 images are auto-captured (300ms apart, 112x112
crops). 50 images total are saved to `dataset/students/<ROLL_NO>/`, and the
student is appended to `database/students.csv`.

### Phase 2: Generate embeddings
```bash
python generate_embeddings.py
```
Reads every student's images, generates a 512-D ArcFace embedding per
image, and saves all of a student's embeddings (shape `(N, 512)`) to
`embeddings/students/<ROLL_NO>.npy`.

### Phase 3: Train the classifier
```bash
python train_model.py
```
Loads all embeddings, trains a linear SVM (`probability=True`), and saves
`models/svm_model.pkl` and `models/label_encoder.pkl`. Prints student count,
image count, and accuracy.

### Phase 4: Run real-time attendance
```bash
python attendance_system.py
```
Opens the webcam, detects/recognizes faces live, displays roll number,
name, confidence, and attendance status, and marks attendance once per day
per student in `database/attendance.csv`. Press `q` to quit.

## Re-registering a new student later

Just re-run `register_student.py` for the new student, then re-run
`generate_embeddings.py` and `train_model.py` to retrain the classifier with
the updated roster. (Phase 4 always reflects the most recently trained model.)

## Project structure

```
Smart_Attendance_AI/
├── database/
│   ├── students.csv
│   └── attendance.csv
├── dataset/students/<ROLL_NO>/         (raw 112x112 pose images)
├── embeddings/students/<ROLL_NO>.npy   (per-student stacked ArcFace embeddings)
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   ├── svm_model.pkl
│   └── label_encoder.pkl
├── utils/
│   ├── camera.py        (Camera + YuNet FaceDetector + crop/resize)
│   ├── embedding.py      (ArcFace embedding extraction)
│   ├── recognition.py    (SVM prediction wrapper)
│   └── attendance.py     (students.csv / attendance.csv management)
├── register_student.py       (Phase 1)
├── generate_embeddings.py    (Phase 2)
├── train_model.py            (Phase 3)
├── attendance_system.py      (Phase 4)
├── download_models.py        (setup helper)
└── requirements.txt
```

## Notes / known limits

- Requires a connected webcam (index 0 by default).
- Training requires at least 2 registered students.
- Attendance confidence threshold defaults to 0.55 (`CONFIDENCE_THRESHOLD`
  in `attendance_system.py`); faces below this are shown as "Unknown" and
  not marked.
- Future features listed in the original spec (anti-spoofing, blink
  detection, masked-face recognition, web dashboard, SQLite/Firebase sync,
  etc.) are intentionally out of scope for this build.
