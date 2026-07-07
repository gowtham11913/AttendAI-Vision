"""
app.py
------
AttendAI Vision web application.

Web registration mirrors register_student.py:
    FRONT 10 -> immediate duplicate-face check
    LEFT 10 -> RIGHT 10 -> UP 10 -> DOWN 10
    -> finalize student CSV -> background embeddings + SVM training

Run:
    python app.py
"""

from __future__ import annotations

import csv
import secrets
import shutil
import sys
import threading
import time
from config import (
    ARCFACE_VERIFICATION_THRESHOLD,
    LIVENESS_SCORE_THRESHOLD,
    RECOGNITION_CONFIDENCE_THRESHOLD,
    load_settings,
    save_settings,
    validate_config,
)
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import uvicorn
import pandas as pd

from fastapi import (
    FastAPI,
    File,
    Request,
    UploadFile,
    HTTPException,
)

from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from run_pipeline import PipelineError, STAGES, _run_stage


from utils.attendance import AttendanceLog, StudentDatabase
from utils.camera import FaceDetector, crop_and_resize_face
from utils.embedding import EmbeddingExtractor, find_duplicate_face
from utils.recognition import FaceRecognizer
from utils.antispoof import AntiSpoofDetector
from utils.identity_verifier import IdentityVerifier
from utils.logger import get_logger


# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "database"
STUDENTS_DATASET_DIR = PROJECT_ROOT / "dataset" / "students"
EMBEDDINGS_STUDENTS_DIR = PROJECT_ROOT / "embeddings" / "students"

SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

STUDENTS_CSV = DATABASE_DIR / "students.csv"
ATTENDANCE_CSV = DATABASE_DIR / "attendance.csv"

POSES = ["front", "left", "right", "up", "down"]
POSE_INSTRUCTIONS = {
    "front": "Look straight at the camera",
    "left": "Turn your head to the LEFT",
    "right": "Turn your head to the RIGHT",
    "up": "Tilt your head UP",
    "down": "Tilt your head DOWN",
}
IMAGES_PER_POSE = 10
REQUIRED_SAMPLE_COUNT = len(POSES) * IMAGES_PER_POSE
CAPTURE_INTERVAL_MS = 300
COUNTDOWN_SECONDS = 3
DUPLICATE_FACE_THRESHOLD = 0.55
REGISTRATION_SESSION_TTL = 15 * 60
ANTISPOOF_MODELS_DIR = MODELS_DIR / "anti_spoof_models"

log = get_logger("system")

REGISTRATION_SESSIONS: dict[str, dict] = {}
face_detector: FaceDetector | None = None
embedding_extractor: EmbeddingExtractor | None = None
face_recognizer: FaceRecognizer | None = None
antispoof_detector: AntiSpoofDetector | None = None
identity_verifier: IdentityVerifier | None = None

student_db = StudentDatabase(str(STUDENTS_CSV))
attendance_log = AttendanceLog(str(ATTENDANCE_CSV))


# Prevent simultaneous background retraining jobs from writing shared files.
_pipeline_lock = threading.Lock()

# In-memory status for post-registration background jobs.
# This is intentionally limited to webpage progress reporting.
PIPELINE_JOBS: dict[str, dict] = {}
_pipeline_jobs_lock = threading.Lock()

# Safe re-enrollment transactions. Old artifacts stay backed up until the
# complete embeddings -> SVM -> recognizer -> verifier pipeline succeeds.
REENROLL_TRANSACTIONS: dict[str, dict] = {}
_reenroll_transactions_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Models / helpers
# ---------------------------------------------------------------------------
class SettingsUpdateRequest(BaseModel):
    recognition_confidence_threshold: float = Field(
        ge=0.0,
        le=1.0,
    )

    liveness_score_threshold: float = Field(
        ge=0.0,
        le=1.0,
    )

    attendance_once_per_day: bool

    camera_index: int = Field(
        ge=0,
    )

    arcface_verification_threshold: float = Field(
        ge=0.0,
        le=1.0,
    )




class StudentUpdateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    department: str = Field(min_length=2, max_length=120)

class StudentValidationRequest(BaseModel):
    roll_no: str = Field(min_length=4, max_length=30)
    name: str = Field(min_length=2, max_length=100)
    department: str = Field(min_length=2, max_length=120)

    @field_validator("roll_no")
    @classmethod
    def validate_roll_no(cls, value: str) -> str:
        normalized = "".join(ch for ch in value.upper() if ch.isalnum())
        if len(normalized) < 4:
            raise ValueError("Invalid roll number.")
        return normalized

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = " ".join(value.strip().split()).title()
        if len(normalized) < 2:
            raise ValueError("Invalid student name.")
        return normalized

    @field_validator("department")
    @classmethod
    def validate_department(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if len(normalized) < 2:
            raise ValueError("Invalid department.")
        return normalized


def cleanup_expired_registration_sessions() -> None:
    now = time.time()
    expired = [
        token
        for token, session in REGISTRATION_SESSIONS.items()
        if now - session["created_at"] > REGISTRATION_SESSION_TTL
    ]
    for token in expired:
        session = REGISTRATION_SESSIONS.pop(token, None)
        # Do not automatically delete data here: a browser refresh may reconnect
        # before cleanup. Expired provisional folders can be cleaned manually.
        if session:
            log.info("Expired registration session removed: %s", token)


def build_student_folder_name(roll_no: str, name: str) -> str:
    safe_name = "_".join(name.split())
    return f"{roll_no}_{safe_name}"


def get_registered_student_count() -> int:
    try:
        return len(student_db.load())
    except Exception:
        log.error("Failed to count registered students", exc_info=True)
        return 0


def get_present_today_count() -> int:
    try:
        if not ATTENDANCE_CSV.is_file() or ATTENDANCE_CSV.stat().st_size == 0:
            return 0
        today = datetime.now().strftime("%Y-%m-%d")
        with ATTENDANCE_CSV.open("r", newline="", encoding="utf-8-sig") as fh:
            rows = csv.DictReader(fh)
            return sum(
                1
                for row in rows
                if row.get("date") == today
                and str(row.get("status", "")).lower() == "present"
            )
    except Exception:
        log.error("Failed to count today's attendance", exc_info=True)
        return 0


def get_session(token: str) -> dict | None:
    cleanup_expired_registration_sessions()
    return REGISTRATION_SESSIONS.get(token)


def get_pose_count(student_dir: Path, pose: str) -> int:
    return len(list(student_dir.glob(f"{pose}_*.jpg")))


def get_total_count(student_dir: Path) -> int:
    return sum(get_pose_count(student_dir, pose) for pose in POSES)


def expected_pose(session: dict) -> str | None:
    student_dir = STUDENTS_DATASET_DIR / session["student_folder"]
    for pose in POSES:
        if get_pose_count(student_dir, pose) < IMAGES_PER_POSE:
            return pose
    return None


def pose_state(session: dict) -> dict:
    student_dir = STUDENTS_DATASET_DIR / session["student_folder"]
    pose = expected_pose(session)
    return {
        "current_pose": pose,
        "pose_instruction": POSE_INSTRUCTIONS.get(pose, "Capture complete"),
        "pose_saved_count": get_pose_count(student_dir, pose) if pose else IMAGES_PER_POSE,
        "images_per_pose": IMAGES_PER_POSE,
        "saved_count": get_total_count(student_dir),
        "required_sample_count": REQUIRED_SAMPLE_COUNT,
        "capture_complete": pose is None,
        "duplicate_checked": bool(session.get("duplicate_checked", False)),
    }


def decode_upload(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def check_front_duplicate(student_dir: Path) -> tuple[str | None, float]:
    if embedding_extractor is None:
        raise RuntimeError("ArcFace embedding extractor is unavailable.")

    embeddings = []
    for image_path in sorted(student_dir.glob("front_*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        embedding = embedding_extractor.get_embedding(image)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        return None, 0.0

    candidate = np.vstack(embeddings)
    return find_duplicate_face(
        candidate,
        str(EMBEDDINGS_STUDENTS_DIR),
        similarity_threshold=DUPLICATE_FACE_THRESHOLD,
    )


def _set_pipeline_job(
    job_id: str,
    *,
    status: str,
    message: str,
    progress: int,
    error: str | None = None,
) -> None:
    """Update one background registration job atomically."""
    now = datetime.now().isoformat()

    with _pipeline_jobs_lock:
        job = PIPELINE_JOBS.setdefault(
            job_id,
            {
                "job_id": job_id,
                "status": "queued",
                "message": "Preparing registration...",
                "progress": 5,
                "error": None,
                "created_at": now,
                "updated_at": now,
            },
        )

        job["status"] = status
        job["message"] = message
        job["progress"] = max(0, min(100, int(progress)))
        job["error"] = error
        job["updated_at"] = now


def _get_pipeline_job(job_id: str) -> dict | None:
    """Return a defensive copy of one pipeline job."""
    with _pipeline_jobs_lock:
        job = PIPELINE_JOBS.get(job_id)
        return dict(job) if job is not None else None


def _rollback_reenrollment(job_id: str) -> None:
    """Restore the previous working biometric profile after pipeline failure."""
    global identity_verifier

    with _reenroll_transactions_lock:
        tx = REENROLL_TRANSACTIONS.get(job_id)

    if not tx:
        return

    canonical_dir = Path(tx["canonical_dir"])
    backup_dataset_dir = Path(tx["backup_dataset_dir"])
    canonical_embedding = Path(tx["canonical_embedding"])
    backup_embedding = Path(tx["backup_embedding"])
    roll_no = str(tx["roll_no"])

    try:
        if canonical_dir.exists():
            shutil.rmtree(canonical_dir, ignore_errors=True)
        if backup_dataset_dir.exists():
            backup_dataset_dir.rename(canonical_dir)

        canonical_embedding.unlink(missing_ok=True)
        if backup_embedding.exists():
            backup_embedding.rename(canonical_embedding)

        for emb in EMBEDDINGS_STUDENTS_DIR.glob(f"{roll_no}_*.npy"):
            if emb.resolve() != canonical_embedding.resolve():
                emb.unlink(missing_ok=True)

        if identity_verifier is not None:
            identity_verifier.reload()

        log.warning(
            "Re-enrollment rolled back to previous biometric profile: %s",
            roll_no,
        )
    finally:
        with _reenroll_transactions_lock:
            REENROLL_TRANSACTIONS.pop(job_id, None)


def _commit_reenrollment(job_id: str) -> None:
    """Delete backups only after the complete re-enrollment pipeline succeeds."""
    with _reenroll_transactions_lock:
        tx = REENROLL_TRANSACTIONS.pop(job_id, None)

    if not tx:
        return

    backup_dataset_dir = Path(tx["backup_dataset_dir"])
    backup_embedding = Path(tx["backup_embedding"])

    if backup_dataset_dir.exists():
        shutil.rmtree(backup_dataset_dir, ignore_errors=True)
    backup_embedding.unlink(missing_ok=True)

    log.info(
        "Re-enrollment transaction committed successfully: %s",
        tx["roll_no"],
    )


def _run_post_registration_pipeline(job_id: str):
    """
    Run the existing post-registration work while exposing its real
    current stage to the capture webpage.
    """
    global face_recognizer
    global identity_verifier

    log.info("Post-registration pipeline requested: %s", job_id)

    with _pipeline_lock:
        try:
            _set_pipeline_job(
                job_id,
                status="generating_embeddings",
                message="Generating face embeddings...",
                progress=30,
            )
            _run_stage("embed", STAGES["embed"], sys.executable)

            _set_pipeline_job(
                job_id,
                status="training_svm",
                message="Training recognition model...",
                progress=60,
            )
            _run_stage("train", STAGES["train"], sys.executable)

            if (
                not SVM_MODEL_PATH.is_file()
                or not LABEL_ENCODER_PATH.is_file()
            ):
                raise PipelineError(
                    "Training completed but model files are missing."
                )

            _set_pipeline_job(
                job_id,
                status="reloading_recognizer",
                message="Reloading recognition model...",
                progress=80,
            )

            current_settings = load_settings()

            new_recognizer = FaceRecognizer(
                str(SVM_MODEL_PATH),
                str(LABEL_ENCODER_PATH),
                confidence_threshold=float(
                    current_settings[
                        "recognition_confidence_threshold"
                    ]
                ),
            )
            face_recognizer = new_recognizer

            _set_pipeline_job(
                job_id,
                status="reloading_verifier",
                message="Reloading identity verifier...",
                progress=92,
            )

            if identity_verifier is None:
                identity_verifier = IdentityVerifier(
                    str(EMBEDDINGS_STUDENTS_DIR),
                    threshold=float(
                        current_settings[
                            "arcface_verification_threshold"
                        ]
                    ),
                )
            else:
                identity_verifier.reload()

            _commit_reenrollment(job_id)

            _set_pipeline_job(
                job_id,
                status="ready",
                message="Registration completed successfully.",
                progress=100,
            )

            log.info(
                "Post-registration pipeline completed successfully: %s",
                job_id,
            )

        except Exception as exc:
            log.exception(
                "Post-registration pipeline failed for %s: %s",
                job_id,
                exc,
            )

            try:
                _rollback_reenrollment(job_id)
            except Exception:
                log.exception(
                    "Re-enrollment rollback failed for job %s",
                    job_id,
                )

            _set_pipeline_job(
                job_id,
                status="failed",
                message="Registration processing failed.",
                progress=100,
                error=str(exc),
            )


def _run_post_deletion_pipeline(job_id: str):
    """
    Rebuild runtime identity state after deleting a student.

    Important edge case:
    if fewer than 2 student embedding classes remain, an SVM cannot be
    trained. In that case stale SVM/encoder files are removed and the
    recognizer is disabled instead of leaving the deleted identity active.
    """
    global face_recognizer
    global identity_verifier

    log.info("Post-deletion rebuild requested: %s", job_id)

    with _pipeline_lock:
        try:
            _set_pipeline_job(
                job_id,
                status="generating_embeddings",
                message="Checking remaining face embeddings...",
                progress=30,
            )
            _run_stage("embed", STAGES["embed"], sys.executable)

            remaining_embedding_files = sorted(
                EMBEDDINGS_STUDENTS_DIR.glob("*.npy")
            )

            if len(remaining_embedding_files) < 2:
                _set_pipeline_job(
                    job_id,
                    status="reloading_recognizer",
                    message="Disabling classifier because fewer than two students remain...",
                    progress=75,
                )

                SVM_MODEL_PATH.unlink(missing_ok=True)
                LABEL_ENCODER_PATH.unlink(missing_ok=True)
                face_recognizer = None

                _set_pipeline_job(
                    job_id,
                    status="reloading_verifier",
                    message="Reloading identity verifier...",
                    progress=92,
                )

                current_settings = load_settings()
                if identity_verifier is None:
                    identity_verifier = IdentityVerifier(
                        str(EMBEDDINGS_STUDENTS_DIR),
                        threshold=float(
                            current_settings["arcface_verification_threshold"]
                        ),
                    )
                else:
                    identity_verifier.reload()

                _set_pipeline_job(
                    job_id,
                    status="ready",
                    message=(
                        "Student deleted successfully. "
                        "Classifier disabled until at least two students are enrolled."
                    ),
                    progress=100,
                )
                return

            _set_pipeline_job(
                job_id,
                status="training_svm",
                message="Retraining recognition model...",
                progress=60,
            )
            _run_stage("train", STAGES["train"], sys.executable)

            _set_pipeline_job(
                job_id,
                status="reloading_recognizer",
                message="Reloading recognition model...",
                progress=80,
            )

            current_settings = load_settings()
            face_recognizer = FaceRecognizer(
                str(SVM_MODEL_PATH),
                str(LABEL_ENCODER_PATH),
                confidence_threshold=float(
                    current_settings["recognition_confidence_threshold"]
                ),
            )

            _set_pipeline_job(
                job_id,
                status="reloading_verifier",
                message="Reloading identity verifier...",
                progress=92,
            )

            if identity_verifier is None:
                identity_verifier = IdentityVerifier(
                    str(EMBEDDINGS_STUDENTS_DIR),
                    threshold=float(
                        current_settings["arcface_verification_threshold"]
                    ),
                )
            else:
                identity_verifier.reload()

            _set_pipeline_job(
                job_id,
                status="ready",
                message="Student deleted and recognition model rebuilt successfully.",
                progress=100,
            )

        except Exception as exc:
            log.exception(
                "Post-deletion rebuild failed for %s: %s",
                job_id,
                exc,
            )
            _set_pipeline_job(
                job_id,
                status="failed",
                message="Model rebuild after deletion failed.",
                progress=100,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_detector
    global embedding_extractor
    global face_recognizer
    global antispoof_detector
    global identity_verifier

    log.info("AttendAI Vision web application starting")
    try:
        # Load fresh persisted settings at application startup instead of
        # relying only on module-import-time effective constants.
        settings = load_settings()

        face_detector = FaceDetector(
            model_path=str(YUNET_MODEL_PATH),
            score_threshold=0.85,
            nms_threshold=0.3,
            top_k=50,
        )
        log.info("YuNet face detector loaded successfully")

        embedding_extractor = EmbeddingExtractor()
        log.info("ArcFace embedding model loaded successfully")
        if SVM_MODEL_PATH.is_file() and LABEL_ENCODER_PATH.is_file():
            face_recognizer = FaceRecognizer(
                str(SVM_MODEL_PATH),
                str(LABEL_ENCODER_PATH),
                confidence_threshold=float(
                    settings["recognition_confidence_threshold"]
                ),
            )
            log.info("SVM face recognizer loaded successfully")
        else:
            face_recognizer = None
            log.warning(
                "SVM recognizer not loaded because model files are missing"
            )

        antispoof_detector = AntiSpoofDetector(
            str(ANTISPOOF_MODELS_DIR),
            threshold=float(
                settings["liveness_score_threshold"]
            ),
        )

        log.info("Anti-spoofing model loaded successfully")

        # Fail-closed design: if the verifier cannot load, identity_verifier
        # stays None. process_attendance_frame() below treats a None
        # verifier as a 503 "not ready" condition, the same way a missing
        # face_recognizer or antispoof_detector is handled -- attendance
        # can never be marked with unknown-person rejection silently
        # bypassed.
        try:
            identity_verifier = IdentityVerifier(
                str(EMBEDDINGS_STUDENTS_DIR),
                threshold=float(
                    settings["arcface_verification_threshold"]
                ),
            )
            log.info("Identity verifier loaded successfully")
        except Exception:
            identity_verifier = None
            log.error(
                "Identity verifier failed to load. Attendance recognition "
                "will be unavailable (fail-closed) until this is resolved.",
                exc_info=True,
            )

        yield
    except Exception:
        log.critical("Application startup failed", exc_info=True)
        raise
    finally:
        face_detector = None
        embedding_extractor = None
        face_recognizer = None
        antispoof_detector = None
        identity_verifier = None

        log.info("AttendAI Vision web application stopped")


# ---------------------------------------------------------------------------
# App / routes
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    validate_config()
    app = FastAPI(
        title="AttendAI Vision",
        description="AI-Based Smart Attendance System",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))




    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        model_ready = (
            SVM_MODEL_PATH.is_file()
            and LABEL_ENCODER_PATH.is_file()
        )

        # Default values keep the dashboard safe
        # even if CSV files are empty or unavailable.
        total_students = 0
        present_today = 0
        total_attendance_records = 0
        recent_attendance = []

        try:
            # --------------------------------------------------------------
            # Registered students
            # --------------------------------------------------------------
            students_df = student_db.load()

            if not students_df.empty:
                total_students = len(students_df)

            # --------------------------------------------------------------
            # Attendance records
            # --------------------------------------------------------------
            attendance_df = attendance_log.load()

            if not attendance_df.empty:
                attendance_df = attendance_df.copy()

                total_attendance_records = len(attendance_df)

                today = datetime.now().strftime("%Y-%m-%d")

                # Count unique students present today.
                if (
                    "date" in attendance_df.columns
                    and "roll_no" in attendance_df.columns
                ):
                    today_df = attendance_df[
                        attendance_df["date"]
                        .astype(str)
                        .str.strip()
                        == today
                    ]

                    present_today = (
                        today_df["roll_no"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .replace("", pd.NA)
                        .dropna()
                        .nunique()
                    )

                # ----------------------------------------------------------
                # Recent attendance activity
                # ----------------------------------------------------------
                if (
                    "date" in attendance_df.columns
                    and "time" in attendance_df.columns
                ):
                    attendance_df["_sort_datetime"] = pd.to_datetime(
                        attendance_df["date"]
                        .astype(str)
                        .str.strip()
                        + " "
                        + attendance_df["time"]
                        .astype(str)
                        .str.strip(),
                        errors="coerce",
                    )

                    attendance_df = attendance_df.sort_values(
                        by="_sort_datetime",
                        ascending=False,
                        na_position="last",
                    )

                    attendance_df = attendance_df.drop(
                        columns=["_sort_datetime"]
                    )

                recent_attendance = (
                    attendance_df
                    .head(5)
                    .fillna("")
                    .to_dict(orient="records")
                )

        except Exception as exc:
            log.exception(
                f"Unable to load dashboard statistics: {exc}"
            )

        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "request": request,
                "page_title": "Overview",
                "active_page": "overview",
                "model_ready": model_ready,
                "current_year": datetime.now().year,

                # Real dashboard data
                "total_students": total_students,
                "present_today": present_today,
                "total_attendance_records": total_attendance_records,
                "recent_attendance": recent_attendance,
            },
        )








    @app.get("/register", response_class=HTMLResponse)
    async def register_page(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "request": request,
                "page_title": "Register Student",
                "active_page": "register",
                "current_year": datetime.now().year,
            },
        )
    

    @app.get("/students", response_class=HTMLResponse)
    async def students_page(request: Request):
        try:
            students_df = student_db.load()

            if students_df.empty:
                students = []
            else:
                students = students_df.fillna("").to_dict(
                    orient="records"
                )

        except Exception as exc:
            log.exception(
                f"Unable to load students page data: {exc}"
            )
            students = []

        return templates.TemplateResponse(
            request=request,
            name="students.html",
            context={
                "request": request,
                "page_title": "Students",
                "active_page": "students",
                "current_year": datetime.now().year,
                "students": students,
                "student_count": len(students),
            },
        )

    @app.get("/attendance", response_class=HTMLResponse)
    async def attendance_records_page(request: Request):
        try:
            attendance_df = attendance_log.load()

            if attendance_df.empty:
                attendance_records = []
            else:
                # Show newest attendance records first.
                attendance_df = attendance_df.copy()

                attendance_df["_sort_datetime"] = pd.to_datetime(
                    attendance_df["date"].astype(str)
                    + " "
                    + attendance_df["time"].astype(str),
                    errors="coerce",
                )

                attendance_df = attendance_df.sort_values(
                    by="_sort_datetime",
                    ascending=False,
                    na_position="last",
                )

                attendance_df = attendance_df.drop(
                    columns=["_sort_datetime"]
                )

                attendance_records = (
                    attendance_df
                    .fillna("")
                    .to_dict(orient="records")
                )

            today = datetime.now().strftime("%Y-%m-%d")

            today_count = sum(
                1
                for record in attendance_records
                if str(record.get("date", "")) == today
            )

            unique_students = len({
                str(record.get("roll_no", ""))
                for record in attendance_records
                if str(record.get("roll_no", "")).strip()
            })

        except Exception as exc:
            log.exception(
                f"Unable to load attendance records page data: {exc}"
            )

            attendance_records = []
            today_count = 0
            unique_students = 0

        return templates.TemplateResponse(
            request=request,
            name="attendance.html",
            context={
                "request": request,
                "page_title": "Attendance",
                "active_page": "attendance",
                "current_year": datetime.now().year,
                "attendance_records": attendance_records,
                "attendance_count": len(attendance_records),
                "today_count": today_count,
                "unique_students": unique_students,
            },
        )

    @app.get("/analytics", response_class=HTMLResponse)
    async def analytics_page(request: Request):
        total_records = 0
        unique_students = 0
        present_today = 0
        active_days = 0
        daily_labels = []
        daily_counts = []
        student_attendance = []

        try:
            attendance_df = attendance_log.load()

            if not attendance_df.empty:
                attendance_df = attendance_df.copy()

                # ----------------------------------------------------------
                # Normalize important fields
                # ----------------------------------------------------------
                if "roll_no" in attendance_df.columns:
                    attendance_df["roll_no"] = (
                        attendance_df["roll_no"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                    )

                if "name" in attendance_df.columns:
                    attendance_df["name"] = (
                        attendance_df["name"]
                        .astype(str)
                        .str.strip()
                    )

                if "date" in attendance_df.columns:
                    attendance_df["date"] = (
                        attendance_df["date"]
                        .astype(str)
                        .str.strip()
                    )

                # ----------------------------------------------------------
                # Core metrics
                # ----------------------------------------------------------
                total_records = len(attendance_df)

                if "roll_no" in attendance_df.columns:
                    unique_students = (
                        attendance_df["roll_no"]
                        .replace("", pd.NA)
                        .dropna()
                        .nunique()
                    )

                today = datetime.now().strftime("%Y-%m-%d")

                if (
                    "date" in attendance_df.columns
                    and "roll_no" in attendance_df.columns
                ):
                    today_df = attendance_df[
                        attendance_df["date"] == today
                    ]

                    present_today = (
                        today_df["roll_no"]
                        .replace("", pd.NA)
                        .dropna()
                        .nunique()
                    )

                if "date" in attendance_df.columns:
                    active_days = (
                        attendance_df["date"]
                        .replace("", pd.NA)
                        .dropna()
                        .nunique()
                    )

                # ----------------------------------------------------------
                # Daily attendance trend
                # One attendance count per unique student per date
                # ----------------------------------------------------------
                if (
                    "date" in attendance_df.columns
                    and "roll_no" in attendance_df.columns
                ):
                    daily_df = (
                        attendance_df[
                            ["date", "roll_no"]
                        ]
                        .dropna()
                        .drop_duplicates(
                            subset=["date", "roll_no"]
                        )
                        .groupby("date")["roll_no"]
                        .nunique()
                        .reset_index(name="count")
                    )

                    daily_df["_date_sort"] = pd.to_datetime(
                        daily_df["date"],
                        errors="coerce",
                    )

                    daily_df = daily_df.sort_values(
                        by="_date_sort",
                        ascending=True,
                        na_position="last",
                    )

                    # Keep the most recent 14 attendance dates.
                    daily_df = daily_df.tail(14)

                    daily_labels = (
                        daily_df["date"]
                        .astype(str)
                        .tolist()
                    )

                    daily_counts = (
                        daily_df["count"]
                        .astype(int)
                        .tolist()
                    )

                # ----------------------------------------------------------
                # Student-wise attendance ranking
                # ----------------------------------------------------------
                if (
                    "roll_no" in attendance_df.columns
                    and "name" in attendance_df.columns
                    and "date" in attendance_df.columns
                ):
                    student_df = (
                        attendance_df[
                            ["roll_no", "name", "date"]
                        ]
                        .dropna()
                        .drop_duplicates(
                            subset=["roll_no", "date"]
                        )
                        .groupby(
                            ["roll_no", "name"],
                            as_index=False,
                        )
                        .agg(
                            attendance_days=(
                                "date",
                                "nunique",
                            )
                        )
                        .sort_values(
                            by="attendance_days",
                            ascending=False,
                        )
                    )

                    student_attendance = (
                        student_df
                        .head(10)
                        .fillna("")
                        .to_dict(orient="records")
                    )

        except Exception as exc:
            log.exception(
                f"Unable to load analytics data: {exc}"
            )

        return templates.TemplateResponse(
            request=request,
            name="analytics.html",
            context={
                "request": request,
                "page_title": "Analytics",
                "active_page": "analytics",
                "current_year": datetime.now().year,
                "total_records": total_records,
                "unique_students": unique_students,
                "present_today": present_today,
                "active_days": active_days,
                "daily_labels": daily_labels,
                "daily_counts": daily_counts,
                "student_attendance": student_attendance,
            },
        )


    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        try:
            settings = load_settings()

        except Exception as exc:
            log.exception(
                f"Unable to load settings: {exc}"
            )

            raise HTTPException(
                status_code=500,
                detail="Unable to load settings.",
            )

        return templates.TemplateResponse(
            request=request,
            name="settings.html",
            context={
                "request": request,
                "page_title": "Settings",
                "active_page": "settings",
                "current_year": datetime.now().year,
                "settings": settings,
            },
        )


    @app.post("/api/settings")
    async def update_settings(
        payload: SettingsUpdateRequest,
    ):
        global face_recognizer
        global antispoof_detector
        global identity_verifier

        try:
            saved_settings = save_settings(
                recognition_confidence_threshold=(
                    payload.recognition_confidence_threshold
                ),
                liveness_score_threshold=(
                    payload.liveness_score_threshold
                ),
                attendance_once_per_day=(
                    payload.attendance_once_per_day
                ),
                camera_index=payload.camera_index,
                arcface_verification_threshold=(
                    payload.arcface_verification_threshold
                ),
            )

            # Apply recognition threshold immediately.
            if face_recognizer is not None:
                face_recognizer.confidence_threshold = float(
                    saved_settings[
                        "recognition_confidence_threshold"
                    ]
                )

            # Apply liveness threshold immediately.
            if antispoof_detector is not None:
                antispoof_detector.threshold = float(
                    saved_settings[
                        "liveness_score_threshold"
                    ]
                )

            # Apply verification threshold immediately.
            if identity_verifier is not None:
                identity_verifier.threshold = float(
                    saved_settings[
                        "arcface_verification_threshold"
                    ]
                )

            log.info(
                "Runtime settings updated successfully"
            )

            return {
                "success": True,
                "message": "Settings saved successfully.",
                "settings": saved_settings,
                "runtime_applied": {
                    "recognition_confidence_threshold": (
                        face_recognizer is not None
                    ),
                    "liveness_score_threshold": (
                        antispoof_detector is not None
                    ),
                    "arcface_verification_threshold": (
                        identity_verifier is not None
                    ),
                },
            }

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        except Exception as exc:
            log.exception(
                f"Unable to save settings: {exc}"
            )

            raise HTTPException(
                status_code=500,
                detail="Unable to save settings.",
            )





    @app.get("/live-attendance", response_class=HTMLResponse)
    async def live_attendance_page(request: Request):
        model_ready = (
            SVM_MODEL_PATH.is_file()
            and LABEL_ENCODER_PATH.is_file()
            and face_recognizer is not None
        )

        return templates.TemplateResponse(
            request=request,
            name="live_attendance.html",
            context={
                "request": request,
                "page_title": "Live Attendance",
                "active_page": "live_attendance",
                "current_year": datetime.now().year,
                "model_ready": model_ready,
                "confidence_threshold": int(
                    float(
                        load_settings()[
                            "recognition_confidence_threshold"
                        ]
                    ) * 100
                ),
            },
        )

    @app.post("/api/attendance/process-frame")
    async def process_attendance_frame(
        frame: UploadFile = File(...)
    ):
        # ---------------------------------------------------------
        # 1. Verify backend models are ready
        # ---------------------------------------------------------
        if face_detector is None:
            raise HTTPException(
                status_code=503,
                detail="Face detector is not ready.",
            )

        if embedding_extractor is None:
            raise HTTPException(
                status_code=503,
                detail="Embedding model is not ready.",
            )

        if face_recognizer is None:
            raise HTTPException(
                status_code=503,
                detail="Recognition model is not ready.",
            )

        if antispoof_detector is None:
            raise HTTPException(
                status_code=503,
                detail="Anti-spoofing model is not ready.",
            )

        if identity_verifier is None:
            # Fail-closed: unknown-person rejection cannot be silently
            # bypassed just because the verifier failed to load.
            raise HTTPException(
                status_code=503,
                detail="Identity verifier is not ready.",
            )

        # ---------------------------------------------------------
        # 2. Read uploaded browser frame
        # ---------------------------------------------------------
        try:
            raw_bytes = await frame.read()
        except Exception as exc:
            log.warning(f"Unable to read attendance frame: {exc}")
            raise HTTPException(
                status_code=400,
                detail="Unable to read camera frame.",
            )

        image = decode_upload(raw_bytes)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid camera frame.",
            )

        # ---------------------------------------------------------
        # 3. YuNet single-face validation
        # ---------------------------------------------------------
        face, face_status = face_detector.detect_single_face(image)

        if face is None:
            messages = {
                "no_face": "No face detected.",
                "multiple_faces": "Multiple faces detected.",
                "face_too_small": "Move closer to the camera.",
                "face_too_large": "Move farther from the camera.",
            }

            return JSONResponse(
                status_code=200,
                content={
                    "status": face_status,
                    "message": messages.get(
                        face_status,
                        "Face validation failed.",
                    ),
                    "recognized": False,
                    "attendance_marked": False,
                },
            )

        # ---------------------------------------------------------
        # 4. Passive anti-spoofing BEFORE recognition
        # ---------------------------------------------------------
        try:
            liveness = antispoof_detector.check(
                image,
                face.box,
            )
        except Exception as exc:
            log.exception(
                f"Anti-spoofing inference failed: {exc}"
            )

            raise HTTPException(
                status_code=500,
                detail="Liveness verification failed.",
            )

        liveness_score = float(liveness.score)

        if not liveness.is_live:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "spoof_detected",
                    "message": "Spoof detected. Live face required.",
                    "recognized": False,
                    "attendance_marked": False,
                    "liveness_score": round(
                        liveness_score,
                        4,
                    ),
                },
            )

        # ---------------------------------------------------------
        # 5. Crop face using existing YuNet crop utility
        # ---------------------------------------------------------
        face_crop = crop_and_resize_face(
            image,
            face,
        )

        if face_crop is None:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "face_crop_failed",
                    "message": "Unable to prepare face image.",
                    "recognized": False,
                    "attendance_marked": False,
                    "liveness_score": round(
                        liveness_score,
                        4,
                    ),
                },
            )

        # ---------------------------------------------------------
        # 6. ArcFace 512-D embedding
        # ---------------------------------------------------------
        try:
            embedding = embedding_extractor.get_embedding(
                face_crop
            )
        except Exception as exc:
            log.exception(
                f"Attendance embedding generation failed: {exc}"
            )

            raise HTTPException(
                status_code=500,
                detail="Embedding generation failed.",
            )

        if embedding is None:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "embedding_failed",
                    "message": "Unable to generate face embedding.",
                    "recognized": False,
                    "attendance_marked": False,
                    "liveness_score": round(
                        liveness_score,
                        4,
                    ),
                },
            )

        # ---------------------------------------------------------
        # 7. SVM recognition
        # ---------------------------------------------------------
        try:
            predicted_label, confidence = face_recognizer.predict(
                embedding
            )
        except Exception as exc:
            log.exception(
                f"SVM recognition failed: {exc}"
            )

            raise HTTPException(
                status_code=500,
                detail="Face recognition failed.",
            )

        confidence = float(confidence)

        if predicted_label == "Unknown":
            log.info(
                "Attendance: SVM confidence rejected candidate "
                "(confidence=%.4f)", confidence,
            )
            return JSONResponse(
                status_code=200,
                content={
                    "status": "unknown",
                    "message": "Face not recognized.",
                    "recognized": False,
                    "attendance_marked": False,
                    "confidence": round(
                        confidence,
                        4,
                    ),
                    "liveness_score": round(
                        liveness_score,
                        4,
                    ),
                },
            )

        # ---------------------------------------------------------
        # 7b. ArcFace cosine-similarity verification of the SVM
        #     candidate. The SVM is closed-set, so an unregistered
        #     person can still be assigned to a known class here --
        #     this step confirms the probe embedding is actually
        #     close to that specific candidate's stored samples
        #     before accepting the identity.
        # ---------------------------------------------------------
        try:
            verification = identity_verifier.verify(
                embedding,
                predicted_label,
            )
        except Exception as exc:
            log.exception(
                f"Identity verification failed: {exc}"
            )

            raise HTTPException(
                status_code=500,
                detail="Identity verification failed.",
            )

        log.info(
            "Attendance: SVM candidate=%s confidence=%.4f "
            "verification_similarity=%.4f threshold=%.4f decision=%s",
            predicted_label, confidence, verification.similarity,
            verification.threshold,
            "accepted" if verification.accepted else "rejected",
        )

        if not verification.accepted:
            log.info(
                "Attendance verification rejection reason=%s",
                verification.reason,
            )

        if not verification.accepted:
            # Do not leak which known student this was matched against --
            # only confidence/similarity scalars, no name or roll number.
            return JSONResponse(
                status_code=200,
                content={
                    "status": "unknown",
                    "message": "Face not recognized.",
                    "recognized": False,
                    "attendance_marked": False,
                    "confidence": round(
                        confidence,
                        4,
                    ),
                    "verification_similarity": round(
                        verification.similarity,
                        4,
                    ),
                    "liveness_score": round(
                        liveness_score,
                        4,
                    ),
                },
            )

        # ---------------------------------------------------------
        # 8. Recover roll number from SVM folder label
        #    Example:
        #    B230238EC_Gowtham -> B230238EC
        # ---------------------------------------------------------
        roll_no = str(predicted_label).split("_", 1)[0]

        student = student_db.get_student(roll_no)

        if student is None:
            log.error(
                f"Recognized label '{predicted_label}' "
                f"but no student record exists for '{roll_no}'"
            )

            return JSONResponse(
                status_code=200,
                content={
                    "status": "student_not_found",
                    "message": "Recognized identity is missing from database.",
                    "recognized": False,
                    "attendance_marked": False,
                    "confidence": round(
                        confidence,
                        4,
                    ),
                    "liveness_score": round(
                        liveness_score,
                        4,
                    ),
                },
            )

        # Student exists in database — extract name once for all
        # attendance responses and attendance-log operations below.
        student_name = str(student["name"])

        if not student_db.is_active(roll_no):
            return JSONResponse(status_code=200, content={
                "status": "inactive_student",
                "message": "Student account is inactive.",
                "recognized": True,
                "attendance_marked": False,
                "roll_no": roll_no,
                "name": student_name,
                "confidence": round(confidence, 4),
                "liveness_score": round(liveness_score, 4),
            })

        # ---------------------------------------------------------
        # 9. Per-day attendance rule

        # ---------------------------------------------------------
        # 9. Per-day attendance rule (honors the "attendance once per
        #    day" setting). Marking is atomic: mark_if_not_marked_today()
        #    checks and writes under a single lock acquisition, so two
        #    concurrent requests for the same student/day cannot both
        #    pass the check and both write a row.
        # ---------------------------------------------------------
        once_per_day = bool(load_settings().get("attendance_once_per_day", True))

        if once_per_day:
            marked_now = attendance_log.mark_if_not_marked_today(
                roll_no, student_name
            )
        else:
            # Always record a new attendance event, even if one already
            # exists today (e.g. entry/exit style logging).
            attendance_log.append_record(roll_no, student_name)
            marked_now = True

        if not marked_now:
            return JSONResponse(status_code=200, content={
                "status": "already_marked",
                "message": "Attendance already marked today.",
                "recognized": True,
                "attendance_marked": False,
                "already_marked": True,
                "roll_no": roll_no,
                "name": student_name,
                "confidence": round(confidence, 4),
                "verification_similarity": round(verification.similarity, 4),
                "liveness_score": round(liveness_score, 4),
            })

        log.info(
            f"Web attendance marked: "
            f"{student_name} ({roll_no}) "
            f"[confidence={confidence * 100:.1f}%, "
            f"verification_similarity={verification.similarity * 100:.1f}%, "
            f"liveness={liveness_score * 100:.1f}%]"
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "marked",
                "message": "Attendance marked successfully.",
                "recognized": True,
                "attendance_marked": True,
                "already_marked": False,
                "roll_no": roll_no,
                "name": student_name,
                "confidence": round(confidence, 4),
                "verification_similarity": round(verification.similarity, 4),
                "liveness_score": round(liveness_score, 4),
            },
        )


    @app.get("/register/capture/{registration_token}", response_class=HTMLResponse)
    async def capture_page(request: Request, registration_token: str):
        session = get_session(registration_token)
        if session is None:
            return templates.TemplateResponse(
                request=request,
                name="capture.html",
                context={
                    "request": request,
                    "page_title": "Face Capture",
                    "active_page": "register",
                    "current_year": datetime.now().year,
                    "registration_token": "",
                    "student": {"roll_no": "", "name": "Invalid Session", "department": ""},
                    "required_sample_count": REQUIRED_SAMPLE_COUNT,
                    "poses": POSES,
                    "images_per_pose": IMAGES_PER_POSE,
                    "capture_interval_ms": CAPTURE_INTERVAL_MS,
                    "countdown_seconds": COUNTDOWN_SECONDS,
                },
                status_code=404,
            )

        return templates.TemplateResponse(
            request=request,
            name="capture.html",
            context={
                "request": request,
                "page_title": "Face Capture",
                "active_page": "register",
                "current_year": datetime.now().year,
                "registration_token": registration_token,
                "student": session["student"],
                "required_sample_count": REQUIRED_SAMPLE_COUNT,
                "poses": POSES,
                "images_per_pose": IMAGES_PER_POSE,
                "capture_interval_ms": CAPTURE_INTERVAL_MS,
                "countdown_seconds": COUNTDOWN_SECONDS,
            },
        )

    @app.post("/api/register/validate")
    async def validate_student_registration(student: StudentValidationRequest):
        cleanup_expired_registration_sessions()

        try:
            if student_db.exists(student.roll_no):
                existing = student_db.get_student(student.roll_no) or {}
                return {
                    "success": False,
                    "duplicate": True,
                    "message": f"Roll number {student.roll_no} is already registered.",
                    "student": {
                        "roll_no": existing.get("roll_no", ""),
                        "name": existing.get("name", ""),
                    },
                }

            token = secrets.token_urlsafe(32)
            student_folder = build_student_folder_name(student.roll_no, student.name)
            student_dir = STUDENTS_DATASET_DIR / student_folder

            # A new registration session must not silently reuse stale samples.
            if student_dir.exists():
                shutil.rmtree(student_dir, ignore_errors=True)

            REGISTRATION_SESSIONS[token] = {
                "student": {
                    "roll_no": student.roll_no,
                    "name": student.name,
                    "department": student.department,
                },
                "student_folder": student_folder,
                "created_at": time.time(),
                "duplicate_checked": False,
            }

            return {
                "success": True,
                "duplicate": False,
                "message": "Student details validated successfully.",
                "registration_token": token,
                "student": REGISTRATION_SESSIONS[token]["student"],
            }

        except Exception:
            log.error("Registration validation failed", exc_info=True)
            return {
                "success": False,
                "duplicate": False,
                "message": "Unable to validate student details.",
            }

    @app.get("/api/register/capture/{registration_token}/state")
    async def capture_state(registration_token: str):
        session = get_session(registration_token)
        if session is None:
            return {
                "success": False,
                "session_valid": False,
                "status": "invalid_session",
                "message": "Registration session invalid or expired.",
            }
        return {
            "success": True,
            "session_valid": True,
            **pose_state(session),
        }

    @app.post("/api/register/capture/{registration_token}/validate-frame")
    async def validate_capture_frame(
        registration_token: str,
        frame: UploadFile = File(...),
    ):
        session = get_session(registration_token)
        if session is None:
            await frame.close()
            return {
                "success": False,
                "session_valid": False,
                "status": "invalid_session",
                "message": "Registration session invalid or expired.",
            }

        if face_detector is None:
            await frame.close()
            return {
                "success": False,
                "session_valid": True,
                "status": "detector_unavailable",
                "message": "Face detector is not available.",
            }

        try:
            image = decode_upload(await frame.read())
            if image is None:
                return {
                    "success": False,
                    "session_valid": True,
                    "status": "invalid_frame",
                    "message": "Unable to decode camera frame.",
                }

            detected_face, status = face_detector.detect_single_face(image)
            messages = {
                "ok": "One valid face detected.",
                "no_face": "No face detected.",
                "multiple_faces": "Multiple faces detected.",
                "face_too_small": "Move closer to the camera.",
                "face_too_large": "Move farther from the camera.",
            }

            state = pose_state(session)
            if status != "ok":
                return {
                    "success": True,
                    "session_valid": True,
                    "face_valid": False,
                    "status": status,
                    "message": messages.get(status, "Face validation failed."),
                    **state,
                }

            return {
                "success": True,
                "session_valid": True,
                "face_valid": True,
                "status": "ok",
                "message": messages["ok"],
                "face": {
                    "x": detected_face.x,
                    "y": detected_face.y,
                    "width": detected_face.w,
                    "height": detected_face.h,
                    "confidence": round(detected_face.confidence, 4),
                },
                **state,
            }
        except Exception:
            log.error("Camera frame validation failed", exc_info=True)
            return {
                "success": False,
                "session_valid": True,
                "status": "processing_error",
                "message": "Unable to process camera frame.",
            }
        finally:
            await frame.close()

    @app.post("/api/register/capture/{registration_token}/save-sample")
    async def save_capture_sample(
        registration_token: str,
        frame: UploadFile = File(...),
    ):
        session = get_session(registration_token)
        if session is None:
            await frame.close()
            return {
                "success": False,
                "session_valid": False,
                "saved": False,
                "status": "invalid_session",
                "message": "Registration session invalid or expired.",
            }

        if face_detector is None:
            await frame.close()
            return {
                "success": False,
                "session_valid": True,
                "saved": False,
                "status": "detector_unavailable",
                "message": "Face detector is not available.",
            }

        student_dir = STUDENTS_DATASET_DIR / session["student_folder"]
        current_pose = expected_pose(session)

        if current_pose is None:
            await frame.close()
            return {
                "success": True,
                "session_valid": True,
                "saved": False,
                "status": "capture_complete",
                **pose_state(session),
            }

        try:
            image = decode_upload(await frame.read())
            if image is None:
                return {
                    "success": False,
                    "session_valid": True,
                    "saved": False,
                    "status": "invalid_frame",
                    "message": "Unable to decode camera frame.",
                    **pose_state(session),
                }

            detected_face, status = face_detector.detect_single_face(image)
            if status != "ok":
                return {
                    "success": True,
                    "session_valid": True,
                    "saved": False,
                    "status": status,
                    "message": "Backend face validation failed; sample not saved.",
                    **pose_state(session),
                }

            face_crop = crop_and_resize_face(image, detected_face)
            if face_crop is None:
                return {
                    "success": False,
                    "session_valid": True,
                    "saved": False,
                    "status": "crop_failed",
                    "message": "Unable to crop face sample.",
                    **pose_state(session),
                }

            student_dir.mkdir(parents=True, exist_ok=True)

            pose_count = get_pose_count(student_dir, current_pose)
            if pose_count >= IMAGES_PER_POSE:
                return {
                    "success": True,
                    "session_valid": True,
                    "saved": False,
                    "status": "pose_complete",
                    **pose_state(session),
                }

            filename = f"{current_pose}_{pose_count + 1:02d}.jpg"
            sample_path = student_dir / filename
            if not cv2.imwrite(str(sample_path), face_crop):
                return {
                    "success": False,
                    "session_valid": True,
                    "saved": False,
                    "status": "write_failed",
                    "message": "Unable to save face sample.",
                    **pose_state(session),
                }

            log.info("Saved %s for %s", filename, session["student_folder"])

            # Exact CLI behavior: duplicate check immediately after FRONT 10.
            if current_pose == "front" and get_pose_count(student_dir, "front") == IMAGES_PER_POSE:
                matched_roll_no, similarity = check_front_duplicate(student_dir)

                if matched_roll_no is not None:
                    existing = student_db.get_student(matched_roll_no)
                    existing_name = existing["name"] if existing else "Unknown"

                    shutil.rmtree(student_dir, ignore_errors=True)
                    REGISTRATION_SESSIONS.pop(registration_token, None)

                    log.warning(
                        "Duplicate face blocked: %s matches %s (%s), similarity %.4f",
                        session["student"]["roll_no"],
                        matched_roll_no,
                        existing_name,
                        similarity,
                    )

                    return {
                        "success": False,
                        "session_valid": False,
                        "saved": True,
                        "status": "duplicate_face",
                        "message": (
                            f"Face already registered as {matched_roll_no} - "
                            f"{existing_name} ({similarity * 100:.1f}% similarity)."
                        ),
                        "matched_roll_no": matched_roll_no,
                        "matched_name": existing_name,
                        "similarity": similarity,
                        "saved_count": 0,
                        "capture_complete": False,
                    }

                session["duplicate_checked"] = True
                log.info(
                    "Front-pose duplicate check passed for %s; best similarity %.4f",
                    session["student"]["roll_no"],
                    similarity,
                )

            state = pose_state(session)
            return {
                "success": True,
                "session_valid": True,
                "saved": True,
                "status": "ok",
                "message": "Face sample saved successfully.",
                "filename": filename,
                **state,
            }

        except Exception:
            log.error("Sample save failed", exc_info=True)
            return {
                "success": False,
                "session_valid": True,
                "saved": False,
                "status": "processing_error",
                "message": "Unable to process camera frame.",
                **pose_state(session),
            }
        finally:
            await frame.close()

    @app.post("/api/register/capture/{registration_token}/finalize")
    async def finalize_registration(registration_token: str):
        session = get_session(registration_token)
        if session is None:
            return {
                "success": False,
                "session_valid": False,
                "saved": False,
                "status": "invalid_session",
                "message": "Registration session invalid or expired.",
            }

        student_dir = STUDENTS_DATASET_DIR / session["student_folder"]
        state = pose_state(session)

        if not state["capture_complete"] or state["saved_count"] != REQUIRED_SAMPLE_COUNT:
            return {
                "success": False,
                "session_valid": True,
                "saved": False,
                "status": "incomplete_capture",
                "message": (
                    f"Registration requires exactly {REQUIRED_SAMPLE_COUNT} samples. "
                    f"Currently found {state['saved_count']}."
                ),
                **state,
            }

        # FRONT duplicate check must have passed before commit.
        if not session.get("duplicate_checked", False):
            try:
                matched_roll_no, similarity = check_front_duplicate(student_dir)
                if matched_roll_no is not None:
                    existing = student_db.get_student(matched_roll_no)
                    existing_name = existing["name"] if existing else "Unknown"
                    shutil.rmtree(student_dir, ignore_errors=True)
                    REGISTRATION_SESSIONS.pop(registration_token, None)
                    return {
                        "success": False,
                        "session_valid": False,
                        "saved": False,
                        "status": "duplicate_face",
                        "message": (
                            f"Face already registered as {matched_roll_no} - "
                            f"{existing_name} ({similarity * 100:.1f}% similarity)."
                        ),
                    }
                session["duplicate_checked"] = True
            except Exception:
                log.error("Final duplicate safety check failed", exc_info=True)
                return {
                    "success": False,
                    "session_valid": True,
                    "saved": False,
                    "status": "duplicate_check_failed",
                    "message": "Unable to verify duplicate-face status.",
                }

        student = session["student"]

        try:
            is_reenroll = bool(session.get("reenroll", False))
            if student_db.exists(student["roll_no"]) and not is_reenroll:
                return {
                    "success": False,
                    "session_valid": True,
                    "saved": False,
                    "status": "duplicate_roll_no",
                    "message": f"Roll number {student['roll_no']} is already registered.",
                }

            if not is_reenroll:
                student_db.add_student(
                    student["roll_no"],
                    student["name"],
                    student["department"],
                    REQUIRED_SAMPLE_COUNT,
                )
            else:
                # Transactional re-enrollment swap. The old working dataset
                # and embedding are moved to hidden backups, not deleted.
                # Backups survive until the complete background pipeline is ready.
                canonical_folder = session.get(
                    "canonical_student_folder"
                ) or build_student_folder_name(
                    student["roll_no"],
                    student["name"],
                )
                canonical_dir = STUDENTS_DATASET_DIR / canonical_folder
                canonical_embedding = (
                    EMBEDDINGS_STUDENTS_DIR / f"{canonical_folder}.npy"
                )

                tx_suffix = secrets.token_hex(8)
                backup_dataset_dir = (
                    STUDENTS_DATASET_DIR
                    / f".reenroll_backup_{student['roll_no']}_{tx_suffix}"
                )
                backup_embedding = (
                    EMBEDDINGS_STUDENTS_DIR
                    / f".reenroll_backup_{student['roll_no']}_{tx_suffix}.npy"
                )

                if backup_dataset_dir.exists() or backup_embedding.exists():
                    raise RuntimeError(
                        "Unable to create safe re-enrollment backup."
                    )

                if canonical_dir.exists():
                    canonical_dir.rename(backup_dataset_dir)

                if canonical_embedding.exists():
                    canonical_embedding.rename(backup_embedding)

                try:
                    student_dir.rename(canonical_dir)
                except Exception:
                    if canonical_dir.exists():
                        shutil.rmtree(canonical_dir, ignore_errors=True)
                    if backup_dataset_dir.exists():
                        backup_dataset_dir.rename(canonical_dir)
                    if backup_embedding.exists():
                        backup_embedding.rename(canonical_embedding)
                    raise

                reenroll_tx = {
                    "roll_no": student["roll_no"],
                    "canonical_dir": str(canonical_dir),
                    "backup_dataset_dir": str(backup_dataset_dir),
                    "canonical_embedding": str(canonical_embedding),
                    "backup_embedding": str(backup_embedding),
                }

                log.info(
                    "Re-enrollment capture staged transactionally for %s",
                    student["roll_no"],
                )

            REGISTRATION_SESSIONS.pop(registration_token, None)

            job_id = secrets.token_urlsafe(18)

            if is_reenroll:
                with _reenroll_transactions_lock:
                    REENROLL_TRANSACTIONS[job_id] = reenroll_tx

            _set_pipeline_job(
                job_id,
                status="queued",
                message="Preparing registration...",
                progress=5,
            )

            threading.Thread(
                target=_run_post_registration_pipeline,
                args=(job_id,),
                name=f"post-registration-{student['roll_no']}",
                daemon=True,
            ).start()

            return {
                "success": True,
                "session_valid": False,
                "saved": True,
                "status": "registered",
                "message": (
                    "Images saved. Registration processing has started."
                ),
                "roll_no": student["roll_no"],
                "image_count": REQUIRED_SAMPLE_COUNT,
                "pipeline_job_id": job_id,
            }

        except Exception:
            log.error("Registration finalize failed", exc_info=True)
            return {
                "success": False,
                "session_valid": True,
                "saved": False,
                "status": "processing_error",
                "message": "Unable to finalize registration.",
            }

    @app.get("/api/register/pipeline/{job_id}/status")
    async def registration_pipeline_status(job_id: str):
        job = _get_pipeline_job(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Registration pipeline job not found.",
            )

        return {
            "success": True,
            **job,
        }


    @app.post("/api/register/pipeline/{job_id}/retry")
    async def retry_registration_pipeline(job_id: str):
        job = _get_pipeline_job(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Registration pipeline job not found.",
            )

        if job["status"] != "failed":
            raise HTTPException(
                status_code=409,
                detail="Only failed jobs can be retried.",
            )

        _set_pipeline_job(
            job_id,
            status="queued",
            message="Retrying registration processing...",
            progress=5,
        )

        threading.Thread(
            target=_run_post_registration_pipeline,
            args=(job_id,),
            name=f"post-registration-retry-{job_id[:8]}",
            daemon=True,
        ).start()

        return {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "message": "Retry started.",
        }


    @app.put("/api/students/{roll_no}")
    async def update_student(
        roll_no: str,
        payload: StudentUpdateRequest,
    ):
        """
        Update student metadata while keeping all identity artifacts
        synchronized with the ROLLNO_SafeName convention.

        If the name changes:
            1. Rename dataset folder.
            2. Rename per-student embedding file.
            3. Update students.csv.
            4. Rebuild SVM classifier.
            5. Reload FaceRecognizer.
            6. Reload IdentityVerifier.

        Department-only changes do not require model retraining.
        """
        normalized_roll_no = str(roll_no).strip().upper()

        new_name = " ".join(
            payload.name.strip().split()
        ).title()

        new_department = " ".join(
            payload.department.strip().split()
        )

        existing_student = student_db.get_student(
            normalized_roll_no
        )

        if existing_student is None:
            raise HTTPException(
                status_code=404,
                detail="Student not found.",
            )

        old_name = str(existing_student["name"]).strip()

        old_folder_name = build_student_folder_name(
            normalized_roll_no,
            old_name,
        )

        new_folder_name = build_student_folder_name(
            normalized_roll_no,
            new_name,
        )

        name_changed = old_folder_name != new_folder_name

        old_dataset_dir = (
            STUDENTS_DATASET_DIR / old_folder_name
        )

        new_dataset_dir = (
            STUDENTS_DATASET_DIR / new_folder_name
        )

        old_embedding_path = (
            EMBEDDINGS_STUDENTS_DIR
            / f"{old_folder_name}.npy"
        )

        new_embedding_path = (
            EMBEDDINGS_STUDENTS_DIR
            / f"{new_folder_name}.npy"
        )

        dataset_renamed = False
        embedding_renamed = False

        try:
            # ------------------------------------------------------
            # 1. Synchronize identity artifact names first
            # ------------------------------------------------------
            if name_changed:

                # Safety: never overwrite another existing dataset.
                if (
                    new_dataset_dir.exists()
                    and new_dataset_dir.resolve()
                    != old_dataset_dir.resolve()
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Cannot rename student because the "
                            "target dataset folder already exists."
                        ),
                    )

                # Safety: never overwrite another embedding file.
                if (
                    new_embedding_path.exists()
                    and new_embedding_path.resolve()
                    != old_embedding_path.resolve()
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Cannot rename student because the "
                            "target embedding file already exists."
                        ),
                    )

                # Rename dataset folder.
                if old_dataset_dir.exists():
                    old_dataset_dir.rename(
                        new_dataset_dir
                    )
                    dataset_renamed = True

                # Rename embedding file.
                if old_embedding_path.exists():
                    old_embedding_path.rename(
                        new_embedding_path
                    )
                    embedding_renamed = True

            # ------------------------------------------------------
            # 2. Update students.csv only after artifact rename
            # ------------------------------------------------------
            updated_student = student_db.update_student(
                normalized_roll_no,
                new_name,
                new_department,
            )

            # ------------------------------------------------------
            # 3. Department-only edit needs no model rebuild
            # ------------------------------------------------------
            if not name_changed:
                return {
                    "success": True,
                    "message": "Student updated successfully.",
                    "student": updated_student,
                    "pipeline_job_id": None,
                    "model_rebuild_required": False,
                }

            # ------------------------------------------------------
            # 4. Name changed:
            #    rebuild classifier because SVM class label changed
            # ------------------------------------------------------
            job_id = secrets.token_urlsafe(18)

            _set_pipeline_job(
                job_id,
                status="queued",
                message=(
                    "Student updated. "
                    "Rebuilding recognition model..."
                ),
                progress=5,
            )

            threading.Thread(
                target=_run_post_registration_pipeline,
                args=(job_id,),
                name=f"edit-retrain-{normalized_roll_no}",
                daemon=True,
            ).start()

            log.info(
                "Student identity renamed: %s -> %s",
                old_folder_name,
                new_folder_name,
            )

            return {
                "success": True,
                "message": (
                    "Student updated successfully. "
                    "Recognition model rebuild started."
                ),
                "student": updated_student,
                "pipeline_job_id": job_id,
                "model_rebuild_required": True,
            }

        except HTTPException:
            # ------------------------------------------------------
            # Roll back filesystem changes
            # ------------------------------------------------------
            if (
                embedding_renamed
                and new_embedding_path.exists()
                and not old_embedding_path.exists()
            ):
                try:
                    new_embedding_path.rename(
                        old_embedding_path
                    )
                except Exception:
                    log.exception(
                        "Embedding rollback failed for %s",
                        normalized_roll_no,
                    )

            if (
                dataset_renamed
                and new_dataset_dir.exists()
                and not old_dataset_dir.exists()
            ):
                try:
                    new_dataset_dir.rename(
                        old_dataset_dir
                    )
                except Exception:
                    log.exception(
                        "Dataset rollback failed for %s",
                        normalized_roll_no,
                    )

            raise

        except ValueError as exc:
            # ------------------------------------------------------
            # Roll back filesystem changes if CSV update failed
            # ------------------------------------------------------
            if (
                embedding_renamed
                and new_embedding_path.exists()
                and not old_embedding_path.exists()
            ):
                try:
                    new_embedding_path.rename(
                        old_embedding_path
                    )
                except Exception:
                    log.exception(
                        "Embedding rollback failed for %s",
                        normalized_roll_no,
                    )

            if (
                dataset_renamed
                and new_dataset_dir.exists()
                and not old_dataset_dir.exists()
            ):
                try:
                    new_dataset_dir.rename(
                        old_dataset_dir
                    )
                except Exception:
                    log.exception(
                        "Dataset rollback failed for %s",
                        normalized_roll_no,
                    )

            raise HTTPException(
                status_code=404,
                detail=str(exc),
            )

        except Exception as exc:
            # ------------------------------------------------------
            # Roll back filesystem changes
            # ------------------------------------------------------
            if (
                embedding_renamed
                and new_embedding_path.exists()
                and not old_embedding_path.exists()
            ):
                try:
                    new_embedding_path.rename(
                        old_embedding_path
                    )
                except Exception:
                    log.exception(
                        "Embedding rollback failed for %s",
                        normalized_roll_no,
                    )

            if (
                dataset_renamed
                and new_dataset_dir.exists()
                and not old_dataset_dir.exists()
            ):
                try:
                    new_dataset_dir.rename(
                        old_dataset_dir
                    )
                except Exception:
                    log.exception(
                        "Dataset rollback failed for %s",
                        normalized_roll_no,
                    )

            log.exception(
                "Unable to update student %s: %s",
                normalized_roll_no,
                exc,
            )

            raise HTTPException(
                status_code=500,
                detail="Unable to update student safely.",
            )

    @app.post("/api/students/{roll_no}/deactivate")
    async def deactivate_student(roll_no: str):
        try:
            return {"success": True, "student": student_db.set_active(roll_no.upper(), False)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/api/students/{roll_no}/activate")
    async def activate_student(roll_no: str):
        try:
            return {"success": True, "student": student_db.set_active(roll_no.upper(), True)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.delete("/api/students/{roll_no}")
    async def delete_student(roll_no: str):
        global identity_verifier

        roll_no = roll_no.upper()
        student = student_db.get_student(roll_no)

        if student is None:
            raise HTTPException(
                status_code=404,
                detail="Student not found.",
            )

        try:
            # 1. Remove all captured face datasets for this roll number.
            for folder in STUDENTS_DATASET_DIR.glob(
                f"{roll_no}_*"
            ):
                if folder.is_dir():
                    shutil.rmtree(
                        folder,
                        ignore_errors=True,
                    )

            # Also remove any abandoned temporary re-enrollment capture.
            for folder in STUDENTS_DATASET_DIR.glob(
                f".reenroll_{roll_no}_*"
            ):
                if folder.is_dir():
                    shutil.rmtree(
                        folder,
                        ignore_errors=True,
                    )

            # 2. Remove all stored per-student embeddings.
            for emb in EMBEDDINGS_STUDENTS_DIR.glob(
                f"{roll_no}_*.npy"
            ):
                emb.unlink(missing_ok=True)

            # 3. Remove all attendance history for this student.
            # Write through a temporary file so the CSV is not left half-written.
            attendance_df = attendance_log.load()

            if not attendance_df.empty and "roll_no" in attendance_df.columns:
                remaining_attendance = attendance_df[
                    attendance_df["roll_no"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    != roll_no
                ].copy()

                temp_attendance_csv = ATTENDANCE_CSV.with_suffix(
                    ".delete.tmp"
                )
                remaining_attendance.to_csv(
                    temp_attendance_csv,
                    index=False,
                )
                temp_attendance_csv.replace(
                    ATTENDANCE_CSV
                )

            # 4. Remove student master record.
            if not student_db.delete_student(roll_no):
                raise HTTPException(
                    status_code=404,
                    detail="Student not found.",
                )

            # 5. Reload verifier immediately so deleted embeddings disappear
            # from verification memory before the SVM rebuild completes.
            if identity_verifier is not None:
                identity_verifier.reload()

            # 6. Rebuild classifier asynchronously. The dedicated deletion
            # pipeline safely handles the <2 remaining students edge case.
            job_id = secrets.token_urlsafe(18)
            _set_pipeline_job(
                job_id,
                status="queued",
                message="Rebuilding model after deletion...",
                progress=5,
            )

            threading.Thread(
                target=_run_post_deletion_pipeline,
                args=(job_id,),
                name=f"delete-retrain-{roll_no}",
                daemon=True,
            ).start()

            return {
                "success": True,
                "message": (
                    "Student, attendance history, captured images, and "
                    "embeddings deleted. Model rebuild started."
                ),
                "pipeline_job_id": job_id,
            }

        except HTTPException:
            raise
        except Exception as exc:
            log.exception(
                "Unable to fully delete student %s: %s",
                roll_no,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail="Unable to fully delete student.",
            )


    @app.post("/api/students/{roll_no}/reenroll")
    async def reenroll_student(roll_no: str):
        roll_no = roll_no.upper()
        student = student_db.get_student(roll_no)
        if student is None:
            raise HTTPException(status_code=404, detail="Student not found.")

        # SAFE RE-ENROLLMENT:
        # Keep the current working biometric profile untouched while the new
        # 50 samples are being captured. The swap happens only after capture
        # completes successfully in finalize_registration().
        token = secrets.token_urlsafe(32)
        canonical_folder = build_student_folder_name(
            roll_no, str(student["name"])
        )
        temporary_folder = (
            f".reenroll_{roll_no}_{secrets.token_hex(8)}"
        )

        temporary_dir = STUDENTS_DATASET_DIR / temporary_folder
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)

        REGISTRATION_SESSIONS[token] = {
            "student": {
                "roll_no": roll_no,
                "name": str(student["name"]),
                "department": str(student["department"]),
            },
            "student_folder": temporary_folder,
            "canonical_student_folder": canonical_folder,
            "created_at": time.time(),
            "duplicate_checked": True,
            "reenroll": True,
        }

        return {
            "success": True,
            "message": "Re-enrollment session created. Existing biometric data remains active until the new capture completes.",
            "registration_token": token,
            "capture_url": f"/register/capture/{token}",
        }

    @app.get("/api/system/status")
    async def system_status():
        svm_ready = SVM_MODEL_PATH.is_file()
        encoder_ready = LABEL_ENCODER_PATH.is_file()
        return {
            "status": "online",
            "model_ready": svm_ready and encoder_ready,
            "svm_model": svm_ready,
            "label_encoder": encoder_ready,
            "students_database": STUDENTS_CSV.is_file(),
            "attendance_database": ATTENDANCE_CSV.is_file(),
            "student_count": get_registered_student_count(),
            "present_today": get_present_today_count(),
            "timestamp": datetime.now().isoformat(),
        }

    return app


app = create_app()


if __name__ == "__main__":
    try:
        log.info("Starting AttendAI Vision server")
        uvicorn.run(
            "app:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
        )
    except KeyboardInterrupt:
        log.warning("Web server stopped by user")
    except Exception:
        log.critical("Web server failed", exc_info=True)
        sys.exit(1)