"""
embedding.py
------------
ArcFace embedding extraction utilities using InsightFace (buffalo_l).

Provides:
    - EmbeddingExtractor: loads the ArcFace recognition model via InsightFace's
      FaceAnalysis app (buffalo_l pack) and generates 512-D embeddings from
      already-cropped face images (112x112 BGR), as produced by camera.py.

Design notes:
    - ArcFace is used ONLY for embedding generation (per project rules).
    - We reuse YuNet for detection elsewhere; here, since input images are
      already tightly cropped 112x112 faces, we call the InsightFace
      recognition model directly via its `get_feat`/`model.get` API instead
      of re-running InsightFace's own detector, to avoid double detection
      and stay consistent with "YuNet only for detection".
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "insightface is required. Install it with: pip install insightface onnxruntime"
    ) from exc


EMBEDDING_DIM = 512


class EmbeddingExtractor:
    """
    Wraps the ArcFace recognition model (buffalo_l) from InsightFace.

    We initialize a FaceAnalysis app but disable its internal detector usage
    for our pipeline by feeding it already-cropped 112x112 face images and
    calling the underlying recognition model directly. This guarantees a
    single, consistent face-detection step (YuNet) across the whole project.
    """

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0):
        """
        Args:
            model_name: InsightFace model pack name. 'buffalo_l' includes the
                        ArcFace recognition model required by this project.
            ctx_id: 0 for GPU (if available via onnxruntime-gpu), -1 for CPU.
        """
        self.app = FaceAnalysis(name=model_name, providers=self._get_providers())
        # det_size is irrelevant to us since we only use the recognition
        # sub-model, but FaceAnalysis.prepare() must still be called to load
        # all sub-models including 'recognition'.
        self.app.prepare(ctx_id=ctx_id, det_size=(112, 112))

        self._rec_model = self.app.models.get("recognition")
        if self._rec_model is None:
            raise RuntimeError(
                "Could not load ArcFace recognition model from InsightFace "
                f"model pack '{model_name}'."
            )

    @staticmethod
    def _get_providers():
        # Prefer GPU execution provider if available, fall back to CPU.
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        return ["CPUExecutionProvider"]

    def get_embedding(self, face_image_112: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate a 512-D ArcFace embedding from a 112x112 BGR face crop.

        Args:
            face_image_112: np.ndarray of shape (112, 112, 3), BGR, uint8.

        Returns:
            1D float32 np.ndarray of shape (512,), L2-normalized, or None
            if the input is invalid.
        """
        if face_image_112 is None or face_image_112.shape[:2] != (112, 112):
            return None

        # ArcFaceONNX.get_feat expects a list of aligned face crops and
        # returns embeddings directly (the model performs its own internal
        # preprocessing: BGR->RGB, normalization, NCHW conversion).
        feats = self._rec_model.get_feat(face_image_112)
        embedding = np.asarray(feats, dtype=np.float32).reshape(-1)

        # L2-normalize for cosine-similarity-friendly, SVM-friendly features.
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def load_embedding(npy_path: str) -> np.ndarray:
        return np.load(npy_path)

    @staticmethod
    def save_embedding(embedding: np.ndarray, npy_path: str) -> None:
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, embedding)


def find_duplicate_face(
    candidate_embeddings: np.ndarray,
    embeddings_dir: str,
    similarity_threshold: float = 0.55,
):
    """
    Check whether a face already matches an existing registered student.
    Cosine similarity = dot product, since embeddings are L2-normalized.
    Returns (matched_roll_no, best_similarity) or (None, best_similarity).
    """
    if candidate_embeddings.ndim == 1:
        candidate_embeddings = candidate_embeddings.reshape(1, -1)

    if not os.path.isdir(embeddings_dir):
        return None, 0.0

    best_match_roll_no = None
    best_similarity = 0.0

    for filename in os.listdir(embeddings_dir):
        if not filename.endswith(".npy"):
            continue
        student_folder = os.path.splitext(filename)[0]

        roll_no = student_folder.split("_", 1)[0]

        existing = np.load(os.path.join(embeddings_dir, filename))
        if existing.ndim == 1:
            existing = existing.reshape(1, -1)

        similarities = candidate_embeddings @ existing.T
        max_sim = float(np.max(similarities))

        if max_sim > best_similarity:
            best_similarity = max_sim
            if max_sim >= similarity_threshold:
                # FIX: return the bare roll number (what StudentDatabase.get_student
                # looks up by), not the full "<ROLL>_<Name>" folder name. Previously
                # this was `best_match_roll_no = student_folder`, which meant the
                # downstream db.get_student(matched_roll_no) lookup always missed.
                best_match_roll_no = roll_no

    return best_match_roll_no, best_similarity