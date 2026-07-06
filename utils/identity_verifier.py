"""
identity_verifier.py
---------------------
Hybrid identity verification layer.

Provides ArcFace cosine-similarity verification of an SVM-predicted
candidate identity, to reject unregistered people that an SVM
classifier -- being closed-set -- would otherwise incorrectly assign
to some known class.

Given:
    - a probe embedding (512-D, from the live frame)
    - an SVM-predicted candidate label (e.g. "B230238EC_Gowtham")

This verifier:
    1. Loads the stored (N, 512) embedding array for that exact
       candidate from embeddings/students/<label>.npy
    2. L2-normalizes both probe and stored embeddings defensively.
       (Stored embeddings are already normalized by EmbeddingExtractor
       at write time, but this module does not assume that from the
       outside -- it re-normalizes so correctness does not depend on
       another module's internal behavior.)
    3. Computes cosine similarity between the probe and every stored
       embedding belonging to that candidate only (never against the
       full population -- this is candidate-specific verification,
       not open-set search).
    4. Takes the mean of the top-k highest similarities rather than a
       single best match or an average over all samples.
    5. Accepts the candidate only if that top-k mean similarity is
       >= the configured threshold.

k-selection rationale (project requirement: "do not compare with a
single embedding; document the decision"):
    Each student has up to 50 stored embeddings (5 poses x 10 images:
    front/left/right/up/down). Comparing the probe against a single
    stored sample is fragile -- that one sample could be a blink,
    motion blur, or an unusual angle, producing a false accept or
    reject. Averaging over ALL 50 samples is also not ideal: off-axis
    poses (up/down/left/right) are legitimately less similar to a
    frontal live-attendance probe than front-pose samples are, so a
    full average dilutes a genuine match with real pose variance.

    Top-k mean is a middle ground: it rewards a candidate whose
    embedding is consistently close across *several* stored samples
    (resistant to any one bad or unusually-posed sample skewing the
    result), without needing every one of the 50 samples to agree.

    k = min(TOP_K, num_available_embeddings), TOP_K = 5.
    5 was chosen because it is roughly "half a pose bucket" (10
    images per pose) -- enough samples to smooth out per-image noise
    while still requiring a real, repeated match rather than one
    lucky frame. This is a provisional heuristic, not a tuned value;
    revisit once real accept/reject similarity distributions have
    been logged in production (see arcface_verification_threshold in
    config.py for the same caveat on the threshold itself).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from utils.logger import get_logger

log = get_logger("identity_verifier")

TOP_K = 5


class VerificationResult(NamedTuple):
    accepted: bool
    candidate_label: str
    similarity: float
    threshold: float
    reason: str


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of an (N, D) matrix, safely handling zero norms."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


class IdentityVerifier:
    """
    Loads per-student ArcFace embedding arrays from
    embeddings/students/<label>.npy and verifies an SVM candidate
    prediction via top-k mean cosine similarity.

    Stateless with respect to the SVM / label encoder: it only needs
    the embeddings directory and works purely off the candidate label
    string handed to it by the caller. That label must match a .npy
    filename stem exactly (e.g. "B230238EC_Gowtham"), which is the
    same convention generate_embeddings.py and train_model.py already
    use, so no extra mapping step is required.
    """

    def __init__(self, embeddings_dir: str | Path, threshold: float = 0.45):
        """
        Args:
            embeddings_dir: path to embeddings/students/
            threshold: minimum top-k mean cosine similarity required
                       to accept the SVM-predicted candidate.

        PROVISIONAL THRESHOLD WARNING:
            0.45 is a configurable starting value, not a calibrated
            deployment threshold. The correct threshold depends on this
            project's camera, preprocessing, ArcFace model, registration
            samples, pose variation, and operating environment.

            Calibrate it from genuine and impostor similarity distributions
            collected with this actual deployment. In normal operation this
            default is overridden by the
            "arcface_verification_threshold" runtime setting.
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.threshold = float(threshold)
        # label -> (N, 512) L2-normalized embedding matrix
        self._cache: dict[str, np.ndarray] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load (or reload) every student's embedding file into memory."""
        cache: dict[str, np.ndarray] = {}

        if not self.embeddings_dir.is_dir():
            log.warning(
                "Identity verifier: embeddings directory not found at %s. "
                "All candidates will fail closed until it exists.",
                self.embeddings_dir,
            )
            self._cache = cache
            return

        loaded = 0
        failed = 0

        for npy_path in sorted(self.embeddings_dir.glob("*.npy")):
            label = npy_path.stem
            try:
                arr = np.load(npy_path)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.ndim != 2 or arr.shape[1] != 512:
                    log.error(
                        "Identity verifier: malformed embedding file '%s' "
                        "(shape=%s); skipping. That candidate will fail "
                        "closed until re-registered or fixed.",
                        npy_path.name, arr.shape,
                    )
                    failed += 1
                    continue
                arr = arr.astype(np.float32)

                if not np.isfinite(arr).all():
                    log.error(
                        "Identity verifier: non-finite values found in '%s'; "
                        "skipping. That candidate will fail closed.",
                        npy_path.name,
                    )
                    failed += 1
                    continue

                cache[label] = _l2_normalize_rows(arr)
                loaded += 1
            except Exception:
                log.exception(
                    "Identity verifier: failed to load embedding file "
                    "'%s'; skipping. That candidate will fail closed.",
                    npy_path.name,
                )
                failed += 1

        self._cache = cache
        log.info(
            "Identity verifier: loaded %d student embedding set(s) "
            "(%d failed) from %s",
            loaded, failed, self.embeddings_dir,
        )

    def reload(self) -> None:
        """Refresh the in-memory cache from disk (e.g. after registration)."""
        self._load_all()

    def verify(
        self,
        probe_embedding: np.ndarray,
        candidate_label: str,
    ) -> VerificationResult:
        """
        Verify a probe embedding against the stored embeddings for one
        SVM-predicted candidate identity only.

        Args:
            probe_embedding: 512-D embedding from the live frame.
            candidate_label: SVM-predicted label; must match a .npy
                              filename stem (e.g. "B230238EC_Gowtham").

        Returns:
            VerificationResult. Fails closed (accepted=False) on any
            missing or malformed data rather than raising, so a bad
            embedding file can never crash the attendance endpoint.
        """
        stored = self._cache.get(candidate_label)

        if stored is None or stored.shape[0] == 0:
            log.warning(
                "Identity verifier: no stored embeddings for candidate "
                "'%s'; failing closed.", candidate_label,
            )
            return VerificationResult(
                accepted=False,
                candidate_label=candidate_label,
                similarity=0.0,
                threshold=self.threshold,
                reason="no_stored_embeddings",
            )

        try:
            probe = np.asarray(
                probe_embedding,
                dtype=np.float32,
            ).reshape(-1)

            if probe.shape[0] != 512:
                log.warning(
                    "Identity verifier: invalid probe dimension for "
                    "candidate '%s' (dimension=%d); failing closed.",
                    candidate_label,
                    probe.shape[0],
                )
                return VerificationResult(
                    accepted=False,
                    candidate_label=candidate_label,
                    similarity=0.0,
                    threshold=self.threshold,
                    reason="invalid_probe_dimension",
                )

            if not np.isfinite(probe).all():
                log.warning(
                    "Identity verifier: non-finite probe values for "
                    "candidate '%s'; failing closed.",
                    candidate_label,
                )
                return VerificationResult(
                    accepted=False,
                    candidate_label=candidate_label,
                    similarity=0.0,
                    threshold=self.threshold,
                    reason="non_finite_probe",
                )

            probe = _l2_normalize_vector(probe)

            if np.linalg.norm(probe) == 0:
                log.warning(
                    "Identity verifier: zero-norm probe for candidate "
                    "'%s'; failing closed.",
                    candidate_label,
                )
                return VerificationResult(
                    accepted=False,
                    candidate_label=candidate_label,
                    similarity=0.0,
                    threshold=self.threshold,
                    reason="zero_norm_probe",
                )

            similarities = stored @ probe  # (N,) cosine similarities
            k = min(TOP_K, similarities.shape[0])
            top_k_similarity = float(np.mean(np.sort(similarities)[-k:]))
        except Exception:
            log.exception(
                "Identity verifier: similarity computation failed for "
                "candidate '%s'; failing closed.", candidate_label,
            )
            return VerificationResult(
                accepted=False,
                candidate_label=candidate_label,
                similarity=0.0,
                threshold=self.threshold,
                reason="computation_error",
            )

        accepted = top_k_similarity >= self.threshold

        # Do not log the raw embeddings themselves, only the scalar result.
        log.info(
            "Verification: candidate=%s similarity=%.4f threshold=%.4f -> %s",
            candidate_label, top_k_similarity, self.threshold,
            "accepted" if accepted else "rejected",
        )

        return VerificationResult(
            accepted=accepted,
            candidate_label=candidate_label,
            similarity=top_k_similarity,
            threshold=self.threshold,
            reason="ok" if accepted else "below_threshold",
        )