"""
run_pipeline.py

Automatic end-to-end pipeline for the AI-Based Smart Attendance System.

Chains the existing, unmodified project scripts in sequence:

    register_student.py  -->  generate_embeddings.py  -->  train_model.py
                                                              |
                                                              v
                                                   attendance_system.py (optional)

Design notes
------------
- This file does NOT reimplement any business logic. Each stage is
  invoked as a subprocess running the existing script exactly as-is,
  so registration, embedding, training, and attendance logic remain
  untouched.
- generate_embeddings.py already skips students whose .npy embedding
  exists, so re-running it after every registration is cheap and safe
  (only the new student gets embedded).
- train_model.py is retrained on the full embedding set each time a
  new student is added, since an SVM needs to be refit whenever the
  label set changes (a single new class can't be added incrementally
  to a linear SVM without retraining).
- If any stage fails, the pipeline stops immediately and logs the
  failure — it will NOT silently continue to training/attendance with
  incomplete data.

Usage
-----
    python run_pipeline.py                      # register -> embed -> train
    python run_pipeline.py --skip-register       # embed -> train (e.g. bulk re-train)
    python run_pipeline.py --start-attendance    # also launch attendance_system.py after training
    python run_pipeline.py --python /path/to/venv/python   # use a specific interpreter
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from utils.logger import get_logger

log = get_logger("system")

PROJECT_ROOT = Path(__file__).resolve().parent

STAGES = {
    "register": PROJECT_ROOT / "register_student.py",
    "embed": PROJECT_ROOT / "generate_embeddings.py",
    "train": PROJECT_ROOT / "train_model.py",
    "attendance": PROJECT_ROOT / "attendance_system.py",
}


class PipelineError(Exception):
    """Raised when a pipeline stage fails, to stop the chain immediately."""


def _run_stage(stage_name, script_path, python_exe):
    """Run one stage as a subprocess of the existing script, streaming its
    output live and logging start/success/failure centrally."""
    if not script_path.exists():
        raise PipelineError(f"{stage_name}: expected script not found at {script_path}")

    log.info(f"Pipeline stage started: {stage_name} ({script_path.name})")
    start = time.time()

    try:
        result = subprocess.run(
            [python_exe, str(script_path)],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
    except Exception as e:
        log.critical(f"Pipeline stage '{stage_name}' crashed launching subprocess: {e}", exc_info=True)
        raise PipelineError(f"{stage_name} failed to launch") from e

    elapsed = time.time() - start

    if result.returncode != 0:
        log.error(f"Pipeline stage failed: {stage_name} (exit code {result.returncode}, {elapsed:.1f}s)")
        raise PipelineError(f"{stage_name} exited with code {result.returncode}")

    log.info(f"Pipeline stage completed: {stage_name} ({elapsed:.1f}s)")


def run_pipeline(skip_register=False, start_attendance=False, python_exe=None):
    """
    Runs the full registration -> embedding -> training pipeline, and
    optionally launches attendance afterwards.

    Returns True if the pipeline completed successfully, False otherwise.
    """
    python_exe = python_exe or sys.executable

    log.info("Automatic pipeline started")
    pipeline_start = time.time()

    try:
        if not skip_register:
            _run_stage("register", STAGES["register"], python_exe)
        else:
            log.info("Pipeline stage skipped: register (--skip-register)")

        _run_stage("embed", STAGES["embed"], python_exe)
        _run_stage("train", STAGES["train"], python_exe)

        total_elapsed = time.time() - pipeline_start
        log.info(f"Automatic pipeline completed successfully ({total_elapsed:.1f}s total)")

        if start_attendance:
            log.info("Launching attendance session after successful pipeline run")
            _run_stage("attendance", STAGES["attendance"], python_exe)

        return True

    except PipelineError as e:
        log.error(f"Automatic pipeline aborted: {e}")
        return False
    except KeyboardInterrupt:
        log.warning("Automatic pipeline cancelled by user (KeyboardInterrupt)")
        return False


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the attendance system pipeline: registration -> embeddings -> training -> (optional) attendance."
    )
    parser.add_argument(
        "--skip-register",
        action="store_true",
        help="Skip the registration step (e.g. when just retraining on existing data).",
    )
    parser.add_argument(
        "--start-attendance",
        action="store_true",
        help="Automatically launch attendance_system.py after training completes.",
    )
    parser.add_argument(
        "--python",
        dest="python_exe",
        default=None,
        help="Path to the Python interpreter to use for each stage (defaults to current interpreter).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    success = run_pipeline(
        skip_register=args.skip_register,
        start_attendance=args.start_attendance,
        python_exe=args.python_exe,
    )
    sys.exit(0 if success else 1)
