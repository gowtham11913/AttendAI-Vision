"""
utils/logger.py

Centralized, singleton logging module for the AI-Based Smart Attendance
System. Every other module in the project must obtain its logger through
get_logger() defined here. Do not instantiate logging.Logger directly
anywhere else in the project.

Design:
    - One physical log file per subsystem (registration, embedding,
      training, recognition, attendance, system), each with its own
      RotatingFileHandler (10 MB per file, 5 backups, UTF-8).
    - A single shared, colored console handler attached to the root
      "attendance_system" logger so console output is not duplicated
      per-module.
    - Console shows INFO and above by default; files capture DEBUG and
      above. This keeps the console clean while file logs stay detailed.
    - Thread-safe singleton setup guarded by a lock, safe to call
      get_logger() many times / from many modules.
"""

import logging
import logging.handlers
import os
import sys
import threading

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5
_ENCODING = "utf-8"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module_name)-18s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Maps a logical "module" name (as used by callers) to its dedicated log file.
_MODULE_LOG_FILES = {
    "register_student": "registration.log",
    "generate_embeddings": "embedding.log",
    "train_model": "training.log",
    "attendance_system": "recognition.log",
    "attendance": "attendance.log",
    "system": "system.log",
    # Fallback / shared utility modules default to system.log unless
    # explicitly mapped above.
}

_DEFAULT_LOG_FILE = "system.log"

_ROOT_LOGGER_NAME = "attendance_system"

_lock = threading.Lock()
_initialized = False
_loggers = {}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _ModuleNameFilter(logging.Filter):
    """Injects a stable 'module_name' field into every log record so the
    format string can print a clean subsystem name regardless of the
    actual Python logger name (e.g. 'attendance_system.recognition')."""

    def __init__(self, module_name):
        super().__init__()
        self.module_name = module_name

    def filter(self, record):
        record.module_name = self.module_name
        return True


class _ColorFormatter(logging.Formatter):
    """Adds ANSI colors to console output based on log level.
    Falls back to plain text automatically if the stream is not a TTY
    or the platform does not support ANSI (best-effort only)."""

    _COLORS = {
        logging.DEBUG: "\033[36m",     # cyan
        logging.INFO: "\033[32m",      # green
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[1;41m",  # bold white on red
    }
    _RESET = "\033[0m"

    def __init__(self, fmt, datefmt, use_color):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        message = super().format(record)
        if not self.use_color:
            return message
        color = self._COLORS.get(record.levelno, "")
        return f"{color}{message}{self._RESET}" if color else message


def _supports_color(stream):
    try:
        return hasattr(stream, "isatty") and stream.isatty() and os.name != "nt" or \
            (os.name == "nt" and os.environ.get("ANSICON") is not None)
    except Exception:
        return False


def _ensure_log_dir():
    os.makedirs(_LOG_DIR, exist_ok=True)


def _build_file_handler(filename):
    path = os.path.join(_LOG_DIR, filename)
    handler = logging.handlers.RotatingFileHandler(
        filename=path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding=_ENCODING,
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def _build_console_handler():
    stream = sys.stdout
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)  # console stays clean; details go to files
    handler.setFormatter(
        _ColorFormatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT, use_color=_supports_color(stream))
    )
    return handler


def _initialize():
    """One-time setup: creates the logs/ directory and the root logger's
    shared console handler. Idempotent."""
    global _initialized
    with _lock:
        if _initialized:
            return
        _ensure_log_dir()

        root = logging.getLogger(_ROOT_LOGGER_NAME)
        root.setLevel(logging.DEBUG)
        root.propagate = False

        # Avoid duplicate console handlers if _initialize() is somehow
        # triggered more than once in odd import scenarios.
        if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                   for h in root.handlers):
            root.addHandler(_build_console_handler())

        _initialized = True


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_logger(module_name):
    """
    Return a singleton logger configured for the given logical module name.

    Args:
        module_name: logical name of the calling module, e.g.
            "register_student", "generate_embeddings", "train_model",
            "attendance_system", "attendance", "camera", "embedding",
            "recognition", "antispoof", "system".

    Log routing:
        register_student      -> logs/registration.log
        generate_embeddings   -> logs/embedding.log
        train_model           -> logs/training.log
        attendance_system     -> logs/recognition.log
        attendance             -> logs/attendance.log
        anything else          -> logs/system.log

    All loggers additionally share ONE console handler (attached once to
    the root "attendance_system" logger), so console output is never
    duplicated no matter how many modules call get_logger().
    """
    _initialize()

    with _lock:
        if module_name in _loggers:
            return _loggers[module_name]

        logger = logging.getLogger(f"{_ROOT_LOGGER_NAME}.{module_name}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = True  # let console handler on root see it

        log_file = _MODULE_LOG_FILES.get(module_name, _DEFAULT_LOG_FILE)
        file_handler = _build_file_handler(log_file)
        file_handler.addFilter(_ModuleNameFilter(module_name))
        logger.addHandler(file_handler)

        # The root's console handler doesn't have the module_name attribute
        # filter, so attach the same filter logic at the logger level too
        # (filters on a logger apply to records before they propagate).
        logger.addFilter(_ModuleNameFilter(module_name))

        _loggers[module_name] = logger
        return logger


def shutdown_logging():
    """Flush and close all handlers cleanly. Call on application exit
    if you want to guarantee buffered log lines are flushed to disk."""
    logging.shutdown()
