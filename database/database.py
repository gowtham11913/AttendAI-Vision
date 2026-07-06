"""
attendance/database.py
──────────────────────
All database I/O via SQLite.  Replaces every pandas.to_csv() call.

Why SQLite over CSV
───────────────────
• ACID-safe: a crash mid-write never corrupts the file (WAL journal).
• Concurrent-safe: multiple processes can read simultaneously; writes are
  serialised at the file level automatically.
• Fast on read: indexed lookups beat scanning a CSV at 200+ rows.
• Rich queries: "students below 80% this month" is one SQL statement,
  not a pandas groupby chain loaded into RAM.

Schema
──────
  students    – roster (one row per student)
  attendance  – one row per student per day (UNIQUE enforced)
  sessions    – one row per camera run (audit trail)
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from project_1.attendance.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(BASE_DIR, "database", "attendance.db")
# ─── Schema ───────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   TEXT    UNIQUE NOT NULL,
    name         TEXT    NOT NULL,
    email        TEXT    DEFAULT '',
    class_name   TEXT    DEFAULT '',
    phone        TEXT    DEFAULT '',
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active    INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS attendance (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   TEXT    NOT NULL,
    date         DATE    NOT NULL,
    time         TIME    NOT NULL,
    timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status       TEXT    DEFAULT 'present',
    confidence   REAL    DEFAULT 1.0,
    session_id   INTEGER,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    UNIQUE (student_id, date)
);

CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at     TIMESTAMP,
    camera_index INTEGER DEFAULT 0,
    total_marks  INTEGER DEFAULT 0,
    notes        TEXT    DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_att_date    ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_att_student ON attendance(student_id);
"""


# ─── Connection factory ───────────────────────────────────────────────────────

@contextmanager
def get_db(path: str = DB_PATH):
    """
    Context manager returning a committed-or-rolled-back SQLite connection.

    Usage:
        with get_db() as conn:
            conn.execute("INSERT ...")
    """
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    # WAL mode: readers never block writers; writers never block readers.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: str = DB_PATH) -> None:
    """Create tables and indexes if they don't already exist."""
    with get_db(path) as conn:
        conn.executescript(_SCHEMA)
    logger.info(f"Database ready: {path}")


# ─── Student operations ───────────────────────────────────────────────────────

def add_student(student_id: str, name: str, email: str = "",
                class_name: str = "", phone: str = "") -> bool:
    """
    Insert a new student row.

    Returns:
        True on success, False if student_id already exists.
    """
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO students (student_id, name, email, class_name, phone) "
                "VALUES (?, ?, ?, ?, ?)",
                (student_id.strip(), name.strip(), email, class_name, phone),
            )
        logger.info(f"Student added: {student_id} ({name})")
        return True
    except sqlite3.IntegrityError:
        logger.warning(f"Duplicate student_id: {student_id}")
        return False
    except Exception as e:
        logger.error(f"add_student failed for {student_id}: {e}", exc_info=True)
        return False


def get_student(student_id: str) -> Optional[Dict]:
    """Return a student dict or None if not found / inactive."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE student_id = ? AND is_active = 1",
            (student_id,),
        ).fetchone()
        return dict(row) if row else None


def get_all_students(active_only: bool = True) -> List[Dict]:
    """Return all students ordered by name."""
    with get_db() as conn:
        q = "SELECT * FROM students"
        if active_only:
            q += " WHERE is_active = 1"
        q += " ORDER BY name"
        return [dict(r) for r in conn.execute(q).fetchall()]


def update_student(student_id: str, **fields) -> bool:
    """Update mutable student fields (name, email, class_name, phone)."""
    allowed = {"name", "email", "class_name", "phone"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [student_id]
    with get_db() as conn:
        conn.execute(
            f"UPDATE students SET {set_clause} WHERE student_id = ?", values
        )
    logger.info(f"Student updated: {student_id} {updates}")
    return True


def deactivate_student(student_id: str) -> bool:
    """Soft-delete (keeps attendance history intact)."""
    with get_db() as conn:
        conn.execute(
            "UPDATE students SET is_active = 0 WHERE student_id = ?", (student_id,)
        )
    logger.info(f"Student deactivated: {student_id}")
    return True


# ─── Attendance operations ────────────────────────────────────────────────────

def mark_attendance(
    student_id: str,
    confidence: float = 1.0,
    session_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Mark a student present for today.

    Rules:
    • One row per student per day (UNIQUE constraint).
    • If already marked, returns already_marked=True — not an error.

    Returns:
        {success, already_marked, message}
    """
    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M:%S")

    try:
        with get_db() as conn:
            existing = conn.execute(
                "SELECT time FROM attendance WHERE student_id = ? AND date = ?",
                (student_id, today),
            ).fetchone()

            if existing:
                logger.debug(f"Already marked: {student_id} at {existing['time']}")
                return {
                    "success": True,
                    "already_marked": True,
                    "message": f"Already marked at {existing['time']}",
                }

            conn.execute(
                "INSERT INTO attendance (student_id, date, time, confidence, session_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (student_id, today, now, confidence, session_id),
            )
            if session_id:
                conn.execute(
                    "UPDATE sessions SET total_marks = total_marks + 1 WHERE id = ?",
                    (session_id,),
                )

        logger.info(f"Attendance marked: {student_id} @ {now}  conf={confidence:.2f}")
        return {"success": True, "already_marked": False, "message": f"Marked present at {now}"}

    except Exception as e:
        logger.error(f"mark_attendance failed for {student_id}: {e}", exc_info=True)
        return {"success": False, "already_marked": False, "message": str(e)}


def get_attendance_today() -> List[Dict]:
    """Present students today, joined with their names."""
    today = date.today().isoformat()
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT a.student_id, s.name, s.class_name,
                   a.time, a.confidence, a.status
            FROM   attendance a
            JOIN   students   s ON a.student_id = s.student_id
            WHERE  a.date = ?
            ORDER  BY a.time
            """,
            (today,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_absent_today() -> List[Dict]:
    """Active students with no attendance row for today."""
    today = date.today().isoformat()
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT s.student_id, s.name, s.class_name
            FROM   students s
            WHERE  s.is_active = 1
              AND  s.student_id NOT IN (
                       SELECT student_id FROM attendance WHERE date = ?
                   )
            ORDER  BY s.name
            """,
            (today,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_attendance_by_date(target_date: str) -> List[Dict]:
    """Full attendance list for a given date (YYYY-MM-DD)."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT a.student_id, s.name, s.class_name,
                   a.time, a.confidence, a.status
            FROM   attendance a
            JOIN   students   s ON a.student_id = s.student_id
            WHERE  a.date = ?
            ORDER  BY s.name
            """,
            (target_date,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_attendance_summary(start_date: str, end_date: str) -> List[Dict]:
    """
    Per-student summary for a date range.

    Columns: student_id, name, class_name, days_present, total_days, percentage.
    """
    with get_db() as conn:
        total_days = conn.execute(
            "SELECT COUNT(DISTINCT date) FROM attendance WHERE date BETWEEN ? AND ?",
            (start_date, end_date),
        ).fetchone()[0] or 0

        rows = conn.execute(
            """
            SELECT s.student_id, s.name, s.class_name,
                   COUNT(a.id)                                    AS days_present,
                   ?                                              AS total_days,
                   ROUND(COUNT(a.id) * 100.0 / MAX(?, 1), 1)    AS percentage
            FROM   students   s
            LEFT JOIN attendance a
                   ON  s.student_id = a.student_id
                   AND a.date BETWEEN ? AND ?
            WHERE  s.is_active = 1
            GROUP  BY s.student_id
            ORDER  BY percentage DESC, s.name
            """,
            (total_days, total_days, start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]


def get_stats_today() -> Dict:
    """Quick dashboard stats."""
    today = date.today().isoformat()
    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM students WHERE is_active = 1"
        ).fetchone()[0]
        present = conn.execute(
            "SELECT COUNT(*) FROM attendance WHERE date = ?", (today,)
        ).fetchone()[0]
    absent = total - present
    rate = round(present * 100 / max(total, 1), 1)
    return {
        "total_students": total,
        "present_today": present,
        "absent_today": absent,
        "attendance_rate": rate,
        "date": today,
    }


def get_low_attendance(threshold: float = 80.0, days: int = 30) -> List[Dict]:
    """Students below *threshold* % attendance over the last *days* days."""
    end = date.today()
    start = end - timedelta(days=days)
    summary = get_attendance_summary(start.isoformat(), end.isoformat())
    return [s for s in summary if s["percentage"] < threshold]


def get_attendance_heatmap(student_id: str, days: int = 60) -> List[Dict]:
    """Per-day presence/absence for the last N days — used in the report chart."""
    end = date.today()
    start = end - timedelta(days=days)
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT date, status FROM attendance
            WHERE  student_id = ? AND date BETWEEN ? AND ?
            ORDER  BY date
            """,
            (student_id, start.isoformat(), end.isoformat()),
        ).fetchall()
        return [dict(r) for r in rows]


# ─── Session operations ───────────────────────────────────────────────────────

def start_session(camera_index: int = 0, notes: str = "") -> int:
    """Create a new session row and return its id."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO sessions (camera_index, notes) VALUES (?, ?)",
            (camera_index, notes),
        )
        sid = cursor.lastrowid
    logger.info(f"Session {sid} started (camera {camera_index})")
    return sid


def end_session(session_id: int) -> None:
    """Stamp ended_at on the session row."""
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )
    logger.info(f"Session {session_id} ended")