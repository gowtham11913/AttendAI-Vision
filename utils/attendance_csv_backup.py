"""
attendance.py
-------------
SQLite-backed utilities for:
- student lifecycle management (Active/Inactive)
- per-day attendance marking
- optional repeated same-day attendance records

Public class names and method signatures are preserved so the existing
AttendAI Vision app can migrate with minimal changes.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import pandas as pd


STUDENT_COLUMNS = [
    "roll_no",
    "name",
    "department",
    "image_count",
    "registered_date",
    "is_active",
]

ATTENDANCE_COLUMNS = [
    "roll_no",
    "name",
    "date",
    "time",
    "status",
]


class _SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    roll_no TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    image_count INTEGER NOT NULL DEFAULT 0,
                    registered_date TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                        CHECK (is_active IN (0, 1))
                );

                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    roll_no TEXT NOT NULL COLLATE NOCASE,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'Present',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (roll_no)
                        REFERENCES students(roll_no)
                        ON UPDATE CASCADE
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_attendance_roll_no
                    ON attendance(roll_no);

                CREATE INDEX IF NOT EXISTS idx_attendance_date
                    ON attendance(date);

                CREATE INDEX IF NOT EXISTS idx_attendance_roll_date
                    ON attendance(roll_no, date);
                """
            )


class StudentDatabase(_SQLiteStore):
    def __init__(self, db_path: str):
        super().__init__(db_path)

    def load(self) -> pd.DataFrame:
        with self._connection() as connection:
            df = pd.read_sql_query(
                """
                SELECT
                    roll_no,
                    name,
                    department,
                    image_count,
                    registered_date,
                    is_active
                FROM students
                ORDER BY id ASC
                """,
                connection,
            )

        if df.empty:
            return pd.DataFrame(columns=STUDENT_COLUMNS)

        df["roll_no"] = df["roll_no"].astype(str)
        df["is_active"] = df["is_active"].astype(bool)
        return df[STUDENT_COLUMNS]

    def exists(self, roll_no: str) -> bool:
        normalized_roll_no = str(roll_no).strip().upper()

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM students
                WHERE roll_no = ? COLLATE NOCASE
                LIMIT 1
                """,
                (normalized_roll_no,),
            ).fetchone()

        return row is not None

    def add_student(
        self,
        roll_no: str,
        name: str,
        department: str,
        image_count: int,
    ):
        normalized_roll_no = str(roll_no).strip().upper()

        if self.exists(normalized_roll_no):
            raise ValueError(
                f"Roll number '{normalized_roll_no}' is already registered."
            )

        record = (
            normalized_roll_no,
            str(name),
            str(department),
            int(image_count),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            1,
        )

        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO students (
                        roll_no,
                        name,
                        department,
                        image_count,
                        registered_date,
                        is_active
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    record,
                )
                connection.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"Roll number '{normalized_roll_no}' is already registered."
            ) from exc

    def get_student(self, roll_no: str) -> Optional[dict]:
        normalized_roll_no = str(roll_no).strip().upper()

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT
                    roll_no,
                    name,
                    department,
                    image_count,
                    registered_date,
                    is_active
                FROM students
                WHERE roll_no = ? COLLATE NOCASE
                LIMIT 1
                """,
                (normalized_roll_no,),
            ).fetchone()

        if row is None:
            return None

        student = dict(row)
        student["is_active"] = bool(student["is_active"])
        return student

    def is_active(self, roll_no: str) -> bool:
        student = self.get_student(roll_no)
        if student is None:
            return False
        return bool(student.get("is_active", True))

    def update_student(
        self,
        roll_no: str,
        name: str,
        department: str,
    ) -> dict:
        normalized_roll_no = str(roll_no).strip().upper()
        normalized_name = " ".join(str(name).strip().split()).title()
        normalized_department = " ".join(
            str(department).strip().split()
        )

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE students
                SET name = ?, department = ?
                WHERE roll_no = ? COLLATE NOCASE
                """,
                (
                    normalized_name,
                    normalized_department,
                    normalized_roll_no,
                ),
            )

            if cursor.rowcount == 0:
                connection.rollback()
                raise ValueError("Student not found.")

            connection.commit()

        return self.get_student(normalized_roll_no)

    def set_active(self, roll_no: str, active: bool) -> dict:
        normalized_roll_no = str(roll_no).strip().upper()

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE students
                SET is_active = ?
                WHERE roll_no = ? COLLATE NOCASE
                """,
                (1 if active else 0, normalized_roll_no),
            )

            if cursor.rowcount == 0:
                connection.rollback()
                raise ValueError("Student not found.")

            connection.commit()

        return self.get_student(normalized_roll_no)

    def delete_student(self, roll_no: str) -> bool:
        normalized_roll_no = str(roll_no).strip().upper()

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                DELETE FROM students
                WHERE roll_no = ? COLLATE NOCASE
                """,
                (normalized_roll_no,),
            )
            deleted = cursor.rowcount > 0
            connection.commit()

        return deleted


class AttendanceLog(_SQLiteStore):
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        super().__init__(db_path)

    def load(self) -> pd.DataFrame:
        with self._connection() as connection:
            df = pd.read_sql_query(
                """
                SELECT
                    roll_no,
                    name,
                    date,
                    time,
                    status
                FROM attendance
                ORDER BY id ASC
                """,
                connection,
            )

        if df.empty:
            return pd.DataFrame(columns=ATTENDANCE_COLUMNS)

        df["roll_no"] = df["roll_no"].astype(str)
        return df[ATTENDANCE_COLUMNS]

    def already_marked_today(self, roll_no: str) -> bool:
        normalized_roll_no = str(roll_no).strip().upper()
        today = datetime.now().strftime("%Y-%m-%d")

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM attendance
                WHERE roll_no = ? COLLATE NOCASE
                  AND date = ?
                LIMIT 1
                """,
                (normalized_roll_no, today),
            ).fetchone()

        return row is not None

    def mark_if_not_marked_today(
        self,
        roll_no: str,
        name: str,
        status: str = "Present",
    ) -> bool:
        """
        Atomically check and insert inside one SQLite write transaction.

        Returns:
            True  -> a new row was inserted
            False -> the student was already marked today
        """
        normalized_roll_no = str(roll_no).strip().upper()

        with self._lock:
            with self._connection() as connection:
                try:
                    connection.execute("BEGIN IMMEDIATE")

                    now = datetime.now()
                    today = now.strftime("%Y-%m-%d")

                    existing = connection.execute(
                        """
                        SELECT 1
                        FROM attendance
                        WHERE roll_no = ? COLLATE NOCASE
                          AND date = ?
                        LIMIT 1
                        """,
                        (normalized_roll_no, today),
                    ).fetchone()

                    if existing is not None:
                        connection.rollback()
                        return False

                    connection.execute(
                        """
                        INSERT INTO attendance (
                            roll_no,
                            name,
                            date,
                            time,
                            status
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            normalized_roll_no,
                            str(name),
                            today,
                            now.strftime("%H:%M:%S"),
                            str(status),
                        ),
                    )
                    connection.commit()
                    return True

                except Exception:
                    connection.rollback()
                    raise

    def mark_attendance(
        self,
        roll_no: str,
        name: str,
        status: str = "Present",
    ) -> bool:
        """Backward-compatible wrapper."""
        return self.mark_if_not_marked_today(
            roll_no,
            name,
            status,
        )

    def append_record(
        self,
        roll_no: str,
        name: str,
        status: str = "Present",
    ) -> None:
        """
        Always insert a new attendance row.

        This intentionally permits multiple same-day records when the
        attendance_once_per_day setting is disabled.
        """
        normalized_roll_no = str(roll_no).strip().upper()

        with self._lock:
            with self._connection() as connection:
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    now = datetime.now()

                    connection.execute(
                        """
                        INSERT INTO attendance (
                            roll_no,
                            name,
                            date,
                            time,
                            status
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            normalized_roll_no,
                            str(name),
                            now.strftime("%Y-%m-%d"),
                            now.strftime("%H:%M:%S"),
                            str(status),
                        ),
                    )
                    connection.commit()

                except Exception:
                    connection.rollback()
                    raise
