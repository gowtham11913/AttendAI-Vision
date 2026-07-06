"""
attendance.py
-------------
CSV utilities for:
- student lifecycle management (Active/Inactive)
- per-day attendance marking
"""

from __future__ import annotations
import os
import threading
from datetime import datetime
from typing import Optional
import pandas as pd

STUDENTS_CSV_COLUMNS = [
    "roll_no", "name", "department", "image_count",
    "registered_date", "is_active"
]
ATTENDANCE_CSV_COLUMNS = [
    "roll_no", "name", "date", "time", "status"
]

class StudentDatabase:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._ensure_file()
        self._migrate()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.isfile(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            pd.DataFrame(columns=STUDENTS_CSV_COLUMNS).to_csv(self.csv_path, index=False)

    def _migrate(self):
        df = pd.read_csv(self.csv_path, dtype={"roll_no": str})
        changed = False
        if "is_active" not in df.columns:
            df["is_active"] = True
            changed = True
        for col in STUDENTS_CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
                changed = True
        if changed:
            df[STUDENTS_CSV_COLUMNS].to_csv(self.csv_path, index=False)

    def load(self) -> pd.DataFrame:
        self._migrate()
        return pd.read_csv(self.csv_path, dtype={"roll_no": str})

    def exists(self, roll_no: str) -> bool:
        df = self.load()
        return not df.empty and str(roll_no) in df["roll_no"].astype(str).values

    def add_student(self, roll_no: str, name: str, department: str, image_count: int):
        if self.exists(roll_no):
            raise ValueError(f"Roll number '{roll_no}' is already registered.")
        record = {
            "roll_no": str(roll_no), "name": name, "department": department,
            "image_count": image_count,
            "registered_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_active": True,
        }
        df = pd.concat([self.load(), pd.DataFrame([record])], ignore_index=True)
        df[STUDENTS_CSV_COLUMNS].to_csv(self.csv_path, index=False)

    def get_student(self, roll_no: str) -> Optional[dict]:
        df = self.load()
        match = df[df["roll_no"].astype(str) == str(roll_no)]
        return None if match.empty else match.iloc[0].to_dict()

    def is_active(self, roll_no: str) -> bool:
        student = self.get_student(roll_no)
        if student is None:
            return False
        value = student.get("is_active", True)
        return str(value).strip().lower() not in {"false", "0", "no", "inactive"}

    def update_student(
        self,
        roll_no: str,
        name: str,
        department: str,
    ) -> dict:
        """
        Update student metadata in students.csv.

        Important:
        Filesystem artifact synchronization is handled by app.py.
        This method only updates the student database record.
        """
        normalized_roll_no = str(roll_no).strip().upper()
        normalized_name = " ".join(str(name).strip().split()).title()
        normalized_department = " ".join(
            str(department).strip().split()
        )

        df = self.load()

        mask = (
            df["roll_no"]
            .astype(str)
            .str.strip()
            .str.upper()
            == normalized_roll_no
        )

        if not mask.any():
            raise ValueError("Student not found.")

        df.loc[mask, "name"] = normalized_name
        df.loc[mask, "department"] = normalized_department

        df[STUDENTS_CSV_COLUMNS].to_csv(
            self.csv_path,
            index=False,
        )

        return self.get_student(normalized_roll_no)

    def set_active(self, roll_no: str, active: bool) -> dict:
        df = self.load()
        mask = df["roll_no"].astype(str) == str(roll_no)
        if not mask.any():
            raise ValueError("Student not found.")
        df.loc[mask, "is_active"] = bool(active)
        df[STUDENTS_CSV_COLUMNS].to_csv(self.csv_path, index=False)
        return self.get_student(roll_no)

    def delete_student(self, roll_no: str) -> bool:
        df = self.load()
        mask = df["roll_no"].astype(str) == str(roll_no)
        if not mask.any():
            return False
        df.loc[~mask, STUDENTS_CSV_COLUMNS].to_csv(self.csv_path, index=False)
        return True


class AttendanceLog:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        # Guards check-then-write so two concurrent requests for the same
        # student on the same day can't both pass the "not already marked"
        # check and both append a row.
        self._lock = threading.Lock()
        self._ensure_file()
        self._migrate()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.isfile(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            pd.DataFrame(columns=ATTENDANCE_CSV_COLUMNS).to_csv(self.csv_path, index=False)

    def _migrate(self):
        df = pd.read_csv(self.csv_path, dtype={"roll_no": str})
        changed = False
        for col in ATTENDANCE_CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
                changed = True
        # Drop any leftover columns from the removed session feature
        # (session_id, session_name) -- attendance is per-day again.
        extra_cols = [c for c in df.columns if c not in ATTENDANCE_CSV_COLUMNS]
        if extra_cols:
            changed = True
        if changed:
            df[ATTENDANCE_CSV_COLUMNS].to_csv(self.csv_path, index=False)

    def load(self) -> pd.DataFrame:
        self._migrate()
        return pd.read_csv(self.csv_path, dtype={"roll_no": str})

    def already_marked_today(self, roll_no: str) -> bool:
        df = self.load()
        if df.empty:
            return False
        today = datetime.now().strftime("%Y-%m-%d")
        match = df[(df["roll_no"].astype(str) == str(roll_no)) & (df["date"] == today)]
        return not match.empty

    def mark_if_not_marked_today(self, roll_no: str, name: str, status: str = "Present") -> bool:
        """
        Atomically check-and-mark within a single lock acquisition.
        Returns True if a new row was written, False if the student was
        already marked today (no row written).
        """
        with self._lock:
            df = self.load()
            today = datetime.now().strftime("%Y-%m-%d")
            already = not df.empty and not df[
                (df["roll_no"].astype(str) == str(roll_no)) & (df["date"] == today)
            ].empty
            if already:
                return False

            now = datetime.now()
            record = {
                "roll_no": str(roll_no), "name": name,
                "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"),
                "status": status,
            }
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            df[ATTENDANCE_CSV_COLUMNS].to_csv(self.csv_path, index=False)
            return True

    def mark_attendance(self, roll_no: str, name: str, status: str = "Present") -> bool:
        """Kept for backward compatibility. Prefer mark_if_not_marked_today()."""
        return self.mark_if_not_marked_today(roll_no, name, status)

    def append_record(self, roll_no: str, name: str, status: str = "Present") -> None:
        """
        Always write a new attendance row, regardless of existing entries
        today. Used when the "attendance once per day" setting is off
        (e.g. separate entry/exit style logging).
        """
        with self._lock:
            now = datetime.now()
            record = {
                "roll_no": str(roll_no), "name": name,
                "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"),
                "status": status,
            }
            df = pd.concat([self.load(), pd.DataFrame([record])], ignore_index=True)
            df[ATTENDANCE_CSV_COLUMNS].to_csv(self.csv_path, index=False)