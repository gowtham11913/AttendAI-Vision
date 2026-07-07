"""
migrate_csv_to_sqlite.py
------------------------
One-time migration of existing AttendAI Vision CSV data into SQLite.

Expected CSV files:
- database/students.csv
- database/attendance.csv

Output:
- database/attendance.db

Safety:
- validates CSV columns before writing
- preserves CSV files
- refuses duplicate migration unless --force is used
- detects incompatible pre-existing SQLite schemas before CREATE INDEX runs
- uses one explicit write transaction for imported rows
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"

STUDENTS_CSV = DATABASE_DIR / "students.csv"
ATTENDANCE_CSV = DATABASE_DIR / "attendance.csv"
DATABASE_FILE = DATABASE_DIR / "attendance.db"

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

EXPECTED_SQLITE_COLUMNS = {
    "students": {
        "id",
        "roll_no",
        "name",
        "department",
        "image_count",
        "registered_date",
        "is_active",
    },
    "attendance": {
        "id",
        "roll_no",
        "name",
        "date",
        "time",
        "status",
        "created_at",
    },
}


def parse_bool(value) -> int:
    return 0 if str(value).strip().lower() in {
        "false",
        "0",
        "no",
        "inactive",
        "",
    } else 1


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()
    }


def validate_existing_schema(
    connection: sqlite3.Connection,
) -> None:
    """
    Fail with a clear message before CREATE INDEX statements execute.

    This prevents confusing errors such as:
        sqlite3.OperationalError: no such column: roll_no
    when attendance.db already contains an older or unrelated schema.
    """
    problems: list[str] = []

    for table_name, required_columns in EXPECTED_SQLITE_COLUMNS.items():
        if not table_exists(connection, table_name):
            continue

        actual_columns = table_columns(connection, table_name)
        missing = sorted(required_columns - actual_columns)

        if missing:
            problems.append(
                f"table '{table_name}' is missing columns: "
                f"{', '.join(missing)}"
            )

    if problems:
        details = "; ".join(problems)
        raise RuntimeError(
            "Existing attendance.db uses an incompatible schema. "
            f"{details}. "
            "Migration stopped before modifying the database. "
            "Back up the existing database, then rename or remove "
            "database/attendance.db (and any attendance.db-wal / "
            "attendance.db-shm files) before running migration again."
        )


def create_schema(connection: sqlite3.Connection) -> None:
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


def validate_columns(
    dataframe: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: "
            f"{', '.join(missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing SQLite student and attendance rows.",
    )
    args = parser.parse_args()

    if not STUDENTS_CSV.is_file():
        raise FileNotFoundError(f"Missing file: {STUDENTS_CSV}")

    if not ATTENDANCE_CSV.is_file():
        raise FileNotFoundError(f"Missing file: {ATTENDANCE_CSV}")

    students_df = pd.read_csv(
        STUDENTS_CSV,
        dtype={"roll_no": str},
    )
    attendance_df = pd.read_csv(
        ATTENDANCE_CSV,
        dtype={"roll_no": str},
    )

    validate_columns(
        students_df,
        STUDENT_COLUMNS,
        "students.csv",
    )
    validate_columns(
        attendance_df,
        ATTENDANCE_COLUMNS,
        "attendance.csv",
    )

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = connect(DATABASE_FILE)

    try:
        validate_existing_schema(connection)
        create_schema(connection)

        existing_students = connection.execute(
            "SELECT COUNT(*) FROM students"
        ).fetchone()[0]

        existing_attendance = connection.execute(
            "SELECT COUNT(*) FROM attendance"
        ).fetchone()[0]

        if (
            existing_students or existing_attendance
        ) and not args.force:
            raise RuntimeError(
                "attendance.db already contains data. "
                "Migration stopped to prevent duplicates. "
                "Use --force only if you intentionally want to replace "
                "the SQLite student and attendance rows."
            )

        connection.execute("BEGIN IMMEDIATE")

        if args.force:
            connection.execute("DELETE FROM attendance")
            connection.execute("DELETE FROM students")

        for _, row in students_df.iterrows():
            roll_no = str(row["roll_no"]).strip().upper()

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
                (
                    roll_no,
                    str(row["name"]),
                    str(row["department"]),
                    int(row["image_count"]),
                    str(row["registered_date"]),
                    parse_bool(row["is_active"]),
                ),
            )

        skipped_attendance = []

        for index, row in attendance_df.iterrows():
            roll_no = str(row["roll_no"]).strip().upper()

            student_exists = connection.execute(
                """
                SELECT 1
                FROM students
                WHERE roll_no = ? COLLATE NOCASE
                LIMIT 1
                """,
                (roll_no,),
            ).fetchone()

            if student_exists is None:
                skipped_attendance.append(
                    {
                        "csv_row": int(index) + 2,
                        "roll_no": roll_no,
                    }
                )
                continue

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
                    roll_no,
                    str(row["name"]),
                    str(row["date"]),
                    str(row["time"]),
                    str(row["status"]),
                ),
            )

        connection.commit()

        student_count = connection.execute(
            "SELECT COUNT(*) FROM students"
        ).fetchone()[0]

        attendance_count = connection.execute(
            "SELECT COUNT(*) FROM attendance"
        ).fetchone()[0]

        print("Migration completed successfully.")
        print(f"SQLite file: {DATABASE_FILE}")
        print(f"Students migrated: {student_count}")
        print(f"Attendance rows migrated: {attendance_count}")

        if skipped_attendance:
            print(
                "Warning: some attendance rows were skipped because "
                "their roll numbers did not exist in students.csv:"
            )
            for item in skipped_attendance:
                print(
                    f"  CSV row {item['csv_row']}: "
                    f"{item['roll_no']}"
                )

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
