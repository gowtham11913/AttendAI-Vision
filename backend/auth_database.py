from __future__ import annotations
import hashlib, hmac, os, sqlite3
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATABASE_FILE=PROJECT_ROOT/"database"/"attendance.db"
PBKDF2_ITERATIONS=600_000

def _connect():
    c=sqlite3.connect(str(DATABASE_FILE),timeout=30.0)
    c.row_factory=sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    c.execute("PRAGMA busy_timeout = 30000")
    return c

def hash_password(password:str)->str:
    if not password or len(password)<8:
        raise ValueError("Password must contain at least 8 characters.")
    salt=os.urandom(16)
    digest=hashlib.pbkdf2_hmac("sha256",password.encode(),salt,PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"

def verify_password(password:str,stored_hash:str)->bool:
    try:
        alg,it,salt,h=stored_hash.split("$",3)
        if alg!="pbkdf2_sha256": return False
        calc=hashlib.pbkdf2_hmac("sha256",password.encode(),bytes.fromhex(salt),int(it))
        return hmac.compare_digest(calc.hex(),h)
    except (ValueError,TypeError,AttributeError):
        return False

def initialize_auth_database()->None:
    with _connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS auth_accounts(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT NOT NULL UNIQUE COLLATE NOCASE,
          password_hash TEXT NOT NULL,
          role TEXT NOT NULL CHECK(role IN ('admin','student')),
          roll_no TEXT COLLATE NOCASE,
          is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0,1)),
          must_change_password INTEGER NOT NULL DEFAULT 0 CHECK(must_change_password IN (0,1)),
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          last_login TEXT,
          FOREIGN KEY(roll_no) REFERENCES students(roll_no) ON UPDATE CASCADE ON DELETE CASCADE,
          CHECK((role='admin' AND roll_no IS NULL) OR (role='student' AND roll_no IS NOT NULL))
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_auth_student_roll_no
          ON auth_accounts(roll_no) WHERE roll_no IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_auth_role ON auth_accounts(role);
        """)
        cols={r["name"] for r in c.execute("PRAGMA table_info(auth_accounts)").fetchall()}
        if "must_change_password" not in cols:
            c.execute("""ALTER TABLE auth_accounts ADD COLUMN must_change_password
                         INTEGER NOT NULL DEFAULT 0 CHECK(must_change_password IN (0,1))""")

def create_admin_account(username:str,password:str)->None:
    u=username.strip()
    if not u: raise ValueError("Administrator username is required.")
    ph=hash_password(password)
    with _connect() as c:
        c.execute("""INSERT INTO auth_accounts(username,password_hash,role,roll_no,is_active,must_change_password)
                     VALUES(?,?,'admin',NULL,1,0)
                     ON CONFLICT(username) DO UPDATE SET password_hash=excluded.password_hash,
                     role='admin',roll_no=NULL,is_active=1,must_change_password=0""",(u,ph))

def student_account_exists(roll_no:str)->bool:
    r=roll_no.strip().upper()
    if not r:return False
    with _connect() as c:
        return c.execute("""SELECT 1 FROM auth_accounts WHERE role='student'
                            AND roll_no=? COLLATE NOCASE LIMIT 1""",(r,)).fetchone() is not None

def create_student_account(roll_no:str,password:str)->None:
    r=roll_no.strip().upper()
    if not r: raise ValueError("Roll number is required.")
    ph=hash_password(password)
    with _connect() as c:
        s=c.execute("""SELECT roll_no,name,is_active FROM students
                       WHERE roll_no=? COLLATE NOCASE""",(r,)).fetchone()
        if s is None: raise ValueError("Student is not registered in AttendAI Vision.")
        if int(s["is_active"])!=1: raise ValueError("Student account cannot be created for an inactive student.")
        old=c.execute("""SELECT id FROM auth_accounts WHERE username=? COLLATE NOCASE
                         OR roll_no=? COLLATE NOCASE LIMIT 1""",(r,r)).fetchone()
        if old is not None:return
        c.execute("""INSERT INTO auth_accounts(username,password_hash,role,roll_no,is_active,must_change_password)
                     VALUES(?,?,'student',?,1,1)""",(r,ph,r))

def authenticate_admin(username:str,password:str)->dict|None:
    with _connect() as c:
        a=c.execute("""SELECT id,username,password_hash,role FROM auth_accounts
                       WHERE username=? AND role='admin' AND is_active=1 COLLATE NOCASE""",
                    (username.strip(),)).fetchone()
        if a is None or not verify_password(password,a["password_hash"]):return None
        c.execute("UPDATE auth_accounts SET last_login=CURRENT_TIMESTAMP WHERE id=?",(a["id"],))
        return {"username":a["username"],"role":a["role"]}

def authenticate_student(roll_no:str,password:str)->dict|None:
    r=roll_no.strip().upper()
    with _connect() as c:
        a=c.execute("""SELECT a.id,a.username,a.password_hash,a.role,a.roll_no,
                       a.must_change_password,s.name,s.department
                       FROM auth_accounts a INNER JOIN students s
                       ON s.roll_no=a.roll_no COLLATE NOCASE
                       WHERE a.username=? AND a.role='student' AND a.is_active=1 AND s.is_active=1""",(r,)).fetchone()
        if a is None or not verify_password(password,a["password_hash"]):return None
        c.execute("UPDATE auth_accounts SET last_login=CURRENT_TIMESTAMP WHERE id=?",(a["id"],))
        return {"username":a["username"],"role":a["role"],"roll_no":a["roll_no"],
                "name":a["name"],"department":a["department"],
                "must_change_password":bool(a["must_change_password"])}

def change_student_password(roll_no:str,current_password:str,new_password:str)->bool:
    r=roll_no.strip().upper()
    with _connect() as c:
        a=c.execute("""SELECT id,password_hash FROM auth_accounts WHERE roll_no=?
                       AND role='student' AND is_active=1 COLLATE NOCASE""",(r,)).fetchone()
        if a is None or not verify_password(current_password,a["password_hash"]):return False
        if verify_password(new_password,a["password_hash"]):
            raise ValueError("New password must be different from the current password.")
        ph=hash_password(new_password)
        c.execute("""UPDATE auth_accounts SET password_hash=?,must_change_password=0 WHERE id=?""",(ph,a["id"]))
        return True

def get_student_attendance(roll_no:str)->list[dict]:
    r=roll_no.strip().upper()
    with _connect() as c:
        rows=c.execute("""SELECT date,time,status FROM attendance WHERE roll_no=?
                          COLLATE NOCASE ORDER BY date DESC,time DESC""",(r,)).fetchall()
        return [dict(x) for x in rows]


def change_admin_password(
    username: str,
    current_password: str,
    new_password: str,
) -> bool:
    clean_username = username.strip()

    with _connect() as connection:
        account = connection.execute(
            """
            SELECT id, password_hash
            FROM auth_accounts
            WHERE username = ?
              AND role = 'admin'
              AND is_active = 1
            COLLATE NOCASE
            """,
            (clean_username,),
        ).fetchone()

        if account is None:
            return False

        if not verify_password(
            current_password,
            account["password_hash"],
        ):
            return False

        if verify_password(
            new_password,
            account["password_hash"],
        ):
            raise ValueError(
                "New password must be different from the current password."
            )

        new_hash = hash_password(new_password)

        connection.execute(
            """
            UPDATE auth_accounts
            SET password_hash = ?
            WHERE id = ?
            """,
            (new_hash, account["id"]),
        )

        return True


def reset_student_password(
    roll_no: str,
    temporary_password: str,
) -> dict:
    normalized_roll_no = roll_no.strip().upper()

    if not normalized_roll_no:
        raise ValueError("Roll number is required.")

    password_hash = hash_password(temporary_password)

    with _connect() as connection:
        student = connection.execute(
            """
            SELECT roll_no, name, is_active
            FROM students
            WHERE roll_no = ?
            COLLATE NOCASE
            """,
            (normalized_roll_no,),
        ).fetchone()

        if student is None:
            raise ValueError("Student was not found.")

        if int(student["is_active"]) != 1:
            raise ValueError(
                "Password cannot be reset for an inactive student."
            )

        account = connection.execute(
            """
            SELECT id
            FROM auth_accounts
            WHERE role = 'student'
              AND roll_no = ?
            COLLATE NOCASE
            """,
            (normalized_roll_no,),
        ).fetchone()

        if account is None:
            raise ValueError(
                "This student does not have a portal account."
            )

        connection.execute(
            """
            UPDATE auth_accounts
            SET
                password_hash = ?,
                must_change_password = 1,
                is_active = 1
            WHERE id = ?
            """,
            (password_hash, account["id"]),
        )

        return {
            "roll_no": normalized_roll_no,
            "name": student["name"],
        }
