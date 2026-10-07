"""SQLite persistence: users, login sessions and generation jobs.

All SQL uses bound parameters (the old PHP version interpolated user input into queries).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    email TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    status TEXT NOT NULL,            -- queued | running | done | failed
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error TEXT,
    options TEXT NOT NULL DEFAULT '{}',
    fmt TEXT NOT NULL DEFAULT 'mp3',
    engine TEXT,
    duration_s REAL,
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_by_user ON jobs(user_id, created_at DESC);
"""

SESSION_TTL_S = 14 * 24 * 3600


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        candidate = hash_password(password, bytes.fromhex(salt_hex)).split("$")[2]
    except ValueError:
        return False
    return hmac.compare_digest(candidate, digest_hex)


# Used to burn the same CPU time when a username doesn't exist (no user enumeration by timing).
_DUMMY_HASH = hash_password("dummy-password-for-timing")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")
            # jobs that were running when the process died can never finish
            conn.execute(
                "UPDATE jobs SET status='failed', error='Interrupted by a server restart', finished_at=? "
                "WHERE status IN ('queued','running')",
                (time.time(),),
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:  # commit / rollback
                yield conn
        finally:
            conn.close()

    # ---- users

    def create_user(self, username: str, email: str, password: str) -> int | None:
        """Returns the new user id, or None if the username is taken."""
        try:
            with self.connect() as conn:
                cur = conn.execute(
                    "INSERT INTO users (username, email, password_hash, created_at) VALUES (?,?,?,?)",
                    (username, email, hash_password(password), time.time()),
                )
                return cur.lastrowid
        except sqlite3.IntegrityError:
            return None

    def authenticate(self, username: str, password: str) -> int | None:
        with self.connect() as conn:
            row = conn.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            verify_password(password, _DUMMY_HASH)
            return None
        return row["id"] if verify_password(password, row["password_hash"]) else None

    def get_user(self, user_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,)).fetchone()

    # ---- sessions

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        with self.connect() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?,?,?)",
                (_token_hash(token), user_id, time.time() + SESSION_TTL_S),
            )
        return token

    def user_for_session(self, token: str | None) -> sqlite3.Row | None:
        if not token:
            return None
        with self.connect() as conn:
            return conn.execute(
                "SELECT u.id, u.username, u.email FROM sessions s JOIN users u ON u.id = s.user_id "
                "WHERE s.token_hash = ? AND s.expires_at > ?",
                (_token_hash(token), time.time()),
            ).fetchone()

    def delete_session(self, token: str | None) -> None:
        if token:
            with self.connect() as conn:
                conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))

    # ---- jobs

    def create_job(self, job_id: str, user_id: int, title: str, options_json: str, fmt: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, user_id, title, status, options, fmt, created_at) VALUES (?,?,?,?,?,?,?)",
                (job_id, user_id, title, "queued", options_json, fmt, time.time()),
            )

    def update_job(self, job_id: str, **fields: object) -> None:
        allowed = {"status", "progress", "message", "error", "engine", "duration_s", "finished_at"}
        cols = [k for k in fields if k in allowed]
        if not cols:
            return
        with self.connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?",  # column names are whitelisted
                [fields[c] for c in cols] + [job_id],
            )

    def get_job(self, job_id: str, user_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id)).fetchone()

    def list_jobs(self, user_id: int, limit: int = 100) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)
            ).fetchall()

    def delete_job(self, job_id: str, user_id: int) -> bool:
        with self.connect() as conn:
            return conn.execute("DELETE FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id)).rowcount > 0
