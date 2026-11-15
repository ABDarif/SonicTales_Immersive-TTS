"""SQLite persistence for generation jobs.

There are no accounts. Each browser gets a random private token in a cookie; only a hash of it is
stored here (`owner`), so a leaked database cannot be used to open anyone's audiobooks.
All SQL uses bound parameters.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,             -- sha256 of the visitor's cookie token
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
CREATE INDEX IF NOT EXISTS jobs_by_owner ON jobs(owner, created_at DESC);
"""

ACTIVE = ("queued", "running")


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            self._drop_legacy_account_tables(conn)
            conn.executescript(SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")
            # jobs that were running when the process died can never finish
            conn.execute(
                "UPDATE jobs SET status='failed', error='Interrupted by a server restart', finished_at=? "
                "WHERE status IN ('queued','running')",
                (time.time(),),
            )

    @staticmethod
    def _drop_legacy_account_tables(conn: sqlite3.Connection) -> None:
        """Earlier versions had users/sessions and per-user jobs. Those histories belonged to
        accounts that no longer exist and can't be opened by anyone, so start clean."""
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
        if "user_id" in cols:
            conn.execute("DROP TABLE jobs")
        conn.execute("DROP TABLE IF EXISTS sessions")
        conn.execute("DROP TABLE IF EXISTS users")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commit / rollback
                yield conn
        finally:
            conn.close()

    def create_job(self, job_id: str, owner: str, title: str, options_json: str, fmt: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, owner, title, status, options, fmt, created_at) VALUES (?,?,?,?,?,?,?)",
                (job_id, owner, title, "queued", options_json, fmt, time.time()),
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

    def get_job(self, job_id: str, owner: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM jobs WHERE id = ? AND owner = ?", (job_id, owner)).fetchone()

    def list_jobs(self, owner: str, limit: int = 100) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM jobs WHERE owner = ? ORDER BY created_at DESC LIMIT ?", (owner, limit)
            ).fetchall()

    def delete_job(self, job_id: str, owner: str) -> bool:
        with self.connect() as conn:
            return conn.execute("DELETE FROM jobs WHERE id = ? AND owner = ?", (job_id, owner)).rowcount > 0

    def count_active(self, owner: str | None = None) -> int:
        marks = ",".join("?" * len(ACTIVE))
        sql, args = f"SELECT COUNT(*) FROM jobs WHERE status IN ({marks})", list(ACTIVE)
        if owner is not None:
            sql, args = sql + " AND owner = ?", args + [owner]
        with self.connect() as conn:
            return conn.execute(sql, args).fetchone()[0]

    def all_job_ids(self) -> set[str]:
        with self.connect() as conn:
            return {r["id"] for r in conn.execute("SELECT id FROM jobs")}

    def delete_finished_before(self, cutoff: float) -> list[str]:
        """Remove finished jobs created before `cutoff`; returns their ids so files can be deleted too."""
        with self.connect() as conn:
            ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM jobs WHERE created_at < ? AND status NOT IN ('queued','running')", (cutoff,)
                )
            ]
            conn.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in ids])
        return ids
