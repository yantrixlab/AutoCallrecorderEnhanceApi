"""SQLite-backed job table. Single-instance scale (one VPS, one process), so a
plain file-based DB is enough - no need for Postgres/Redis for this."""

import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

DB_PATH = Path(__import__("os").environ.get("DB_PATH", "/data/jobs.db"))

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'queued',
                extension TEXT NOT NULL,
                error TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        # Additive migration for deployments created before original_filename
        # existed - jobs are short-lived (swept after 48h) so there's no real
        # data to migrate, just the column shape.
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN original_filename TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()


def create_job(extension: str, original_filename: str) -> str:
    job_id = uuid.uuid4().hex
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, extension, created_at, original_filename) VALUES (?, 'queued', ?, ?, ?)",
            (job_id, extension, time.time(), original_filename),
        )
        conn.commit()
    return job_id


def set_status(job_id: str, status: str, error: Optional[str] = None) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
            (status, error, job_id),
        )
        conn.commit()


def get_job(job_id: str) -> Optional[sqlite3.Row]:
    with _lock, _connect() as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        return cur.fetchone()


def get_jobs_older_than(seconds: float) -> list:
    cutoff = time.time() - seconds
    with _lock, _connect() as conn:
        cur = conn.execute("SELECT job_id FROM jobs WHERE created_at < ?", (cutoff,))
        return [row["job_id"] for row in cur.fetchall()]


def delete_job(job_id: str) -> None:
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        conn.commit()
