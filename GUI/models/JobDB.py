from __future__ import annotations

"""Simple SQLite-backed job tracking for the GUI scheduler."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH = Path.home() / ".attentionunet" / "jobs.db"


def _ensure_db() -> None:
    """Create the jobs table if needed."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                config TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def add_job(name: str, config: Optional[Dict[str, Any]] = None) -> int:
    """Insert a new job and return its ID."""
    _ensure_db()
    cfg = json.dumps(config or {})
    with sqlite3.connect(_DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO jobs (name, config, status) VALUES (?, ?, ?)",
            (name, cfg, "queued"),
        )
        conn.commit()
        return cur.lastrowid


def update_status(job_id: int, status: str) -> None:
    """Update the status for *job_id*."""
    _ensure_db()
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, job_id),
        )
        conn.commit()


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recently recorded jobs."""
    _ensure_db()
    with sqlite3.connect(_DB_PATH) as conn:
        cur = conn.execute(
            "SELECT id, name, config, status FROM jobs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "id": row[0],
                "name": row[1],
                "config": json.loads(row[2]) if row[2] else {},
                "status": row[3],
            }
            for row in cur.fetchall()
        ]


