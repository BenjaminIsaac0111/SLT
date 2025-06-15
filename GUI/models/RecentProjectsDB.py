from __future__ import annotations

"""Helpers for persisting recently used project paths."""

import sqlite3
from pathlib import Path
from typing import List

_DB_PATH = Path.home() / ".attentionunet" / "recent_projects.db"


def _ensure_db() -> None:
    """Create the database schema if needed."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_path(path: str) -> None:
    """Record *path* in the recent projects database."""
    _ensure_db()
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO projects (path) VALUES (?)",
            (str(Path(path).expanduser()),),
        )
        conn.commit()


def get_recent_paths(limit: int = 5) -> List[str]:
    """Return up to *limit* most recently saved project paths."""
    _ensure_db()
    with sqlite3.connect(_DB_PATH) as conn:
        cur = conn.execute(
            "SELECT DISTINCT path FROM projects ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [row[0] for row in cur.fetchall()]
