from __future__ import annotations

"""Utilities for persisting MC banker configurations."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

_DB_PATH = Path.home() / ".attentionunet" / "mc_banker_configs.db"


def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_path TEXT,
                data_dir TEXT,
                file_list TEXT,
                mc_iter INTEGER,
                temperature REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur = conn.execute("PRAGMA table_info(configs)")
        cols = [row[1] for row in cur.fetchall()]
        if "temperature" not in cols:
            conn.execute("ALTER TABLE configs ADD COLUMN temperature REAL")


def save_config(cfg: Dict[str, Any]) -> None:
    """Persist configuration details for later reuse."""
    _ensure_db()
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO configs (model_path, data_dir, file_list, mc_iter, temperature) VALUES (?, ?, ?, ?, ?)",
            (
                str(Path(cfg["MODEL_DIR"]) / cfg["MODEL_NAME"]),
                cfg["DATA_DIR"],
                cfg["FILE_LIST"],
                int(cfg["MC_N_ITER"]),
                float(cfg.get("TEMPERATURE", 1.0)),
            ),
        )
        conn.commit()


def get_recent_model_paths(limit: int = 5) -> List[str]:
    """Return up to *limit* previously used model paths."""
    _ensure_db()
    with sqlite3.connect(_DB_PATH) as conn:
        cur = conn.execute(
            "SELECT DISTINCT model_path FROM configs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [row[0] for row in cur.fetchall()]
