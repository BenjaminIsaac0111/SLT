# -----------------------------
# export/writer.py
# -----------------------------
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

__all__ = ["BaseWriter", "JSONWriter"]

Grouped = Dict[str, List[dict]]


class BaseWriter(ABC):
    """Abstract interface for writer implementations."""

    @abstractmethod
    def write(self, path: Path, grouped: Grouped) -> None:  # pragma: no cover
        """Persist *grouped* annotations to *path*."""
        raise NotImplementedError


class JSONWriter(BaseWriter):
    """Write grouped annotations to a single JSON file."""

    def __init__(self, *, indent: int = 4) -> None:
        self._indent = indent

    def write(self, path: Path | str, grouped: Grouped) -> None:
        path = Path(path).expanduser().resolve()
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf‑8") as fh:
            json.dump(grouped, fh, indent=self._indent, ensure_ascii=False)
