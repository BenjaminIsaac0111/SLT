# -----------------------------
# export/options.py
# -----------------------------
"""Options objects shared across the *export* package."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExportOptions:
    """Configuration flags that influence the export behaviour.

    Attributes
    ----------
    include_artifacts
        If *True*, annotations whose ``class_id`` equals ``-3`` (artifacts) are
        included in the export.  Otherwise, they are filtered‑out.
    """

    include_artifacts: bool = False


# Re‑export to allow ``from export import ExportOptions``
__all__ = ["ExportOptions"]
