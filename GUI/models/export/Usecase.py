# -----------------------------
# export/usecase.py
# -----------------------------
"""Application‑level façade that orchestrates the export workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from GUI.models.annotations import AnnotationBase
from .ExportService import build_grouped_annotations
from .Options import ExportOptions
from .Writer import BaseWriter, JSONWriter

__all__ = ["ExportAnnotationsUseCase"]


class ExportAnnotationsUseCase:
    """High‑level use‑case class invoked by the GUI layer."""

    def __init__(self, *, writer: BaseWriter | None = None) -> None:
        self._writer: BaseWriter = writer or JSONWriter()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def __call__(
            self,
            cluster_annos: Iterable[Tuple[int, AnnotationBase]],
            opts: ExportOptions,
            out_path: Path | str,
    ) -> int:
        """Filter, group, and write *cluster_annos* to *out_path*.

        Returns the total number of exported annotations.
        """
        grouped = build_grouped_annotations(cluster_annos, opts)
        if not grouped:
            raise ValueError("Nothing to export with the given options.")

        self._writer.write(Path(out_path), grouped)
        return sum(len(v) for v in grouped.values())
