"""
Pure‑Python helpers for serialising / de‑serialising project files.
Features
--------
* Explicit schema validation with Pydantic
* Atomic writes via tempfile + os.replace
* Zstandard compression (falls back to gzip if package missing)
* **Version‑aware migrations** handled by a registry of step functions.

Public API
~~~~~~~~~~
    save_state(state: ProjectState, path: Path, *, level: int = 3) -> None
    load_state(path: Path) -> ProjectState

When the on‑disk `schema_version` is older than the latest model version
this module upgrades the raw dict *incrementally* (v2→v3→v4 …) before
validation so controllers never have to worry about legacy files.
"""

import gzip as _gzip_fallback
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Callable, Union

from GUI.configuration.configuration import LATEST_SCHEMA_VERSION

# ---------------------------------------------------------------------------
#  optional fast codec -------------------------------------------------------
# ---------------------------------------------------------------------------
try:
    import zstandard as zstd

    _ZSTD_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _ZSTD_AVAILABLE = False

# ---------------------------------------------------------------------------
#  pydantic schema -----------------------------------------------------------
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field, validator


class AnnotationJSON(BaseModel):
    image_index: int
    filename: str
    coord: List[int]
    logit_features: List[float]
    uncertainty: Union[float, List[float]]
    adjusted_uncertainty: Union[float, List[float], None] = None
    class_id: int
    cluster_id: Optional[int] = None
    model_prediction: Optional[str] = None
    mask_rle: Optional[List[int]] = None
    mask_shape: Optional[List[int]] = None


class ProjectState(BaseModel):
    schema_version: int = Field(ge=2)
    data_backend: str
    data_path: str
    uncertainty: str = "bald"
    clusters: Dict[str, List[AnnotationJSON]]
    cluster_order: List[int]
    selected_cluster_id: Optional[int]
    annotation_method: str = "Local Uncertainty Maxima"

    @validator("selected_cluster_id")
    def check_selected(cls, v, values):  # noqa: D401
        if v is not None and str(v) not in values["clusters"]:
            raise ValueError("selected_cluster_id not present in clusters")
        return v

    def to_json(self, *, indent: int = 4) -> str:
        """
        Version‑agnostic JSON dump: works with pydantic v1 and v2.
        """
        if hasattr(self, "model_dump_json"):  # v2+
            return self.model_dump_json(indent=indent)
        # pydantic v1 fallback
        return super().json(indent=indent)

    @classmethod
    def from_json(cls, s: Union[str, bytes]) -> "ProjectState":
        return cls.parse_raw(s)


# ---------------------------------------------------------------------------
#  Codec helpers -------------------------------------------------------------
# ---------------------------------------------------------------------------

def _encode(data: str, level: int) -> bytes:
    if _ZSTD_AVAILABLE:
        cctx = zstd.ZstdCompressor(level=max(1, min(level, 22)))
        return cctx.compress(data.encode())
    return _gzip_fallback.compress(data.encode(), compresslevel=level)


def _decode(buffer: bytes) -> str:
    if _ZSTD_AVAILABLE and buffer[:4] == b"\x28\xb5\x2f\xfd":  # zstd magic
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(buffer).decode()
    return _gzip_fallback.decompress(buffer).decode()


# ---------------------------------------------------------------------------
#  Migration framework -------------------------------------------------------
# ---------------------------------------------------------------------------

MigrationFn = Callable[[dict], dict]


def _v2_to_v3(raw: dict) -> dict:  # example migration
    """Upgrade schema v2 → v3.

    Changes:
    * "cluster_order" stored as list[str] → convert to list[int]
    * ensure "uncertainty" key exists (default "bald").
    """
    raw["cluster_order"] = [int(x) for x in raw.get("cluster_order", [])]
    raw.setdefault("uncertainty", "bald")
    raw["schema_version"] = 3
    return raw


def _v3_to_v4(raw: dict) -> dict:
    """
    Upgrade schema v3 → v4.

    * add key "annotation_method" with default
    """
    raw.setdefault("annotation_method", "Local Uncertainty Maxima")
    raw["schema_version"] = 4
    return raw


MIGRATIONS: Dict[int, MigrationFn] = {
    2: _v2_to_v3,
    3: _v3_to_v4,
}

def migrate(raw: dict) -> dict:  # noqa: D401
    """Upgrade *raw* in‑place to LATEST_SCHEMA_VERSION.

    Controllers may pass any historic file; this function applies each
    step sequentially so that no migration ever needs to handle more
    than one version delta.
    """
    v = raw.get("schema_version", 1)
    while v < LATEST_SCHEMA_VERSION:
        try:
            f = MIGRATIONS[v]
        except KeyError as e:  # pragma: no cover
            raise RuntimeError(f"No migration defined for schema v{v} → v{v + 1}") from e
        raw = f(raw)
        v = raw["schema_version"]
    return raw


# ---------------------------------------------------------------------------
#  Public I/O ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def save_state(state: ProjectState, path: Path, *, level: int = 3) -> None:
    """Serialize *state* to *path* atomically."""
    logging.debug("Saving project state to %s (codec=%s, level=%d)", path,
                  "zstd" if _ZSTD_AVAILABLE else "gzip", level)

    data_bytes = _encode(state.to_json(), level)

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data_bytes)
        tmp.flush();
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)  # atomic rename


def load_state(path: Path) -> ProjectState:
    """Load *path*, migrate if needed, and return validated instance."""
    buffer = Path(path).read_bytes()
    raw = json.loads(_decode(buffer))
    raw = migrate(raw)
    return ProjectState.parse_obj(raw)
