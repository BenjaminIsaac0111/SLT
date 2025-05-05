"""
Global constants that must be imported by *multiple* sub‑modules.

Keeping them here prevents circular‑import chains like
MainController ↔ ProjectStateController.
"""

from pathlib import Path
from tempfile import gettempdir


CLASS_COMPONENTS = {
    0: 'Non-Informative',
    1: 'Tumour',
    2: 'Stroma',
    3: 'Necrosis',
    4: 'Vessel',
    5: 'Inflammation',
    6: 'Tumour-Lumen',
    7: 'Mucin',
    8: 'Muscle'
}

LATEST_SCHEMA_VERSION = 3
# ------------------------------------------------------------------ project
PROJECT_EXT = ".slt"  # default save‑file extension
MIME_FILTER = f"SLT Project (*{PROJECT_EXT})"
LEGACY_EXT = ".json.gz"
LEGACY_FILTER = f"Legacy Project (*{LEGACY_EXT})"

# directory for autosaves
AUTOSAVE_DIR = Path(gettempdir()) / "SLT_Temp"
AUTOSAVE_DIR.mkdir(exist_ok=True)
