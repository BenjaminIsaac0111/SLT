#!/usr/bin/env python3
"""
Application entry‑point
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QDialog,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    MIME_FILTER,
    LEGACY_FILTER,
    PROJECT_EXT,
    LEGACY_EXT,
    AUTOSAVE_DIR,
    LATEST_SCHEMA_VERSION,
)
from GUI.models.ImageDataModel import create_image_data_model
from GUI.models.io.Persistence import ProjectState
from GUI.views.AppMenuBar import AppMenuBar

    _win = _main_window(view, controller)  # Needs to be stored for persistence.
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
