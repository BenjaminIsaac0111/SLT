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
    QMessageBox,
)

from GUI.configuration.configuration import (
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

# -------------------------------------------------------------------- Qt init
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


# -------------------------------------------------------------------- logging

def setup_logging() -> None:  # noqa: D401 – imperative style
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# -------------------------------------------------------------------- widgets

class StartupDialog(QDialog):
    """Initial prompt for *continue* /
    *load project* / *start new*."""

    def __init__(self, autosave_exists: bool, icon_path: str):
        super().__init__()
        self.selected_option: Optional[str] = None
        self.project_file: Optional[str] = None
        self.database: Optional[str] = None

        self.setWindowTitle("Welcome to Smart Annotation Tool")
        self.setFixedSize(400, 200)
        if Path(icon_path).exists():
            self.setWindowIcon(QIcon(icon_path))

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Please choose how you'd like to start:", alignment=Qt.AlignCenter))

        cont = QPushButton("Continue Last Session")
        cont.setEnabled(autosave_exists)
        cont.clicked.connect(self._choose_continue)
        lay.addWidget(cont)

        load = QPushButton("Load Project")
        load.clicked.connect(self._choose_load)
        lay.addWidget(load)

        new = QPushButton("Start New Project")
        new.clicked.connect(self._choose_new)
        lay.addWidget(new)

    # ------------ callbacks ----------------------------------------
    def _choose_continue(self):
        self.selected_option = "continue_last"
        self.accept()

    def _choose_load(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", f"{MIME_FILTER};;{LEGACY_FILTER};;All Files (*)"
        )
        if file:
            self.selected_option, self.project_file = "load_project", file
            self.accept()

    def _choose_new(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "HDF5 & SQLite Files (*.h5;*.hdf5;*.sqlite;*.db);;All Files (*)",
        )
        if file:
            self.selected_option, self.database = "start_new", file
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "No data file selected.")


# -------------------------------------------------------------------- helpers

def _app() -> QApplication:  # noqa: D401
    return QApplication(sys.argv)


def _set_app_icon(app: QApplication, icon_path: str):
    if Path(icon_path).exists():
        app.setWindowIcon(QIcon(icon_path))
    else:
        logging.warning("Icon not found at %s", icon_path)


def _latest_autosave() -> Optional[str]:
    patterns = (
        f"project_autosave_*{PROJECT_EXT}",
        f"project_autosave_*{LEGACY_EXT}",
    )
    files = [p for pat in patterns for p in AUTOSAVE_DIR.glob(pat)]
    return str(max(files, key=lambda p: p.stat().st_mtime)) if files else None


def _backend_from_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".h5", ".hdf5"}:
        return "hdf5"
    if ext in {".sqlite", ".db"}:
        return "sqlite"
    raise ValueError(f"Unsupported data file type: {ext}")


def _startup_dialog(autosave: Optional[str], controller, icon_path: str) -> bool:
    dlg = StartupDialog(bool(autosave), icon_path)
    if dlg.exec_() != QDialog.Accepted:
        return False

    opt = dlg.selected_option
    if opt == "continue_last" and autosave:
        controller.load_project(autosave)

    elif opt == "load_project":
        controller.load_project(dlg.project_file)

    elif opt == "start_new":
        state = ProjectState(
            schema_version=LATEST_SCHEMA_VERSION,
            data_backend=_backend_from_path(dlg.database),
            data_path=dlg.database,
            uncertainty="bald",
            clusters={},
            cluster_order=[],
            selected_cluster_id=None,
            annotation_method="Local Uncertainty Maxima"
        )
        controller.set_model(create_image_data_model(state))
    else:
        return False
    return True


# -------------------------------------------------------------------- main‑window factory

def _main_window(view, controller) -> QMainWindow:  # noqa: D401 – imperative
    win = QMainWindow()
    tabs = QTabWidget()
    tabs.addTab(view, "Clustered Crops")
    win.setCentralWidget(tabs)

    # ---- menu bar --------------------------------------------------
    mb = AppMenuBar(win)
    win.setMenuBar(mb)

    # view connections
    mb.request_generate_annos.connect(view.request_clustering.emit)
    mb.request_set_ann_method.connect(view.annotation_method_changed.emit)
    mb.request_export_annotations.connect(view.export_annotations_requested)
    mb.request_annotation_preview.connect(controller.show_annotation_preview)

    # controller connections
    mb.request_save_project.connect(controller.save_project)
    mb.request_save_project_as.connect(controller.save_project_as)  # expects slot(path)
    mb.request_load_project.connect(controller.load_project)  # slot(path)
    mb.request_export_annotations.connect(controller.export_annotations)
    mb.request_build_cv_folds.connect(controller.build_cross_validation_folds)
    mb.request_set_nav_policy.connect(controller.on_navigation_policy_changed)


    win.setWindowTitle("Smart Labelling Tool")
    win.resize(1920, 1080)

    # centre on primary screen
    center = QApplication.primaryScreen().availableGeometry().center()
    geo = win.frameGeometry()
    geo.moveCenter(center)
    win.move(geo.topLeft())
    win.show()
    return win


# -------------------------------------------------------------------- main

def main() -> None:  # noqa: D401
    setup_logging()
    app = _app()
    icon = "GUI/assets/icons/icons8-point-100.png"
    _set_app_icon(app, icon)

    from GUI.views.ClusteredCropsView import ClusteredCropsView
    from GUI.controllers.MainController import MainController

    view = ClusteredCropsView()
    controller = MainController(model=None, view=view)

    latest = _latest_autosave()
    if not _startup_dialog(latest, controller, icon):
        sys.exit()

    _win = _main_window(view, controller)  # Needs to be stored for persistence.
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
