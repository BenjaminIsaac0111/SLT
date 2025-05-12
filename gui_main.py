#!/usr/bin/env python3
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
    QMessageBox, QAction, QActionGroup,
)

from GUI.configuration.configuration import MIME_FILTER, LEGACY_FILTER, PROJECT_EXT, LEGACY_EXT, AUTOSAVE_DIR, \
    LATEST_SCHEMA_VERSION
from GUI.models.ImageDataModel import create_image_data_model
from GUI.models.StatePersistance import ProjectState

# -------------------------------------------------------------------- Qt init
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


# -------------------------------------------------------------------- logging
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# -------------------------------------------------------------------- dialog
class StartupDialog(QDialog):
    def __init__(self, autosave_exists: bool, icon_path: str):
        super().__init__()
        self.selected_option: str | None = None
        self.project_file: str | None = None
        self.database: str | None = None

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
def _app() -> QApplication:
    return QApplication(sys.argv)


def _set_app_icon(app: QApplication, icon_path: str):
    if Path(icon_path).exists():
        app.setWindowIcon(QIcon(icon_path))
    else:
        logging.warning("Icon not found at %s", icon_path)


def _latest_autosave() -> Optional[str]:
    """Return the newest autosave (.slt or legacy) or None."""
    patterns = (f"project_autosave_*{PROJECT_EXT}",
                f"project_autosave_*{LEGACY_EXT}")
    files = [p for pat in patterns for p in AUTOSAVE_DIR.glob(pat)]
    if not files:
        return None
    return str(max(files, key=lambda p: p.stat().st_mtime))


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
        controller.project_state_controller.load_project_state(autosave)

    elif opt == "load_project":
        controller.project_state_controller.load_project_state(dlg.project_file)

    elif opt == "start_new":
        state = ProjectState(
            schema_version=LATEST_SCHEMA_VERSION,
            data_backend=_backend_from_path(dlg.database),
            data_path=dlg.database,
            uncertainty="bald",
            clusters={},
            cluster_order=[],
            selected_cluster_id=None,
        )
        controller.set_model(create_image_data_model(state))
    else:
        return False
    return True


def _main_window(view: QDialog) -> QMainWindow:
    win = QMainWindow()
    tabs = QTabWidget()
    tabs.addTab(view, "Clustered Crops")
    win.setCentralWidget(tabs)

    mb = win.menuBar()  # ← NEW

    # ---------------------------- File menu
    file_menu = mb.addMenu("&File")
    act_load = file_menu.addAction("Load Project…")
    act_save = file_menu.addAction("Save")
    act_save_as = file_menu.addAction("Save As…")
    file_menu.addSeparator()
    act_export = file_menu.addAction("Export Annotations…")

    act_load.triggered.connect(view.load_project_state_requested)
    act_save.triggered.connect(view.save_project_requested)
    act_save_as.triggered.connect(view.save_project_as_requested)
    act_export.triggered.connect(view.export_annotations_requested)

    ann_menu = mb.addMenu("Annotation Method")
    ann_group = QActionGroup(win)
    for label in ["Local Uncertainty Maxima", "Equidistant Spots", "Image Centre"]:
        act = QAction(label, win, checkable=True)
        ann_group.addAction(act)
        ann_menu.addAction(act)
    ann_group.actions()[0].setChecked(True)  # default selection
    ann_group.triggered.connect(
        lambda a: view.annotation_method_changed.emit(a.text())
    )

    win.setWindowTitle("Guided Labelling Tool")
    win.resize(1920, 1080)

    center = QApplication.primaryScreen().availableGeometry().center()
    geo = win.frameGeometry()
    geo.moveCenter(center)
    win.move(geo.topLeft())
    win.show()
    return win


# -------------------------------------------------------------------- main
def main() -> None:
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

    _win = _main_window(view)  # Needs to be stored for persistence.
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
