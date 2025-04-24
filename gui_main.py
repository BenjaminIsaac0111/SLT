#!/usr/bin/env python3
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QDialog,
    QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
)

# Import factory for both HDF5 and SQLite backends
from GUI.models.ImageDataModel import create_image_data_model

# Enable high DPI scaling before QApplication instantiation
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


def setup_logging():
    """
    Configures the logging settings for the application.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    logging.debug("Logging has been configured.")


class StartupDialog(QDialog):
    """
    Dialog prompting the user to either continue the last session,
    load a project, or start a new project.
    """

    def __init__(self, autosave_file_exists: bool, icon_path: str):
        super().__init__()
        self.selected_option = None
        self.project_file = None
        self.database = None

        self.setWindowTitle("Welcome to Smart Annotation Tool")
        self.setFixedSize(400, 200)

        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            logging.warning(f"Icon file not found at {icon_path}. Proceeding without an icon.")

        layout = QVBoxLayout()
        layout.addWidget(
            QLabel("Please choose how you'd like to start:", alignment=Qt.AlignCenter)
        )

        self.continue_button = QPushButton("Continue Last Session")
        self.continue_button.setEnabled(autosave_file_exists)
        self.continue_button.clicked.connect(self._continue_last_session)
        layout.addWidget(self.continue_button)

        load_button = QPushButton("Load Project")
        load_button.clicked.connect(self._load_project)
        layout.addWidget(load_button)

        new_button = QPushButton("Start New Project")
        new_button.clicked.connect(self._start_new_project)
        layout.addWidget(new_button)

        self.setLayout(layout)

    def _continue_last_session(self):
        self.selected_option = "continue_last"
        self.accept()

    def _load_project(self):
        options = QFileDialog.Options()
        project_file, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            "",
            "Compressed JSON Files (*.json.gz);;All Files (*)",
            options=options
        )
        if project_file:
            self.selected_option = "load_project"
            self.project_file = project_file
            self.accept()

    def _start_new_project(self):
        options = QFileDialog.Options()
        data_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "HDF5 & SQLite Files (*.h5;*.hdf5;*.sqlite;*.db);;All Files (*)",
            options=options
        )
        if data_file:
            self.selected_option = "start_new"
            self.database = data_file
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "No data file selected.")


def _initialize_qt_app() -> QApplication:
    """
    Initializes the Qt Application with high DPI settings.
    """
    app = QApplication(sys.argv)
    return app


def _set_app_icon(app: QApplication, icon_path: str):
    """
    Tries to set the application icon; logs a warning if missing.
    """
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        logging.warning(f"Icon file not found at {icon_path}. Proceeding without an icon.")


def _check_latest_autosave() -> Optional[str]:
    """
    Returns the path to the latest autosave JSON or None.
    """
    temp_dir = os.path.join(tempfile.gettempdir(), 'SLT_Temp')
    os.makedirs(temp_dir, exist_ok=True)
    autosave_files = [
        f for f in os.listdir(temp_dir)
        if f.startswith('project_autosave') and f.endswith('.json.gz')
    ]
    if not autosave_files:
        return None
    autosave_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(temp_dir, f)),
        reverse=True
    )
    return os.path.join(temp_dir, autosave_files[0])


# ------------------------------------------------------------------ helpers
def _backend_from_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".h5", ".hdf5"}:
        return "hdf5"
    if ext in {".sqlite", ".db"}:
        return "sqlite"
    raise ValueError(f"Unsupported data file type: {ext}")


def _show_startup_dialog(latest_autosave_file: Optional[str], controller, icon_path: str) -> bool:
    """
    Shows startup dialog and sets up the model accordingly.
    Returns False if the user cancels.
    """

    dialog = StartupDialog(autosave_file_exists=bool(latest_autosave_file), icon_path=icon_path)
    if dialog.exec_() != QDialog.Accepted:
        return False

    if dialog.selected_option == "continue_last" and latest_autosave_file:
        controller.project_state_controller.load_project_state(latest_autosave_file)
    elif dialog.selected_option == "load_project":
        controller.project_state_controller.load_project_state(dialog.project_file)
    elif dialog.selected_option == "start_new":
        data_path = dialog.database
        project_state = {
            "data_backend": _backend_from_path(data_path),
            "data_path": data_path,
            "uncertainty": "bald",
            "clusters": None,  # empty project
        }
        controller.set_model(create_image_data_model(project_state))
    else:
        return False

    return True


def _create_main_window(clustered_crops_view: QDialog) -> QMainWindow:
    """
    Creates and returns the main QMainWindow.
    """
    main_window = QMainWindow()
    tab_widget = QTabWidget()
    tab_widget.addTab(clustered_crops_view, "Clustered Crops")
    main_window.setCentralWidget(tab_widget)
    main_window.setWindowTitle("Guided Labelling Tool")
    main_window.resize(1920, 1080)
    frame_geom = main_window.frameGeometry()
    screen_center = QApplication.primaryScreen().availableGeometry().center()
    frame_geom.moveCenter(screen_center)
    main_window.move(frame_geom.topLeft())
    main_window.show()
    return main_window


def main():
    app = _initialize_qt_app()
    icon_path = "GUI/assets/icons/icons8-point-100.png"
    _set_app_icon(app, icon_path)

    from GUI.views.ClusteredCropsView import ClusteredCropsView
    from GUI.controllers.MainController import MainController

    setup_logging()
    clustered_crops_view = ClusteredCropsView()
    controller = MainController(model=None, view=clustered_crops_view)

    latest_autosave = _check_latest_autosave()
    if not _show_startup_dialog(latest_autosave, controller, icon_path):
        sys.exit()

    main_window = _create_main_window(clustered_crops_view)  # Retain instance.
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
