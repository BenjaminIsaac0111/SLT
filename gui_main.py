import logging
import os
import sys
import tempfile

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QDialog,
    QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
)

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


def setup_logging():
    """
    Configures the logging settings for the application.
    """
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level to DEBUG
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # Logs to console
            # Additional handlers (e.g., FileHandler) can be added as needed
        ]
    )
    logging.debug("Logging has been configured.")


class StartupDialog(QDialog):
    """
    Dialog prompting the user to either continue the last session,
    load a project, or start a new project.
    """

    def __init__(self, autosave_file_exists: bool, icon_path: str):
        super().__init__()
        self.selected_option = None  # Track the user’s selection
        self.project_file = None
        self.hdf5_file = None

        self.setWindowTitle("Welcome to Smart Annotation Tool")
        self.setFixedSize(400, 200)

        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            logging.warning(f"Icon file not found at {icon_path}. Proceeding without an icon.")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Please choose how you'd like to start:", alignment=Qt.AlignCenter))

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
        # Allow compressed and uncompressed JSON files
        project_file, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            "",
            "Compressed JSON Files (*.json.gz);;JSON Files (*.json);;All Files (*)",
            options=options
        )
        if project_file:
            self.selected_option = "load_project"
            self.project_file = project_file
            self.accept()

    def _start_new_project(self):
        options = QFileDialog.Options()
        hdf5_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 File",
            "",
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
            options=options
        )
        if hdf5_file:
            self.selected_option = "start_new"
            self.hdf5_file = hdf5_file
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "No HDF5 file selected.")


def _initialize_qt_app() -> QApplication:
    """
    Initializes the Qt Application with high DPI settings.
    """
    app = QApplication(sys.argv)
    return app


def _set_app_icon(app: QApplication, icon_path: str):
    """
    Tries to set the application icon; logs a warning if the file doesn't exist.
    """
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        logging.warning(f"Icon file not found at {icon_path}. Proceeding without an icon.")


def _check_latest_autosave() -> str:
    """
    Checks for the latest autosave file in a temp directory; returns its path or None.
    """
    temp_dir = os.path.join(tempfile.gettempdir(), 'SLT_Temp')
    os.makedirs(temp_dir, exist_ok=True)

    autosave_files = [
        f for f in os.listdir(temp_dir)
        if f.startswith('project_autosave') and f.endswith('.json.gz')
    ]

    if not autosave_files:
        return None

    # Sort by modification time (descending)
    autosave_files.sort(key=lambda f: os.path.getmtime(os.path.join(temp_dir, f)), reverse=True)
    latest_autosave_file = os.path.join(temp_dir, autosave_files[0])
    return latest_autosave_file


def _show_startup_dialog(latest_autosave_file: str, controller, icon_path: str) -> bool:
    """
    Shows the StartupDialog and handles the user's choice.
    Returns False if the user canceled the dialog (exiting the app).
    """
    from GUI.models.ImageDataModel import ImageDataModel  # Imported here to avoid circular imports

    dialog = StartupDialog(autosave_file_exists=bool(latest_autosave_file), icon_path=icon_path)
    if dialog.exec_() != QDialog.Accepted:
        return False  # User canceled

    if dialog.selected_option == "continue_last" and latest_autosave_file:
        controller.project_state_controller.load_project_state(latest_autosave_file)
    elif dialog.selected_option == "load_project":
        project_file = dialog.project_file
        controller.project_state_controller.load_project_state(project_file)
    elif dialog.selected_option == "start_new":
        hdf5_file_path = dialog.hdf5_file
        model = ImageDataModel(hdf5_file_path, 'variance')
        controller.set_model(model)
    else:
        return False  # No valid selection or canceled

    return True


def _create_main_window(clustered_crops_view: QDialog) -> QMainWindow:
    """
    Creates and returns the main QMainWindow with a QTabWidget,
    and centers the main window on the screen.
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
    # 1) Set up logging
    setup_logging()

    # 2) Initialize the Qt Application
    app = _initialize_qt_app()

    # 3) Attempt to set the app icon
    icon_path = "GUI/assets/icons/icons8-point-100.png"
    _set_app_icon(app, icon_path)

    # 4) Import classes now that QApplication is instantiated
    from GUI.views.ClusteredCropsView import ClusteredCropsView
    from GUI.controllers.MainController import MainController

    # 5) Create view and controller, with model initially None
    clustered_crops_view = ClusteredCropsView()
    global_cluster_controller = MainController(model=None, view=clustered_crops_view)

    # 6) Check for autosave
    latest_autosave_file = _check_latest_autosave()

    # 7) Show startup dialog
    if not _show_startup_dialog(latest_autosave_file, global_cluster_controller, icon_path):
        sys.exit()  # Exit if user canceled or no selection

    # 8) Create and show the main window
    main_window = _create_main_window(clustered_crops_view)

    # 9) Execute the app
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
