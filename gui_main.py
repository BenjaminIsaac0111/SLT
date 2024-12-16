import logging
import os
import sys
import tempfile

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox

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
            logging.StreamHandler(sys.stdout)  # Ensure that logs are sent to the console
            # You can add more handlers here if needed, e.g., FileHandler
        ]
    )
    logging.debug("Logging has been configured.")


class StartupDialog(QDialog):
    def __init__(self, autosave_file_exists):
        super().__init__()
        self.selected_option = None  # Track which option the user selected

        self.setWindowTitle("Welcome to Smart Annotation Tool")
        self.setFixedSize(400, 200)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Please choose how you'd like to start:", alignment=Qt.AlignCenter))

        if autosave_file_exists:
            continue_button = QPushButton("Continue Last Session")
            continue_button.clicked.connect(self.continue_last_session)
            layout.addWidget(continue_button)

        load_button = QPushButton("Load Project")
        load_button.clicked.connect(self.load_project)
        layout.addWidget(load_button)

        new_button = QPushButton("Start New Project")
        new_button.clicked.connect(self.start_new_project)
        layout.addWidget(new_button)

        self.setLayout(layout)

    def continue_last_session(self):
        self.selected_option = "continue_last"
        self.accept()

    def load_project(self):
        options = QFileDialog.Options()
        # Update the file filter to include compressed and uncompressed JSON files
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

    def start_new_project(self):
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


def main():
    app = QApplication(sys.argv)

    # Set the application icon
    icon_path = f"GUI/assets/icons/icons8-point-100.png"
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        logging.warning(f"Icon file not found at {icon_path}. Application will proceed without an icon.")

    # Initialize views and controller
    from GUI.views.ClusteredCropsView import ClusteredCropsView
    from GUI.controllers.GlobalClusterController import GlobalClusterController
    from GUI.models.ImageDataModel import ImageDataModel

    clustered_crops_view = ClusteredCropsView()

    # Initialize the model to None
    model = None
    global_cluster_controller = GlobalClusterController(model=model, view=clustered_crops_view)

    # Check for the latest autosave file
    temp_dir = os.path.join(tempfile.gettempdir(), 'my_application_temp')
    os.makedirs(temp_dir, exist_ok=True)
    autosave_files = [f for f in os.listdir(temp_dir)
                      if f.startswith('project_autosave') and f.endswith('.json.gz')]
    latest_autosave_file = None
    if autosave_files:
        autosave_files.sort(key=lambda f: os.path.getmtime(os.path.join(temp_dir, f)), reverse=True)
        latest_autosave_file = os.path.join(temp_dir, autosave_files[0])

    # Show the startup dialog
    startup_dialog = StartupDialog(autosave_file_exists=bool(latest_autosave_file))
    if startup_dialog.exec_() == QDialog.Accepted:
        if startup_dialog.selected_option == "continue_last" and latest_autosave_file:
            # Continue with last autosaved project
            global_cluster_controller.project_state_controller.load_project_state(latest_autosave_file)
        elif startup_dialog.selected_option == "load_project":
            # Load selected project file
            project_file = startup_dialog.project_file
            global_cluster_controller.project_state_controller.load_project_state(project_file)
            # The model will be set in on_project_loaded
        elif startup_dialog.selected_option == "start_new":
            # Start new project with selected HDF5 file
            hdf5_file_path = startup_dialog.hdf5_file
            model = ImageDataModel(hdf5_file_path, 'variance')
            global_cluster_controller.set_model(model)
        else:
            sys.exit()
    else:
        sys.exit()

    # Create the main application window
    main_window = QMainWindow()
    tab_widget = QTabWidget()
    tab_widget.addTab(clustered_crops_view, "Clustered Crops")
    main_window.setCentralWidget(tab_widget)
    main_window.setWindowTitle("Guided Labelling Tool")
    main_window.resize(1920, 1080)
    main_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    setup_logging()
    main()
