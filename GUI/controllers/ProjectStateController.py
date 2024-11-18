import gzip
import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Optional, List

from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, QEventLoop

from GUI.models.ImageDataModel import ImageDataModel
from GUI.workers.AutosaveWorker import AutosaveWorker

TEMP_DIR = os.path.join(tempfile.gettempdir(), 'my_application_temp')
os.makedirs(TEMP_DIR, exist_ok=True)


class ProjectStateController(QObject):
    """
    ProjectStateController manages saving and loading of the project state.
    It handles autosaving, restoring from autosave files, and maintains versioned backups.
    It communicates with GlobalClusterController via signals and slots.
    """

    # Signals to communicate with other controllers or the view
    autosave_finished = pyqtSignal(bool)
    project_loaded = pyqtSignal(dict)
    project_saved = pyqtSignal(str)
    save_failed = pyqtSignal(str)
    load_failed = pyqtSignal(str)

    def __init__(self, model: ImageDataModel):
        """
        Initializes the ProjectStateController.

        :param model: An instance of the ImageDataModel.
        """
        super().__init__()
        self.model = model
        self.is_saving = False
        self.current_save_path: Optional[str] = None

        # Initialize autosave worker and thread
        self.autosave_worker = AutosaveWorker()
        self.autosave_thread = QThread()
        self.autosave_worker.moveToThread(self.autosave_thread)
        self.autosave_worker.save_finished.connect(self.on_autosave_finished)
        self.autosave_thread.start()

        # Autosave file path within the dedicated temp directory
        self.temp_file_path = os.path.join(TEMP_DIR, 'project_autosave.json.gz')
        logging.info(f"Temporary autosave file path: {self.temp_file_path}")

    def set_current_save_path(self, file_path: str):
        """
        Sets the current file path for saving the project.

        :param file_path: The file path where the project will be saved.
        """
        self.current_save_path = file_path
        logging.info(f"Current save path set to: {file_path}")

    def get_current_save_path(self) -> Optional[str]:
        """
        Returns the current file path where the project is saved.

        :return: The current save path.
        """
        return self.current_save_path

    def autosave_project_state(self, project_state: dict):
        """
        Autosaves the current project state to a versioned backup file.

        :param project_state: The current state of the project.
        """
        if self.is_saving:
            logging.info("Autosave already in progress, skipping this autosave.")
            return  # Skip if a save is already in progress

        if not project_state.get('annotations'):
            logging.info("No annotations to save. Skipping autosave.")
            return  # No data to save

        self.is_saving = True

        # Create a versioned backup path
        versioned_backup_path = self.get_versioned_backup_path(self.temp_file_path, max_backups=5)

        # Send save request to worker
        self.autosave_worker.save_project_state_signal.emit(project_state, versioned_backup_path)

    @pyqtSlot(bool)
    def on_autosave_finished(self, success: bool):
        """
        Handles completion of the autosave operation.

        :param success: True if autosave was successful, False otherwise.
        """
        self.is_saving = False  # Reset the saving flag
        self.autosave_finished.emit(success)

    def save_project_state(self, project_state: dict, file_path: str):
        """
        Saves the current project state to the specified file path.

        :param project_state: The current state of the project.
        :param file_path: The file path where the project will be saved.
        """
        try:
            with gzip.open(file_path, 'wt') as f:
                json.dump(project_state, f, indent=4)
            logging.info(f"Project state saved to {file_path}")
            self.project_saved.emit(file_path)
        except (TypeError, IOError) as e:
            logging.error(f"Failed to save project state: {e}")
            self.save_failed.emit(str(e))

    def load_project_state(self, project_file: str):
        """
        Loads the project state from a saved file to resume the session.

        :param project_file: The file path of the project to load.
        """
        if not os.path.exists(project_file):
            logging.error(f"No project file found at {project_file} to load.")
            self.load_failed.emit(f"No project file found at {project_file}.")
            return

        try:
            with gzip.open(project_file, 'rt') as f:
                project_state = json.load(f)
            logging.info(f"Project state loaded from {project_file}")
            self.project_loaded.emit(project_state)
        except (IOError, json.JSONDecodeError, KeyError, ValueError) as e:
            logging.error(f"Failed to load project state from {project_file}: {e}")
            self.load_failed.emit(str(e))

    def get_versioned_backup_path(self, base_path: str, max_backups: int = 10) -> str:
        """
        Generates a versioned backup path by adding a timestamp.
        Deletes older backups if they exceed max_backups.

        :param base_path: The base file path.
        :param max_backups: Maximum number of backup files to keep.
        :return: The versioned backup file path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.basename(base_path)}_{timestamp}.json.gz"
        backup_path = os.path.join(TEMP_DIR, backup_filename)

        # Remove old backups if exceeding max_backups
        all_backups = sorted([f for f in os.listdir(TEMP_DIR)
                              if f.startswith(os.path.basename(base_path)) and f.endswith(".json.gz")])

        while len(all_backups) > max_backups:
            old_backup = all_backups.pop(0)  # Remove the oldest file
            os.remove(os.path.join(TEMP_DIR, old_backup))
            logging.info(f"Deleted old backup file: {old_backup}")

        return backup_path

    def get_latest_autosave_file(self) -> Optional[str]:
        """
        Finds the most recent autosave file in the TEMP_DIR.

        :return: The path to the latest autosave file, or None if none exist.
        """
        autosave_files = [f for f in os.listdir(TEMP_DIR)
                          if f.startswith('project_autosave') and f.endswith('.json.gz')]
        if not autosave_files:
            return None  # No autosave files found

        # Sort files by modification time in descending order
        autosave_files.sort(key=lambda f: os.path.getmtime(os.path.join(TEMP_DIR, f)), reverse=True)
        latest_autosave = autosave_files[0]
        return os.path.join(TEMP_DIR, latest_autosave)

    def get_autosave_files(self) -> List[str]:
        """
        Returns a list of available autosave files in TEMP_DIR.

        :return: A list of paths to autosave files.
        """
        autosave_files = [f for f in os.listdir(TEMP_DIR)
                          if f.startswith('project_autosave') and f.endswith('.json.gz')]
        if not autosave_files:
            return []
        # Sort files by modification time in descending order
        autosave_files.sort(key=lambda f: os.path.getmtime(os.path.join(TEMP_DIR, f)), reverse=True)
        # Return full paths
        autosave_files_full = [os.path.join(TEMP_DIR, f) for f in autosave_files]
        return autosave_files_full

    def cleanup(self):
        """
        Cleans up the autosave worker and thread before application exit.
        """
        logging.info("Cleaning up autosave thread before application exit.")
        if self.is_saving:
            logging.info("Autosave in progress. Waiting for it to finish before quitting.")
            loop = QEventLoop()
            self.autosave_worker.save_finished.connect(loop.quit)
            loop.exec_()  # This will block until `save_finished` is emitted
        # Now, safely quit the autosave thread
        self.autosave_thread.quit()
        self.autosave_thread.wait()
        logging.info("Autosave thread terminated.")
