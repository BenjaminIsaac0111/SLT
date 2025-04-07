import gzip
import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Optional, List

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThreadPool

from GUI.models.ImageDataModel import ImageDataModel
from GUI.workers.AutosaveWorker import AutosaveWorker

TEMP_DIR = os.path.join(tempfile.gettempdir(), 'SLT_Temp')
os.makedirs(TEMP_DIR, exist_ok=True)


class ProjectStateController(QObject):
    """
    Manages saving and loading of the project state, including autosave
    functionality, restoring from autosave files, and versioned backups.
    """

    # -------------------------------------------------------------------------
    #                                 SIGNALS
    # -------------------------------------------------------------------------
    autosave_finished = pyqtSignal(bool)
    project_loaded = pyqtSignal(dict)
    project_saved = pyqtSignal(str)
    save_failed = pyqtSignal(str)
    load_failed = pyqtSignal(str)

    # -------------------------------------------------------------------------
    #                                INIT
    # -------------------------------------------------------------------------
    def __init__(self, model: ImageDataModel):
        """
        :param model: The ImageDataModel in use.
        """
        super().__init__()
        self.model = model

        self.is_saving = False
        self.current_save_path: Optional[str] = None

        # Use a QThreadPool instead of a dedicated QThread for autosaves
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(1)

        # Temporary file path for autosave
        self.temp_file_path = os.path.join(TEMP_DIR, 'project_autosave.json.gz')
        logging.info(f"Temporary autosave file path: {self.temp_file_path}")

    # -------------------------------------------------------------------------
    #                        PUBLIC METHODS
    # -------------------------------------------------------------------------
    def set_current_save_path(self, file_path: str):
        """
        Sets the file path for saving the project.
        """
        self.current_save_path = file_path
        logging.info(f"Current save path set to: {file_path}")

    def get_current_save_path(self) -> Optional[str]:
        """
        Returns the current file path for the project save.
        """
        return self.current_save_path

    def autosave_project_state(self, project_state: dict):
        """
        Autosaves the project state using a QRunnable-based worker.
        """
        if self.is_saving:
            logging.info("Autosave already in progress; skipping this request.")
            return

        if self._is_project_state_empty(project_state):
            logging.info("No annotations to save; skipping autosave.")
            return

        self.is_saving = True
        versioned_backup_path = self.get_versioned_backup_path(
            self.temp_file_path, max_backups=10
        )

        # Create a new AutosaveWorker for each autosave request
        worker = AutosaveWorker(project_state, versioned_backup_path)
        # Connect the worker's 'save_finished' signal to our slot
        worker.signals.save_finished.connect(self.on_autosave_finished)

        # Start the worker on the thread pool
        self.threadpool.start(worker)

    @pyqtSlot(bool)
    def on_autosave_finished(self, success: bool):
        """
        Called when AutosaveWorker completes (signals.save_finished).
        """
        self.is_saving = False
        self.autosave_finished.emit(success)

    def save_project_state(self, project_state: dict, file_path: str):
        """
        Saves the project state to a specified path (e.g., from "Save" or "Save As").
        """
        try:
            self._save_to_file(project_state, file_path)
            logging.info(f"Project state saved to {file_path}")
            self.project_saved.emit(file_path)
        except (TypeError, IOError) as e:
            logging.error(f"Failed to save project state: {e}")
            self.save_failed.emit(str(e))

    def load_project_state(self, project_file: str):
        """
        Loads the project state from a file to resume the session.
        """
        if not os.path.exists(project_file):
            msg = f"No project file found at {project_file}."
            logging.error(msg)
            self.load_failed.emit(msg)
            return

        try:
            project_state = self._load_from_file(project_file)
            logging.info(f"Project state loaded from {project_file}")
            self.project_loaded.emit(project_state)
        except (IOError, json.JSONDecodeError, KeyError, ValueError) as e:
            err_msg = f"Failed to load project from {project_file}: {e}"
            logging.error(err_msg)
            self.load_failed.emit(str(e))

    def get_versioned_backup_path(self, base_path: str, max_backups: int = 10) -> str:
        """
        Creates a new backup path containing a timestamp and cleans old backups.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(base_path)
        backup_filename = f"{base_name}_{timestamp}.json.gz"
        backup_path = os.path.join(TEMP_DIR, backup_filename)

        self._remove_old_backups(base_name, max_backups)
        return backup_path

    def get_latest_autosave_file(self) -> Optional[str]:
        """
        Returns the most recently modified autosave file in TEMP_DIR, or None.
        """
        autosave_files = [
            f for f in os.listdir(TEMP_DIR)
            if f.startswith('project_autosave') and f.endswith('.json.gz')
        ]
        if not autosave_files:
            return None

        autosave_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(TEMP_DIR, f)),
            reverse=True
        )
        return os.path.join(TEMP_DIR, autosave_files[0])

    def get_autosave_files(self) -> List[str]:
        """
        Lists all autosave files in TEMP_DIR (most recent first).
        """
        autosave_files = [
            f for f in os.listdir(TEMP_DIR)
            if f.startswith('project_autosave') and f.endswith('.json.gz')
        ]
        if not autosave_files:
            return []

        autosave_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(TEMP_DIR, f)),
            reverse=True
        )
        return [os.path.join(TEMP_DIR, f) for f in autosave_files]

    def cleanup(self):
        logging.info("Cleaning up autosave before application exit.")

        if self.is_saving:
            logging.info("Autosave is in progress; waiting for completion.")
            # Wait indefinitely until all worker tasks are done
            self.threadpool.waitForDone()

        logging.info("ProjectStateController cleanup done.")

    # -------------------------------------------------------------------------
    #                            PRIVATE HELPERS
    # -------------------------------------------------------------------------
    @staticmethod
    def _is_project_state_empty(project_state: dict) -> bool:
        return not project_state.get('annotations')

    @staticmethod
    def _save_to_file(project_state: dict, file_path: str):
        with gzip.open(file_path, 'wt') as gzfile:
            json.dump(project_state, gzfile, indent=4)

    @staticmethod
    def _load_from_file(project_file: str) -> dict:
        with gzip.open(project_file, 'rt') as gzfile:
            return json.load(gzfile)

    def _remove_old_backups(self, base_name: str, max_backups: int):
        all_backups = sorted([
            f for f in os.listdir(TEMP_DIR)
            if f.startswith(base_name) and f.endswith(".json.gz")
        ])
        while len(all_backups) > max_backups:
            old_backup = all_backups.pop(0)
            os.remove(os.path.join(TEMP_DIR, old_backup))
            logging.info(f"Deleted old backup file: {old_backup}")
