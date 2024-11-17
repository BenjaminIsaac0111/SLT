import gzip
import json
import logging
import os

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class AutosaveWorker(QObject):
    save_finished = pyqtSignal(bool)
    save_project_state_signal = pyqtSignal(object, str)

    def __init__(self):
        super().__init__()
        self.save_project_state_signal.connect(self.save_project_state)

    @pyqtSlot(object, str)
    def save_project_state(self, project_state, file_path):
        """
        Saves the project state to the specified file path asynchronously, using compression.
        """
        try:
            temp_file_path = file_path + '.tmp'
            with gzip.open(temp_file_path, 'wt') as f:
                json.dump(project_state, f, indent=4)
            os.replace(temp_file_path, file_path)
            logging.info(f"Autosave completed successfully to {file_path}")
            self.save_finished.emit(True)  # Signal successful save
        except TypeError as e:
            logging.error(f"Serialization error during autosave: {e}")
            self.save_finished.emit(False)
        except Exception as e:
            logging.error(f"Error during async autosave: {e}")
            self.save_finished.emit(False)  # Signal failed save
