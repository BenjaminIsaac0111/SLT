import gzip
import json
import logging
import os

from PyQt5.QtCore import QRunnable, QObject, pyqtSignal


class AutosaveWorkerSignals(QObject):
    """
    Defines the signals available from the autosave worker.
    Because QRunnable is not a QObject, we store signals in a separate object.
    """
    save_finished = pyqtSignal(bool)


class AutosaveWorker(QRunnable):
    """
    A QRunnable-based worker that saves a project state to a file asynchronously
    using compression. Designed to be used with QThreadPool.
    """

    def __init__(self, project_state: dict, file_path: str):
        super().__init__()
        self.signals = AutosaveWorkerSignals()
        self.project_state = project_state
        self.file_path = file_path

    def run(self):
        """
        Performs the autosave operation in a background thread (managed by QThreadPool).
        Emits save_finished (True/False) upon completion or error.
        """
        try:
            temp_file_path = self.file_path + '.tmp'
            with gzip.open(temp_file_path, 'wt') as f:
                json.dump(self.project_state, f, indent=4)

            # Safely replace the final file with the temp
            os.replace(temp_file_path, self.file_path)

            logging.info(f"Autosave completed successfully to {self.file_path}")
            self.signals.save_finished.emit(True)

        except (TypeError, OSError, json.JSONDecodeError) as e:
            logging.error(f"Error during autosave: {e}")
            self.signals.save_finished.emit(False)
        except Exception as e:
            logging.error(f"Unexpected exception during autosave: {e}")
            self.signals.save_finished.emit(False)
