import logging
import os

from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget, QProgressBar, \
    QTableWidget, QTableWidgetItem, QFileDialog

import numpy as np
import pandas as pd

from tensorflow.keras import mixed_precision
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mixed_precision.set_global_policy('mixed_float16')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ResultsAnalyserWidget(QWidget):
    def __init__(self, *args, cfg=None):
        super(ResultsAnalyserWidget, self).__init__(*args)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'Initializing Model Evaluator Widget with config: {cfg}')
        self.cfg = cfg
        self.update_anomalies_memory_bank = None
        self.layout = None
        self.worker = None
        self.thread = None
        self.reportTable = None
        self.progressBar = None
        self.evaluateButton = None
        self.reportDisplay = None
        self.statusLabel = None
        self.dl_model = None
        self.memory_bank = None
        self._initUI()

    def _initButtons(self):
        self.visualiseButton = QPushButton('Visualise Feature Space', self)
        self.visualiseButton.clicked.connect(self.visualise_features)
        self.visualiseButton.setEnabled(False)
        self.layout.addWidget(self.visualiseButton)

    def _initUI(self):
        # Main Layout
        self.layout = QVBoxLayout()

        # Classification Report Table
        self.reportTable = QTableWidget(self)
        self.layout.addWidget(self.reportTable)

        self._initButtons()

        if self.load_memory_bank():
            self.visualiseButton.setEnabled(True)

        self.progressBar = QProgressBar(self)
        self.progressBar.hide()
        self.layout.addWidget(self.progressBar)

        # Set the layout for the widget
        self.setLayout(self.layout)

    def _updateProgressBar(self, value):
        self.progressBar.setValue(value)

    def load_memory_bank(self, prefix=''):
        try:
            output_dir = self.cfg['OUTPUT_DIR'].rstrip('/')
            initial_dir = output_dir
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            memory_bank_file, _ = QFileDialog.getOpenFileName(self, "Select Memory Bank File", initial_dir,
                                                              "Numpy Files (*.npy)", options=options)

            if memory_bank_file:
                self.memory_bank = np.load(memory_bank_file)
                self.display_classification_report()
                self.logger.info(f"Memory bank loaded from {memory_bank_file}")
                return True
            else:
                self.logger.info("No file selected.")
                return False
        except Exception as e:
            self.logger.error(f"Error loading memory bank: {e}")
            return False

    def display_classification_report(self):
        try:
            true_labels = self.memory_bank[:, 0].astype(int)
            predicted_labels = self.memory_bank[:, 1].astype(int)

            # Compute classification report for the initial classes
            clf_report = classification_report(
                true_labels,
                predicted_labels,
                output_dict=True,
                zero_division=True
            )

            actual_anomalies = true_labels != predicted_labels
            predicted_anomalies = self.memory_bank[:, -1].astype(int)

            # Calculate metrics
            precision = precision_score(actual_anomalies, predicted_anomalies, zero_division=True)
            recall = recall_score(actual_anomalies, predicted_anomalies)
            f1 = f1_score(actual_anomalies, predicted_anomalies)
            support = actual_anomalies.sum()  # Number of actual anomalies
            accuracy = np.mean(actual_anomalies == predicted_anomalies)

            # Add anomaly detection metrics to the classification report
            clf_report['anomaly_detection'] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': support,
                'accuracy': accuracy
            }

        except Exception as e:
            # Handle potential exceptions, possibly log them or notify the user
            print(f"An error occurred: {e}")

        clf_report_df = pd.DataFrame(clf_report).transpose()
        clf_report_df.index = clf_report_df.index.map(
            lambda x: self.cfg['CLASS_COMPONENTS'][int(x)] if x.isdigit() else x)
        clf_report_df = clf_report_df.reset_index()

        self.reportTable.setRowCount(clf_report_df.shape[0])
        self.reportTable.setColumnCount(clf_report_df.shape[1])
        self.reportTable.setHorizontalHeaderLabels(clf_report_df.columns)

        # Populate the table with data
        for i in range(clf_report_df.shape[0]):
            for j in range(clf_report_df.shape[1]):
                item = QTableWidgetItem(str(clf_report_df.iloc[i, j]))
                self.reportTable.setItem(i, j, item)

        self.reportTable.resizeColumnsToContents()
        self.reportTable.resizeRowsToContents()

    def visualise_features(self):
        # Create the thread and worker
        self.visualiseButton.setEnabled(False)
        self.progressBar.setRange(0, 0)
        self.progressBar.show()

        # Connect signals
        self.thread.started.connect(self.umapWorker.run)
        self.umapWorker.finished.connect(self.thread.quit)
        self.umapWorker.finished.connect(self.umapWorker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.umapWorker.finished.connect(self.display_umap_results)

        # Update UI elements after the UMAP computation is finished
        self.umapWorker.finished.connect(lambda: (
            self.visualiseButton.setEnabled(True),
            self.progressBar.setRange(0, 1),  # Reset progress bar to its default state
            self.progressBar.hide()  # Hide the progress bar
        ))
        self.thread.start()
