from typing import List, Dict

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea, QGridLayout, QGroupBox, QHBoxLayout,
                             QPushButton, \
    QProgressBar, QSpinBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal
import logging


class ClusteredCropsView(QWidget):
    """
    ClusteredCropsView displays sampled zoomed-in crops from various clusters.
    """

    # Signals emitted by the view
    request_clustering = pyqtSignal()
    sample_cluster = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """
        Sets up the user interface layout for displaying clustered crops.
        """
        layout = QVBoxLayout()

        # Header with a button to start clustering
        header_layout = QHBoxLayout()
        self.cluster_button = QPushButton("Start Global Clustering")
        self.cluster_button.clicked.connect(self.on_cluster_button_clicked)
        header_layout.addWidget(self.cluster_button)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Scroll area to hold clusters and their crops
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        # Sampling controls
        sampling_layout = QHBoxLayout()
        sampling_layout.addWidget(QLabel("Number of Clusters:"))
        self.num_clusters_spinbox = QSpinBox()
        self.num_clusters_spinbox.setRange(1, 100)
        self.num_clusters_spinbox.setValue(10)
        sampling_layout.addWidget(self.num_clusters_spinbox)

        sampling_layout.addWidget(QLabel("Crops per Cluster:"))
        self.crops_per_cluster_spinbox = QSpinBox()
        self.crops_per_cluster_spinbox.setRange(1, 20)
        self.crops_per_cluster_spinbox.setValue(5)
        sampling_layout.addWidget(self.crops_per_cluster_spinbox)

        layout.addLayout(sampling_layout)

        # Progress label
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        self.setLayout(layout)
        self.setWindowTitle("Clustered Crops Viewer")
        self.resize(800, 600)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

    def on_cluster_button_clicked(self):
        """
        Handles the clustering button click.
        """
        self.request_clustering.emit()
        logging.info("Clustering button clicked.")

    def display_sampled_crops(self, sampled_crops: List[Dict]):
        """
        Displays the sampled crops in the view.

        :param sampled_crops: List of dictionaries containing 'cluster_id', 'image_index', 'coord', and 'crop'.
        """
        # Clear existing content
        for i in reversed(range(self.scroll_layout.count())):
            widget_to_remove = self.scroll_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        # Organize crops by cluster_id
        clusters = {}
        for crop in sampled_crops:
            cluster_id = crop['cluster_id']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(crop)

        # Display each cluster
        for cluster_id, crops in clusters.items():
            group_box = QGroupBox(f"Cluster {cluster_id}")
            group_layout = QHBoxLayout()

            for crop in crops:
                label = QLabel()
                label.setPixmap(crop['crop'])
                label.setToolTip(f"Image Index: {crop['image_index']}\nCoord: {crop['coord']}")
                group_layout.addWidget(label)

            group_box.setLayout(group_layout)
            self.scroll_layout.addWidget(group_box)

        self.scroll_layout.addStretch()
        logging.info(f"Displayed {len(sampled_crops)} sampled crops.")

    def update_progress(self, progress: int):
        """
        Updates the progress bar with the current clustering progress.

        :param progress: Progress percentage.
        """
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"Clustering Progress: {progress}%")
        logging.info(f"Clustering progress updated to {progress}%.")

    def reset_progress(self):
        """
        Resets the progress bar and label.
        """
        self.progress_bar.setValue(0)
        self.progress_label.setText("Clustering Progress: 0%")
        logging.info("Clustering progress reset.")

    def display_clusters(self, clusters: Dict[int, List[Dict]]):
        """
        Placeholder method if needed to display clusters.

        :param clusters: Dictionary of clusters.
        """
        pass  # Implement as needed
