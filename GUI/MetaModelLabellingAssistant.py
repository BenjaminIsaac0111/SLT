import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget, QGridLayout, \
    QFileDialog, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QPixmap
import h5py
import os


class LabelingAssistant(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Pathologist Labeling Assistant')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.select_file_button = QPushButton('Select HDF5 File', self)
        self.select_file_button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.select_file_button)

        self.select_image_dir_button = QPushButton('Select Image Directory', self)
        self.select_image_dir_button.clicked.connect(self.open_dir_dialog)
        self.layout.addWidget(self.select_image_dir_button)

        self.cluster_layout = QGridLayout()
        self.layout.addLayout(self.cluster_layout)

        self.navigation_layout = QHBoxLayout()
        self.prev_button = QPushButton('Previous', self)
        self.prev_button.clicked.connect(self.prev_cluster)
        self.navigation_layout.addWidget(self.prev_button)

        self.next_button = QPushButton('Next', self)
        self.next_button.clicked.connect(self.next_cluster)
        self.navigation_layout.addWidget(self.next_button)

        self.layout.addLayout(self.navigation_layout)

        self.h5_file_path = None
        self.image_dir_path = None
        self.cluster_indices = {}
        self.current_cluster_id = 0

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select HDF5 File", "", "HDF5 Files (*.h5);;All Files (*)",
                                                   options=options)
        if file_name:
            self.h5_file_path = file_name
            self.load_clusters()

    def open_dir_dialog(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory", options=options)
        if directory:
            self.image_dir_path = directory
            self.load_clusters()

    def load_clusters(self):
        if self.h5_file_path and self.image_dir_path:
            try:
                with h5py.File(self.h5_file_path, "r") as hf:
                    cluster_labels = hf['cluster_labels'][:]  # Assume cluster_labels is stored
                    filelist = hf['filelist'][:]

                    self.cluster_indices = {i: [] for i in range(max(cluster_labels) + 1)}
                    for idx, label in enumerate(cluster_labels):
                        self.cluster_indices[label].append(filelist[idx])

                    self.current_cluster_id = 0
                    self.display_cluster(self.current_cluster_id)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load clusters: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please select both HDF5 file and image directory.")

    def display_cluster(self, cluster_id):
        # Clear previous images
        for i in reversed(range(self.cluster_layout.count())):
            widget = self.cluster_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        cluster_label = QLabel(f"Cluster {cluster_id}")
        self.cluster_layout.addWidget(cluster_label, 0, 0)

        files = self.cluster_indices.get(cluster_id, [])
        for i, file in enumerate(files[:5]):  # Display up to 5 samples per cluster
            image_path = os.path.join(self.image_dir_path, file)
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                label = QLabel(self)
                label.setPixmap(pixmap.scaled(100, 100))
                self.cluster_layout.addWidget(label, 1, i)
            else:
                QMessageBox.warning(self, "Warning", f"Image file not found: {image_path}")

    def next_cluster(self):
        if self.current_cluster_id < len(self.cluster_indices) - 1:
            self.current_cluster_id += 1
            self.display_cluster(self.current_cluster_id)
        else:
            QMessageBox.information(self, "Information", "This is the last cluster.")

    def prev_cluster(self):
        if self.current_cluster_id > 0:
            self.current_cluster_id -= 1
            self.display_cluster(self.current_cluster_id)
        else:
            QMessageBox.information(self, "Information", "This is the first cluster.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LabelingAssistant()
    ex.show()
    sys.exit(app.exec_())
