import logging
import os
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QMetaObject, Q_ARG
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsPixmapItem, \
    QGraphicsScene, QVBoxLayout, QPushButton, QWidget, QSplitter, QProgressBar, QFileSystemModel, QTreeView

from GUI_Workers.image_loader_worker import ImageLoaderWorker

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)


class PatchImageViewer(QWidget):
    def __init__(self, *args, patches_dir):
        super(PatchImageViewer, self).__init__(*args)
        # Initialize attributes
        self.tree = None
        self.fs_model = None
        self.patches_dir = patches_dir
        self.worker = None
        self.current_image_index = None
        self.workers = {}
        self.thread = None
        self.thread_active = False
        self.main_splitter = None
        self.file_list_widget = None
        self.progress_bar = None
        self.graphics_view = None
        self.scene = None
        self.raw_patch_mask = None
        self.raw_patch_image = None
        self.image_item = None
        self.mask_item = None
        self.overlay_item = None
        self.btn_generate_image_list = None
        self.current_directory = None
        self.image_paths = []

        # Setup UI components
        self.setup_splitter()
        self.setup_graphics_view()
        self.setup_buttons()
        self.setup_layout()
        self.initialize_attributes()
        self.initialize_thread()
        self.connect_ui_signals()

    def setup_splitter(self):
        self.main_splitter = QSplitter(Qt.Horizontal, self)

        # Set up the file system model
        self.fs_model = QFileSystemModel()
        image_filters = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff']
        self.fs_model.setNameFilters(image_filters)  # Filter to show only image files
        self.fs_model.setNameFilterDisables(False)  # Hide files that do not match the filter

        # If you have a specific start directory, set it here
        self.fs_model.setRootPath(self.patches_dir)

        self.tree = QTreeView()
        self.tree.setModel(self.fs_model)
        # Adjust the root index to the start directory if specified
        self.tree.setRootIndex(self.fs_model.index(self.patches_dir))

        self.tree.clicked.connect(self.on_file_selected)

        self.tree.hideColumn(1)  # Hide the size column
        self.tree.hideColumn(2)  # Hide the type column
        self.tree.hideColumn(3)  # Hide the date modified column

        self.main_splitter.addWidget(self.tree)

    def setup_graphics_view(self):
        self.graphics_view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        self.image_item = QGraphicsPixmapItem()
        self.mask_item = QGraphicsPixmapItem()
        self.overlay_item = QGraphicsPixmapItem()
        [self.scene.addItem(item) for item in [self.image_item, self.mask_item, self.overlay_item]]
        self.main_splitter.addWidget(self.graphics_view)
        self.graphics_view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.graphics_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def setup_buttons(self):
        self.btn_generate_image_list = QPushButton('Generate Image List', self)

    def setup_layout(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self.main_splitter)
        layout.addWidget(self.btn_generate_image_list)
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)
        initial_sizes = [100, self.width() - 100]  # Adjust these values as needed
        self.main_splitter.setSizes(initial_sizes)

    def initialize_attributes(self):
        self.current_directory = None
        self.image_paths = []
        self.progress_bar.setVisible(False)

    def initialize_thread(self):
        self.thread = QThread()
        self.thread.start()

    def connect_ui_signals(self):
        self.main_splitter.splitterMoved.connect(self.scale_image_to_view)

    def resizeEvent(self, event):
        super(PatchImageViewer, self).resizeEvent(event)
        self.scale_image_to_view()

    def scale_image_to_view(self):
        # Reset the view's transformation
        self.graphics_view.resetTransform()

        # Get the bounding rectangle of the scene
        rect = self.scene.itemsBoundingRect()

        # Check if the scene has non-zero dimensions
        if rect.width() == 0 or rect.height() == 0:
            return  # Avoid division by zero

        # Calculate the scaling factors
        scaleFactorX = self.graphics_view.width() / rect.width() if rect.width() > 0 else 1
        scaleFactorY = self.graphics_view.height() / rect.height() if rect.height() > 0 else 1
        scaleFactor = min(scaleFactorX, scaleFactorY)

        # Scale the view
        self.graphics_view.scale(scaleFactor, scaleFactor)

        # Center the scene in the view
        self.graphics_view.centerOn(rect.center())

    def on_file_selected(self, index):
        file_path = self.fs_model.filePath(index)
        logging.info(f"Selected file: {file_path}")

        # Define a list of acceptable image file extensions
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.ico']

        # Check if the selected file is indeed a file and has a valid image file extension
        if os.path.isfile(file_path) and Path(file_path).suffix.lower() in valid_extensions:
            self.load_image(image_path=file_path)
        else:
            logging.info("Selected file is not a supported image format.")

    def load_image(self, image_path):
        self.worker = ImageLoaderWorker()
        self.worker.moveToThread(self.thread)
        self.worker.imageLoaded.connect(self.update_ui_with_images)
        self.worker.finished.connect(self.worker.deleteLater)
        if hasattr(self, 'worker'):
            QMetaObject.invokeMethod(self.worker, "run", Qt.QueuedConnection, Q_ARG(str, image_path))

    def update_ui_with_images(self, image, mask, overlay):
        image_pixmap = self.convert_pil_to_pixmap(image)
        mask_pixmap = self.convert_pil_to_pixmap(mask)
        overlay_pixmap = self.convert_pil_to_pixmap(overlay)
        self.image_item.setPixmap(image_pixmap)
        self.mask_item.setPixmap(mask_pixmap)
        self.overlay_item.setPixmap(overlay_pixmap)
        # Position the mask and overlay next to the original image (left to right)
        self.mask_item.setPos(image_pixmap.width(), 0)  # Position the mask to the right of the image
        self.overlay_item.setPos(image_pixmap.width() + mask_pixmap.width(), 0)  # Position the overlay next to the mask
        self.scale_image_to_view()

    @staticmethod
    def convert_pil_to_pixmap(pil_image):
        if pil_image.mode == "RGB":
            pass
        elif pil_image.mode == "L":
            pil_image = pil_image.convert("RGB")
        data = pil_image.tobytes("raw", "RGB")
        qim = QImage(data, pil_image.size[0], pil_image.size[1], QImage.Format_RGB888)
        return QPixmap.fromImage(qim)


# Main entry for testing
if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = PatchImageViewer(patches_dir=r'Z:\PathologyData\PATCHES\CRO7_PATCHES')
    viewer.show()
    logging.info("Application started")
    sys.exit(app.exec_())
