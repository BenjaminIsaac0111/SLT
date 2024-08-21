import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QFileDialog, QWidget, QPushButton, \
    QMessageBox, QProgressBar, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import cm
from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision
from Model.custom_layers import DropoutAttentionBlock, GroupNormalization, SpatialConcreteDropout

mixed_precision.set_global_policy('mixed_float16')


class ModelLoaderThread(QThread):
    model_loaded = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            model = load_model(
                filepath=self.model_path,
                compile=False,
                custom_objects={
                    'DropoutAttentionBlock': DropoutAttentionBlock,
                    'GroupNormalization': GroupNormalization,
                    'SpatialConcreteDropout': SpatialConcreteDropout,
                }
            )
            self.model_loaded.emit(model)
        except Exception as e:
            self.error_occurred.emit(str(e))


class PathologistAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the title and geometry of the main window
        self.setWindowTitle("Pathologist Annotation Tool")
        self.setGeometry(100, 100, 1000, 800)

        # Create a central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Layout for central widget (Change from QVBoxLayout to QHBoxLayout)
        self.layout = QHBoxLayout(self.central_widget)

        # Image display labels
        self.rgb_image_label = QLabel(self)
        self.rgb_image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.rgb_image_label)

        self.label_image_label = QLabel(self)
        self.label_image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_image_label)

        # Uncertainty map display label
        self.uncertainty_label = QLabel(self)
        self.uncertainty_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.uncertainty_label)

        # Load image button
        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        # Loading bar (Progress Bar)
        self.loading_bar = QProgressBar(self)
        self.loading_bar.setAlignment(Qt.AlignCenter)
        self.loading_bar.setRange(0, 0)  # Indeterminate progress
        self.layout.addWidget(self.loading_bar)

        # Load model button (initiated in another thread)
        self.model = None
        self.load_model()

    def load_model(self):
        # Open file dialog to select a model file
        model_path, _ = QFileDialog.getOpenFileName(self, "Open Model File", "", "Model Files (*.h5 *.pb)")

        if model_path:
            # Show loading bar
            self.loading_bar.show()

            # Start model loading in a separate thread
            self.thread = ModelLoaderThread(model_path)
            self.thread.model_loaded.connect(self.on_model_loaded)
            self.thread.error_occurred.connect(self.on_model_loading_error)
            self.thread.start()
        else:
            QMessageBox.critical(self, "Error", "No model selected. The application will exit.")
            sys.exit()

    def on_model_loaded(self, model):
        self.model = model
        self.model.summary()
        QMessageBox.information(self, "Model Loaded", "Model loaded successfully.")
        self.loading_bar.hide()

    def on_model_loading_error(self, error_message):
        QMessageBox.critical(self, "Error", f"Failed to load model: {error_message}")
        self.loading_bar.hide()
        sys.exit()

    def load_image(self):
        # Open file dialog to select an image file
        image_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")

        if image_path:
            # Load and split the image into RGB input and labels
            img = Image.open(image_path)
            img_array = np.array(img)
            mid_point = img_array.shape[1] // 2

            # Left half for RGB input
            rgb_img_array = img_array[:, :mid_point, :3]
            rgb_img = Image.fromarray(rgb_img_array)

            # Right half for label input (blue channel)
            label_img_array = img_array[:, mid_point:, 2]
            label_img = Image.fromarray(label_img_array)

            # Display the RGB input image
            self.display_image(rgb_img, self.rgb_image_label)

            # Display the label image
            self.display_image(label_img, self.label_image_label, is_greyscale=False)

            # Predict and display the uncertainty map with MC Dropout
            uncertainty_map = self.predict_uncertainty(rgb_img_array, n_iters=10)  # Adjust n_iters as needed

            # Normalize the uncertainty map
            uncertainty_map_normalized = (uncertainty_map - np.min(uncertainty_map)) / (
                        np.max(uncertainty_map) - np.min(uncertainty_map))

            # Apply Spectral colormap
            spectral_uncertainty_map = cm.Spectral_r(uncertainty_map_normalized)

            # Convert to PIL Image for display
            uncertainty_img = Image.fromarray((spectral_uncertainty_map[:, :, :3] * 255).astype(np.uint8))

            # Display the uncertainty map with colormap
            self.display_image(uncertainty_img, self.uncertainty_label, is_greyscale=False)

    def display_image(self, img, label, is_greyscale=False):
        img_qt = self.pil2pixmap(img, is_greyscale)
        label.setPixmap(img_qt)
        label.setFixedSize(img_qt.size())

    def predict_uncertainty(self, img_array, n_iters=5):
        img_array = tf.expand_dims(img_array, axis=0)

        # Ensure dropout is active by running the model with training=True using JIT compiled function
        predictions = []
        for _ in range(n_iters):
            pred = self.model(img_array, training=True)
            predictions.append(pred)

        # Stack predictions along the first axis and calculate variance as uncertainty measure
        predictions = tf.stack(predictions, axis=0)
        uncertainty_map = tf.math.reduce_variance(predictions, axis=0)
        uncertainty_map = tf.reduce_mean(uncertainty_map, axis=-1)
        return tf.squeeze(uncertainty_map)  # Remove any singleton dimensions

    def pil2pixmap(self, im, is_greyscale=False):
        # Convert PIL Image to QPixmap
        if is_greyscale:
            im = im.convert("L")  # Convert to greyscale
            qim = QImage(im.tobytes(), im.size[0], im.size[1], im.size[0], QImage.Format_Grayscale8)
        else:
            im = im.convert("RGB")
            qim = QImage(im.tobytes("raw", "RGB"), im.size[0], im.size[1], QImage.Format_RGB888)
        return QPixmap.fromImage(qim)


# Main function to run the application
def main():
    app = QApplication(sys.argv)
    window = PathologistAnnotationTool()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
