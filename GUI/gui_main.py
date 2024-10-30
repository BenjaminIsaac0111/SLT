import logging
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from GUI.controllers.GlobalClusterController import GlobalClusterController
from GUI.models.ImageDataModel import ImageDataModel
from GUI.views.ClusteredCropsView import ClusteredCropsView


def main():
    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)

    # Initialize the model (replace 'your_hdf5_file.h5' with the actual HDF5 file path)
    hdf5_file_path = (r"C:\Users\benja\OneDrive - University of Leeds\PhD "
                      r"Projects\Attention-UNET\cfg\unet_training_experiments\outputs\dropout_attention_unet_fl_f1.h5"
                      r"\dropout_attention_unet_fl_f1_inference_output.h5")
    model = ImageDataModel(hdf5_file_path)

    # Initialize views
    clustered_crops_view = ClusteredCropsView()

    # Initialize controllers
    global_cluster_controller = GlobalClusterController(model=model, view=clustered_crops_view)

    # Create a main window with tabs to switch between different views
    main_window = QMainWindow()
    tab_widget = QTabWidget()
    tab_widget.addTab(clustered_crops_view, "Clustered Crops")
    main_window.setCentralWidget(tab_widget)
    main_window.setWindowTitle("Patch Image Analysis Tool")
    main_window.resize(1600, 900)
    main_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
