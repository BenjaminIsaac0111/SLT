from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
import sys
import logging

from GUI.models.ImageDataModel import ImageDataModel
from GUI.views.PatchImageViewer import PatchImageViewer
from GUI.views.ClusteredCropsView import ClusteredCropsView
from GUI.controllers.MainController import MainController
from GUI.controllers.GlobalClusterController import GlobalClusterController

def main():
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)

    # Initialize the model (replace 'your_hdf5_file.h5' with the actual HDF5 file path)
    hdf5_file_path = (r"C:\Users\benja\OneDrive - University of Leeds\PhD "
                      r"Projects\Attention-UNET\cfg\unet_training_experiments\outputs\dropout_attention_unet_fl_f1.h5"
                      r"\dropout_attention_unet_fl_f1_inference_output.h5")
    model = ImageDataModel(hdf5_file_path)

    # Initialize views
    patch_viewer = PatchImageViewer()
    clustered_crops_view = ClusteredCropsView()

    # Initialize controllers
    main_controller = MainController(model=model, view=patch_viewer)
    global_cluster_controller = GlobalClusterController(model=model, view=clustered_crops_view)

    # Create a main window with tabs to switch between different views
    main_window = QMainWindow()
    tab_widget = QTabWidget()
    tab_widget.addTab(patch_viewer, "Image Viewer")
    tab_widget.addTab(clustered_crops_view, "Clustered Crops")
    main_window.setCentralWidget(tab_widget)
    main_window.setWindowTitle("Patch Image Analysis Tool")
    main_window.resize(1600, 900)
    main_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
