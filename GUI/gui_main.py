# main.py

import sys
import logging
from PyQt5.QtWidgets import QApplication
from models.ImageDataModel import ImageDataModel
from views.PatchImageViewer import PatchImageViewer
from controllers.MainController import MainController


def main():
    """
    Entry point for the application. Initializes the QApplication, Model, View, and Controller,
    then starts the application event loop.
    """
    # Initialize logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Start the PyQt application
    app = QApplication(sys.argv)

    # Initialize the model (replace 'your_hdf5_file.h5' with the actual HDF5 file path)
    hdf5_file_path = r"C:\Users\benja\OneDrive - University of Leeds\DATABACKUP\attention_unet_fl_f1.h5_COLLECTED_UNCERTAINTIES_2.h5"
    model = ImageDataModel(hdf5_file_path)

    # Initialize the view
    view = PatchImageViewer()

    # Initialize the controller and connect the model and view
    controller = MainController(model=model, view=view)

    # Show the main window
    view.show()

    # Start the application event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
