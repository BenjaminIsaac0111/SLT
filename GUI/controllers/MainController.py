import logging
from typing import Tuple

from PyQt5.QtCore import QObject, Qt
from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.ImageProcessor import ImageProcessor
from GUI.models.UncertaintyRegionSelector import UncertaintyRegionSelector
from GUI.utils.ImageConversion import pil_image_to_qpixmap
from GUI.views.PatchImageViewer import PatchImageViewer
from PyQt5.QtGui import QPixmap, QColor

class MainController(QObject):
    """
    MainController handles interactions between the Model and the View.
    It listens to signals emitted by the View, processes data using the Model,
    and updates the View accordingly.
    """

    def __init__(self, model: ImageDataModel, view: PatchImageViewer):
        super().__init__()
        self.model = model
        self.view = view
        self.image_processor = ImageProcessor()

        # Initialize the selector with desired parameters
        self.region_selector = UncertaintyRegionSelector()

        # Current filename, index, and selected coordinates
        self.current_filename = None
        self.current_index = -1
        self.selected_coords = []
        self.current_arrow_index = -1  # Initialize to no selection

        # Initialize cluster tracking attributes
        self.current_cluster_index = -1  # Initialize to no cluster selected
        self.total_clusters = 0  # Total number of clusters

        # Connect signals from the view to controller methods
        self.connect_signals()

        # Populate the file list upon initialization
        self.populate_file_list()

    def connect_signals(self):
        """
        Connect signals emitted by the view to the corresponding controller methods.
        """
        self.view.fileSelected.connect(self.on_file_selected)
        self.view.arrowClicked.connect(self.on_arrow_clicked)
        self.view.keyPressed.connect(self.on_key_pressed)

    def populate_file_list(self):
        """
        Retrieves filenames from the model and updates the view's file list.
        """
        try:
            filenames = self.model.get_filenames()
            if not filenames:
                logging.warning("No filenames found in the dataset.")
            self.view.update_file_list(filenames)
            logging.info("File list populated with %d filenames.", len(filenames))
        except Exception as e:
            logging.error(f"Failed to populate file list: {e}")

    def on_file_selected(self, index: int):
        """
        Handles the event when a file is selected from the list.
        """
        # Retrieve image data for the selected file
        data = self.model.get_image_data(index)
        self.current_filename = data['filename']
        self.current_index = index

        # Process the image data using the ImageProcessor
        processed_images = self.image_processor.process_image_data(
            image_array=data['image'],
            logits=data['logits'],
            uncertainty=data['uncertainty']
        )

        # Update the view with the processed images
        self.view.update_images(processed_images, self.current_filename)

        # Select regions based on uncertainty and logits
        clustered_coords = self.region_selector.select_regions(
            uncertainty_map=data['uncertainty'],
            logits=data['logits'],
        )

        # Prepare annotations (clustered by DBSCAN and logit clustering)
        # Initialize a global unique ID counter
        unique_id_counter = 0

        annotations = []
        for cluster_id, cluster_coords in clustered_coords.items():
            for coord in cluster_coords:
                annotation_data = {
                    'id': unique_id_counter,  # Assign unique ID
                    'cid': cluster_id,
                    'position': coord,
                    'name': f"Annotation {unique_id_counter}",
                    'class_label': None,
                    'tags': ['clustered'],
                    'color': QColor(255, 0, 0)  # Default color (red) for the arrow
                }
                annotations.append(annotation_data)
                unique_id_counter += 1  # Increment the unique ID counter

        # Log annotations to verify structure
        logging.debug(f"Annotations being passed: {annotations}")

        # Update the view with annotations (arrows)
        self.view.update_annotations(annotations)

        # Set total clusters and reset current cluster index
        self.total_clusters = len(clustered_coords)  # Update total clusters
        self.current_cluster_index = -1  # Reset the cluster index on new file selection

        logging.info(f"File '{self.current_filename}' selected and processed with {self.total_clusters} clusters.")

    def on_arrow_clicked(self, arrow_id: int):
        """
        Handles the event when an arrow (annotation) is clicked in the view.

        :param arrow_id: The ID of the clicked arrow.
        """
        # Highlight the clicked arrow in all AnnotationViews
        self.view.image_view.highlight_arrow(arrow_id)
        self.view.overlay_view.highlight_arrow(arrow_id)
        self.view.heatmap_view.highlight_arrow(arrow_id)

        # Update the currently selected arrow index
        self.current_arrow_index = arrow_id

        # Retrieve the coordinate of the clicked arrow
        if arrow_id < 0 or arrow_id >= len(self.selected_coords):
            logging.warning(f"Arrow ID {arrow_id} is out of range.")
            return

        coord = self.selected_coords[arrow_id]
        logging.debug(f"Zooming into coordinate: {coord}")
        self.zoom_in_on_region(coord)
        logging.info(f"Arrow {arrow_id} clicked. Zooming in on region {coord}.")

    def zoom_in_on_region(self, coord: Tuple[int, int]):
        """
        Handles zooming into a specific region based on the coordinate.

        :param coord: Tuple of (row, column) representing the region to zoom into.
        """
        if self.current_index == -1:
            logging.warning("No file selected to zoom into.")
            return

        # Get the image data for the current index
        data = self.model.get_image_data(self.current_index)
        image_array = data['image']

        # Get the original image size
        original_image_size = (image_array.shape[0], image_array.shape[1])  # (height, width)

        # Create a zoomed-in image (convert from numpy array to QPixmap for zoomed view)
        zoomed_qpixmap, x_start, y_start = self.model.create_zoomed_crop(
            image_array=image_array,
            coord=coord,
            crop_size=256,  # Adjust the crop size as needed
            zoom_factor=2  # Ensure this matches the ZoomedView's zoom_factor
        )

        # Update the zoomed view and pass the arrow's coordinates and the original image size
        self.view.zoomed_viewer.update_zoomed_image(
            pixmap=zoomed_qpixmap,
            coord=coord,  # Arrow's position in the original image
            original_image_size=original_image_size,
            crop_size=256,  # Crop size for the zoomed view
            zoom_factor=2  # Must match the zoom_factor used in create_zoomed_crop
        )

        logging.info(f"Zoomed in on region {coord}.")

    def on_key_pressed(self, key: int):
        """
        Handles key press events in the view. Cycles through clusters or arrows.
        """
        try:
            if key == Qt.Key_A:
                self.cycle_through_clusters(-1)  # Move to the previous cluster
            elif key == Qt.Key_D:
                self.cycle_through_clusters(1)  # Move to the next cluster
            logging.info(f"Key {key} pressed.")
        except Exception as e:
            logging.error(f"Error handling key press: {e}")

    def cycle_through_arrows(self, direction: int):
        """
        Cycles through the list of arrows based on the direction (-1 for left, +1 for right).
        """
        total_arrows = len(self.selected_coords)
        if total_arrows == 0:
            logging.warning("No arrows available to cycle through.")
            return

        # Update the arrow index, cycling through the list
        self.current_arrow_index = (self.current_arrow_index + direction) % total_arrows

        # Highlight the newly selected arrow
        selected_arrow_id = self.current_arrow_index
        self.view.image_view.highlight_arrow(selected_arrow_id)
        logging.info(f"Selected arrow ID: {selected_arrow_id}")

    def cycle_through_clusters(self, direction: int):
        """
        Cycles through the list of clusters based on the direction (-1 for previous, +1 for next).
        """
        if self.total_clusters == 0:
            logging.warning("No clusters available to cycle through.")
            return

        # Update the cluster index, cycling through the list
        self.current_cluster_index = (self.current_cluster_index + direction) % self.total_clusters

        # Highlight the newly selected cluster
        selected_cluster_id = self.current_cluster_index
        self.view.image_view.highlight_cluster(selected_cluster_id)

        logging.info(f"Selected cluster ID: {selected_cluster_id}")
