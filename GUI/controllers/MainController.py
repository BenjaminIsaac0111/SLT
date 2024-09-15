import logging
from typing import Tuple

from PyQt5.QtCore import QObject, Qt
from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.ImageProcessor import ImageProcessor
from GUI.models.UncertaintyRegionSelector import UncertaintyRegionSelector
from GUI.utils.ImageConversion import pil_image_to_qpixmap
from GUI.views.PatchImageViewer import PatchImageViewer
from PyQt5.QtGui import QPixmap


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
        self.region_selector = UncertaintyRegionSelector(
            filter_size=64,
            aggregation_method='max',
            gaussian_sigma=2.0,
            edge_buffer=8,
            eps=1,  # Adjust based on your data's scale
            min_samples=1  # Minimum samples to form a cluster
        )

        # Current filename, index, and selected coordinates
        self.current_filename = None
        self.current_index = -1  # Initialize to invalid index
        self.selected_coords = []
        self.current_arrow_index = -1  # Initialize to no selection

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

    def on_file_selected(self, index: int):
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
        self.selected_coords = self.region_selector.select_regions(
            uncertainty_map=data['uncertainty'],
            logits=data['logits'],
        )

        logging.debug(f"Selected coordinates: {self.selected_coords}")

        annotations = [{'id': idx, 'position': coord} for idx, coord in enumerate(self.selected_coords)]

        # Log annotations to verify structure
        logging.debug(f"Annotations being passed: {annotations}")

        # Update the view with annotations (arrows representing selected regions)
        self.view.update_annotations(annotations)
        logging.info("File '%s' selected and processed.", self.current_filename)

        # Reset current arrow selection
        self.current_arrow_index = -1
        self.view.image_view.highlight_arrow(-1)
        self.view.overlay_view.highlight_arrow(-1)
        self.view.heatmap_view.highlight_arrow(-1)

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
        Handles key press events in the view. For example, cycling through arrows or removing annotations.

        :param key: The key that was pressed.
        """
        try:
            if key == Qt.Key_A:
                self.cycle_through_arrows(-1)  # Move to the previous arrow
            elif key == Qt.Key_D:
                self.cycle_through_arrows(1)  # Move to the next arrow
            elif key == Qt.Key_Delete:
                self.delete_selected_arrow()
            logging.info(f"Key {key} pressed.")
        except Exception as e:
            logging.error(f"Error handling key press: {e}")

    def cycle_through_arrows(self, direction: int):
        """
        Cycles through arrows (annotations) based on the provided direction.

        :param direction: -1 for previous, 1 for next.
        """
        if not self.selected_coords:
            logging.warning("No arrows to cycle through.")
            return

        # Calculate the next arrow index based on the direction
        next_arrow_index = (self.current_arrow_index + direction) % len(self.selected_coords)

        # Update the view to reflect the selected arrow
        self.view.image_view.highlight_arrow(next_arrow_index)
        self.view.overlay_view.highlight_arrow(next_arrow_index)
        self.view.heatmap_view.highlight_arrow(next_arrow_index)

        # Update the currently selected arrow index
        self.current_arrow_index = next_arrow_index

        # Retrieve the coordinate of the next arrow
        coord = self.selected_coords[next_arrow_index]
        logging.debug(f"Zooming into cycled coordinate: {coord}")
        self.zoom_in_on_region(coord)

        logging.info(f"Cycled to arrow {next_arrow_index}.")

    def delete_selected_arrow(self):
        """
        Removes the currently selected arrow from the view and updates the annotations.
        """
        if self.current_arrow_index == -1:
            logging.warning("No arrow selected to delete.")
            return

        try:
            # Remove the selected arrow from the list of coordinates
            removed_coord = self.selected_coords.pop(self.current_arrow_index)
            logging.info(f"Deleted arrow at index {self.current_arrow_index}, position {removed_coord}.")

            # Update the annotations in the view
            annotations = [{'id': idx, 'position': coord} for idx, coord in enumerate(self.selected_coords)]
            self.view.update_annotations(annotations)

            # Update the selected arrow index
            if self.selected_coords:
                self.current_arrow_index %= len(self.selected_coords)
                selected_arrow_id = self.current_arrow_index
                self.view.image_view.highlight_arrow(selected_arrow_id)
                self.view.overlay_view.highlight_arrow(selected_arrow_id)
                self.view.heatmap_view.highlight_arrow(selected_arrow_id)

                # Retrieve the coordinate of the new selected arrow
                coord = self.selected_coords[self.current_arrow_index]
                logging.debug(f"Zooming into new selected coordinate after deletion: {coord}")
                self.zoom_in_on_region(coord)
                logging.info(f"After deletion, selected arrow {selected_arrow_id}.")
            else:
                # No annotations left
                self.current_arrow_index = -1
                self.view.image_view.highlight_arrow(-1)
                self.view.overlay_view.highlight_arrow(-1)
                self.view.heatmap_view.highlight_arrow(-1)
                self.view.zoomed_viewer.update_zoomed_image(QPixmap(), coord=(0,0), original_image_size=(0,0), crop_size=256, zoom_factor=2)
                logging.info("All arrows deleted. No selection remaining.")

        except Exception as e:
            logging.error(f"Error deleting selected arrow: {e}")
