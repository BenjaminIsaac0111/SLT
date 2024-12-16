import logging
from typing import List, Dict, Any, Optional

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.ImageProcessor import ImageProcessor


class ImageProcessingWorker(QObject):
    processing_finished = pyqtSignal(list)
    progress_updated = pyqtSignal(int)

    def __init__(
            self,
            sampled_annotations: List[Annotation],
            image_data_model: ImageDataModel,
            image_processor: ImageProcessor
    ):
        """
        Initializes the ImageProcessingWorker.

        :param sampled_annotations: List of Annotation objects to process.
        :param image_data_model: Instance of ImageDataModel to access image data.
        :param image_processor: Instance of ImageProcessor to handle image operations.
        """
        super().__init__()
        self.sampled_annotations = sampled_annotations
        self.image_data_model = image_data_model
        self.image_processor = image_processor

    @pyqtSlot()
    def process_images(self) -> None:
        """
        Processes all sampled annotations to extract crops and updates progress.
        Emits a signal upon completion.
        """
        sampled_crops = []
        total = len(self.sampled_annotations)
        if total == 0:
            logging.info("No annotations to process.")
            self.processing_finished.emit(sampled_crops)
            return

        progress_step = max(total // 100, 1)  # Update progress every 1%
        current_progress = 0

        for idx, anno in enumerate(self.sampled_annotations, 1):
            result = self.process_single_annotation(anno)
            if result is not None:
                sampled_crops.append(result)
            # Emit progress at defined intervals
            if idx % progress_step == 0 or idx == total:
                progress = int((idx / total) * 100)
                if progress != current_progress:
                    self.progress_updated.emit(progress)
                    current_progress = progress

        self.processing_finished.emit(sampled_crops)

    def process_single_annotation(self, anno: Annotation) -> Optional[Dict[str, Any]]:
        """
        Processes a single annotation to extract crop data, utilizing caching.

        :param anno: The Annotation object to process.
        :return: A dictionary with processed crop data or None if processing fails.
        """
        # Ensure coord is a tuple
        coord = tuple(anno.coord) if not isinstance(anno.coord, tuple) else anno.coord
        cache_key = (anno.image_index, coord)

        # Attempt to retrieve from cache
        cached_result = self.image_processor.cache.get(cache_key)
        if cached_result:
            processed_crop, coord_pos = cached_result
        else:
            # Process the crop
            image_data = self.image_data_model.get_image_data(anno.image_index)
            image_array = image_data.get('image', None)
            if image_array is None:
                logging.warning(f"No image data found for index {anno.image_index}.")
                return None

            processed_crop, coord_pos = self.image_processor.extract_crop_data(
                image_array, coord, crop_size=512, zoom_factor=2
            )

            # Store in cache
            self.image_processor.cache.set(cache_key, (processed_crop, coord_pos))

        return {
            'annotation': anno,
            'processed_crop': processed_crop,
            'coord_pos': coord_pos
        }
