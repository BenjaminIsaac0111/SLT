# workers/ImageProcessingWorker.py

import logging
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

from PyQt5.QtCore import QRunnable, QObject, pyqtSignal

from GUI.models.Annotation import Annotation
from GUI.models.CacheManager import CacheManager
from GUI.models.ImageDataModel import BaseImageDataModel
from GUI.models.ImageProcessor import ImageProcessor


class WorkerSignals(QObject):
    """
    Defines signals available from the running worker.
    """
    processing_finished = pyqtSignal(list)  # List of processed data
    progress_updated = pyqtSignal(int)  # Progress as integer [0..100]


class ImageProcessingWorker(QRunnable):
    """
    Worker runnable for processing and caching image crops.
    To be used with a QThreadPool instead of manually moving to a thread.
    """
    cache = CacheManager()

    def __init__(
            self,
            sampled_annotations: List[Annotation],
            image_data_model: BaseImageDataModel,
            image_processor: ImageProcessor,
            crop_size: int = 512,
            zoom_factor: int = 2,
            already_processed: Optional[List[Dict[str, Any]]] = None
    ):
        """
        :param sampled_annotations: Annotations to process.
        :param image_data_model: For retrieving image data.
        :param image_processor: For crop extraction / image transformations.
        :param crop_size: Size for the extracted crop (default=512).
        :param zoom_factor: Zoom factor for the extracted crop (default=2).
        :param already_processed: An optional list of dicts for items that are already cached.
                                  Typically you append new results to that list.
        """
        super().__init__()
        self.signals = WorkerSignals()

        self.sampled_annotations = sampled_annotations
        self.image_data_model = image_data_model
        self.image_processor = image_processor
        self.crop_size = crop_size
        self.zoom_factor = zoom_factor

        # If some annotations are already processed, store them to be combined later
        self.already_processed = already_processed or []

    def run(self) -> None:
        """
        Runs the image processing task in a worker thread (via QThreadPool).
        """
        total_annotations = len(self.sampled_annotations)
        if total_annotations == 0:
            logging.info("No annotations to process (or all cached).")
            # Emit whatever we already have
            self.signals.processing_finished.emit(self.already_processed)
            return

        processed_data = []
        progress_step = max(total_annotations // 100, 1)
        current_progress = 0

        for idx, annotation in enumerate(self.sampled_annotations, start=1):
            crop_result = self._process_single_annotation(annotation)
            if crop_result is not None:
                processed_data.append(crop_result)

            # Update progress at defined intervals
            if idx % progress_step == 0 or idx == total_annotations:
                new_progress = int((idx / total_annotations) * 100)
                if new_progress != current_progress:
                    self.signals.progress_updated.emit(new_progress)
                    current_progress = new_progress

        # Combine old + new results
        total_results = self.already_processed + processed_data
        self.signals.processing_finished.emit(total_results)

    def _process_single_annotation(self, annotation: Annotation) -> Optional[Dict[str, Any]]:
        """
        Processes a single Annotation, extracting and caching its crop.
        """
        coord = tuple(annotation.coord) if not isinstance(annotation.coord, tuple) else annotation.coord
        cache_key = (annotation.image_index, coord)

        # Check if already cached:
        cached_result = self.cache.get(cache_key)
        if cached_result:
            processed_crop, coord_pos = cached_result
            mask_crop = None
        else:
            image_data = self.image_data_model.get_image_data(annotation.image_index)
            image_array = image_data.get('image')
            if image_array is None:
                logging.warning(f"No image data found for index {annotation.image_index}.")
                return None

            processed_crop, coord_pos = self.image_processor.extract_crop_data(
                image_array, coord, crop_size=self.crop_size, zoom_factor=self.zoom_factor
            )
            mask_crop = None
            if annotation.mask_rle and annotation.mask_shape:
                mask = Annotation.decode_mask(annotation.mask_rle, annotation.mask_shape)
                row, col = map(int, coord)
                original_height, original_width = mask.shape
                half_crop = self.crop_size // 2
                x_start = max(0, col - half_crop)
                y_start = max(0, row - half_crop)
                if x_start + self.crop_size > original_width:
                    x_start = original_width - self.crop_size
                if y_start + self.crop_size > original_height:
                    y_start = original_height - self.crop_size
                width_crop = min(self.crop_size, original_width - x_start)
                height_crop = min(self.crop_size, original_height - y_start)
                mask_crop = mask[y_start:y_start + height_crop, x_start:x_start + width_crop]
                pil_mask = Image.fromarray(mask_crop * 255)
                new_size = (mask_crop.shape[1] * self.zoom_factor, mask_crop.shape[0] * self.zoom_factor)
                mask_crop = np.array(pil_mask.resize(new_size, Image.NEAREST))

            self.cache.set(cache_key, (processed_crop, coord_pos))

        return {
            'annotation': annotation,
            'processed_crop': processed_crop,
            'coord_pos': coord_pos,
            'mask_crop': mask_crop,
        }
