from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.ImageProcessor import ImageProcessor


class ImageProcessingWorker(QObject):
    processing_finished = pyqtSignal(list)
    progress_updated = pyqtSignal(int)  # New signal for progress

    def __init__(self, sampled_annotations, hdf5_file_path: str, image_processor: ImageProcessor):
        super().__init__()
        self.sampled_annotations = sampled_annotations
        self.hdf5_file_path = hdf5_file_path
        self.image_processor = image_processor
        self.model = None

    @pyqtSlot()
    def process_images(self):
        self.model = ImageDataModel(self.hdf5_file_path)
        sampled_crops = []
        total = len(self.sampled_annotations)
        for idx, anno in enumerate(self.sampled_annotations, 1):
            result = self.process_single_annotation(anno)
            if result is not None:
                sampled_crops.append(result)
            # Emit progress after each image is processed
            progress = int((idx / total) * 100)
            self.progress_updated.emit(progress)
        if self.model is not None:
            self.model.close()
        self.processing_finished.emit(sampled_crops)

    def process_single_annotation(self, anno):
        cache_key = (anno.image_index, tuple(anno.coord))
        cached_result = self.image_processor.cache.get(cache_key)
        if cached_result:
            processed_crop, coord_pos = cached_result
        else:
            # Process the crop
            image_data = self.model.get_image_data(anno.image_index)
            image_array = image_data.get('image', None)
            coord = anno.coord
            if image_array is None:
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
