import logging

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.UncertaintyRegionSelector import UncertaintyRegionSelector


class ClusteringWorker(QThread):
    """
    Worker thread for performing global clustering to keep the UI responsive.
    """
    clustering_finished = pyqtSignal(list)
    progress_updated = pyqtSignal(int)

    def __init__(self, hdf5_file_path: str, labels_acquirer: UncertaintyRegionSelector, parent=None):
        super().__init__(parent)
        self.hdf5_file_path = hdf5_file_path  # Pass the file path
        self.labels_acquirer = labels_acquirer
        self.model = None  # Will be initialized in the thread

    def process_image(self, idx):
        """
        Processes a single image and assigns a cluster_id to each Annotation.
        """
        data = self.model.get_image_data(idx)
        uncertainty_map = data.get('uncertainty', None)
        logits = data.get('logits', None)
        filename = data.get('filename', None)

        if uncertainty_map is None or logits is None or filename is None:
            logging.warning(f"Missing data for image index {idx}. Skipping.")
            return [], idx

        logit_features, dbscan_coords = self.labels_acquirer.generate_point_labels(
            uncertainty_map=uncertainty_map,
            logits=logits
        )

        annotations = [
            Annotation(
                filename=filename,
                coord=coord,
                logit_features=logit_feature,
                class_id=-1,  # Default to -1 (unlabeled)
                image_index=idx,
                uncertainty=uncertainty_map[coord[0], coord[1]],
                cluster_id=None  # Assign the cluster ID here
            )
            for coord, logit_feature in zip(dbscan_coords, logit_features)
        ]

        return annotations, idx

    def run(self):
        """
        Executes the clustering process using a core-set.
        """
        # Initialize the ImageDataModel in this thread
        self.model = ImageDataModel(self.hdf5_file_path)

        all_annotations = []  # List to collect annotations
        total_images = self.model.get_number_of_images()
        logging.info(f"Starting clustering on {total_images} images.")

        # Process each image sequentially
        for i in range(total_images):
            try:
                annotations, idx = self.process_image(i)
                all_annotations.extend(annotations)
            except Exception as e:
                logging.error(f"Error processing image {i}: {e}")
                continue

            # Emit progress update
            progress = int(((i + 1) / total_images) * 100)
            self.progress_updated.emit(progress)
            logging.debug(f"Processed image {i + 1}/{total_images}. Progress: {progress}%")

        logging.info(f"Total annotations collected: {len(all_annotations)}")

        if not all_annotations:
            logging.warning("No annotations found across all images.")
            self.clustering_finished.emit({})
            return

        # Extract logit features for clustering
        try:
            logit_matrix = np.array(
                [np.concatenate([anno.logit_features, [anno.uncertainty]]) for anno in all_annotations])
            logging.debug(f"Logit matrix shape: {logit_matrix.shape}")
        except Exception as e:
            logging.error(f"Error creating logit matrix: {e}")
            self.clustering_finished.emit({})
            return

        # Step 1: Select a core-set using random sampling or k-means++
        core_set_size = min(10000, len(logit_matrix))  # Adjust size of the core-set
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=core_set_size,
                init='k-means++',
                random_state=42,
                batch_size=3072
            )
            kmeans.fit(logit_matrix)
            core_set_indices = np.unique(kmeans.labels_, return_index=True)[1]
            core_set_features = logit_matrix[core_set_indices]
            core_set_annotations = [all_annotations[i] for i in core_set_indices]
            logging.info(f"Core-set selected with {len(core_set_features)} features.")
        except Exception as e:
            logging.error(f"Error in core-set selection: {e}")
            self.clustering_finished.emit({})
            return

        # Step 2: Perform clustering on the core-set
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1.75,
                linkage='ward'
            )
            clustering.fit(core_set_features)
            core_set_labels = clustering.labels_
            logging.info(f"Core-set clustering complete. Number of clusters: {len(set(core_set_labels))}")
        except Exception as e:
            logging.error(f"Clustering on core-set failed: {e}")
            self.clustering_finished.emit({})
            return

        # Assign cluster IDs to the core-set annotations
        for label, annotation in zip(core_set_labels, core_set_annotations):
            annotation.cluster_id = int(label)  # Assign the cluster ID directly

        # Close the model's HDF5 file
        if self.model is not None:
            self.model.close()

        logging.info("Clustering complete for the core-set.")
        self.clustering_finished.emit(core_set_annotations)
