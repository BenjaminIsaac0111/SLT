# controllers/GlobalClusterController.py

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, QTimer
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from sklearn.cluster import AgglomerativeClustering

from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.ImageProcessor import ImageProcessor
from GUI.models.UncertaintyRegionSelector import UncertaintyRegionSelector
from GUI.views.ClusteredCropsView import ClusteredCropsView

CLASS_COMPONENTS = {
    0: 'Non-Informative',
    1: 'Tumour',
    2: 'Stroma',
    3: 'Necrosis',
    4: 'Vessel',
    5: 'Inflammation',
    6: 'Tumour-Lumen',
    7: 'Mucin',
    8: 'Muscle'
}


class ClusteringWorker(QThread):
    """
    Worker thread for performing global clustering to keep the UI responsive.
    """
    clustering_finished = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)

    def __init__(self, model: ImageDataModel, labels_acquirer: UncertaintyRegionSelector, parent=None):
        super().__init__(parent)
        self.model = model
        self.labels_acquirer = labels_acquirer

    def process_image(self, idx):
        """
        Processes a single image and returns annotations.
        """
        data = self.model.get_image_data(idx)
        uncertainty_map = data.get('uncertainty', None)
        logits = data.get('logits', None)

        if uncertainty_map is None or logits is None:
            logging.warning(f"Missing data for image index {idx}. Skipping.")
            return [], idx

        logit_features, dbscan_coords = self.labels_acquirer.generate_point_labels(
            uncertainty_map=uncertainty_map,
            logits=logits
        )

        annotations = []
        for coord, logit_feature in zip(dbscan_coords, logit_features):
            annotation = {
                'image_index': idx,
                'coord': coord,
                'logit_features': logit_feature
            }
            annotations.append(annotation)

        return annotations, idx

    def run(self):
        """
        Executes the clustering process with parallel image processing.
        """
        all_annotations = []  # List of dicts with image_index, coord, logit_features
        total_images = self.model.get_number_of_images() // 8
        logging.info(f"Starting clustering on {total_images} images.")

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.process_image, idx): idx for idx in range(total_images)}

            for i, future in enumerate(as_completed(futures)):
                try:
                    annotations, idx = future.result()
                    all_annotations.extend(annotations)
                except Exception as e:
                    logging.error(f"Error processing image {futures[future]}: {e}")
                    continue

                # Emit progress update
                progress = int(((i + 1) / total_images) * 100)
                self.progress_updated.emit(progress)
                logging.debug(f"Processed image {idx + 1}/{total_images}. Progress: {progress}%")

        logging.info(f"Total annotations collected: {len(all_annotations)}")

        if not all_annotations:
            logging.warning("No annotations found across all images.")
            self.clustering_finished.emit({})
            return

        # Extract logit features for clustering
        try:
            logit_matrix = np.array([anno['logit_features'] for anno in all_annotations])
            logging.debug(f"Logit matrix shape: {logit_matrix.shape}")
        except Exception as e:
            logging.error(f"Error creating logit matrix: {e}")
            self.clustering_finished.emit({})
            return

        # Perform clustering
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.labels_acquirer.distance_threshold,
                linkage=self.labels_acquirer.linkage
            )
            clustering.fit(logit_matrix)
            labels = clustering.labels_
            logging.debug(f"Clustering labels generated: {labels}")
        except Exception as e:
            logging.error(f"Clustering failed: {e}")
            self.clustering_finished.emit({})
            return

        # Map clusters to annotations
        clusters = {}
        for label, annotation in zip(labels, all_annotations):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(annotation)

        logging.info(f"Global clustering complete. Number of clusters: {len(clusters)}")
        self.clustering_finished.emit(clusters)


class ImageProcessingWorker(QObject):
    processing_finished = pyqtSignal(list)

    def __init__(self, sampled_annotations, model, image_processor):
        super().__init__()
        self.sampled_annotations = sampled_annotations
        self.model = model
        self.image_processor = image_processor

    @pyqtSlot()
    def process_images(self):
        sampled_crops = []

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.process_single_annotation, anno): anno for anno in self.sampled_annotations}

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    sampled_crops.append(result)

        self.processing_finished.emit(sampled_crops)

    def process_single_annotation(self, anno):
        image_data = self.model.get_image_data(anno['image_index'])
        image_array = image_data.get('image', None)
        coord = anno['coord']
        if image_array is None:
            return None

        q_crop, coord_pos = self.image_processor.extract_crop(
            image_array, coord, crop_size=512, zoom_factor=2
        )
        if q_crop is None:
            return None

        return {
            'cluster_id': anno['cluster_id'],
            'image_index': anno['image_index'],
            'coord': coord,
            'crop': q_crop,
            'coord_pos': coord_pos
        }


class GlobalClusterController(QObject):
    """
    GlobalClusterController handles global clustering across all images and manages the presentation of sampled crops.
    """

    clustering_started = pyqtSignal()
    clustering_progress = pyqtSignal(int)
    clusters_ready = pyqtSignal(dict)  # Signal to notify when clusters are ready

    def __init__(self, model: ImageDataModel, view: ClusteredCropsView):
        super().__init__()
        self.model = model
        self.view = view
        self.image_processor = ImageProcessor()
        self.region_selector = UncertaintyRegionSelector()

        # Cluster data structure: {cluster_id: [annotations]}
        self.clusters = {}
        self.cluster_labels = {}  # Initialize cluster labels
        self.cluster_class_labels = {}  # Initialize cluster class labels

        # Sampling parameters
        self.crops_per_cluster = 10  # Number of crops per selected cluster

        # Debounce timer for sampling parameter changes
        self.debounce_timer = QTimer()
        self.debounce_timer.setInterval(300)  # 300 milliseconds debounce interval
        self.debounce_timer.setSingleShot(True)  # Timer will fire only once per activation
        self.debounce_timer.timeout.connect(self.handle_sampling_parameters_changed)

        # Connect view signals
        self.connect_signals()

    def connect_signals(self):
        """
        Connect signals from the view to controller methods.
        """
        self.view.request_clustering.connect(self.start_clustering)
        self.view.sample_cluster.connect(self.on_sample_cluster)
        self.view.sampling_parameters_changed.connect(self.on_sampling_parameters_changed)
        self.view.class_selected.connect(self.on_class_selected)
        self.view.save_annotations_requested.connect(self.on_save_annotations)
        self.clustering_progress.connect(self.view.update_progress)
        self.clusters_ready.connect(self.on_clusters_ready)
    @pyqtSlot()
    def start_clustering(self):
        """
        Initiates the global clustering process.
        """
        logging.info("Clustering process initiated by user.")
        self.clustering_started.emit()
        self.view.reset_progress()

        # Initialize and start the clustering worker thread
        self.worker = ClusteringWorker(model=self.model, labels_acquirer=self.region_selector)
        self.worker.progress_updated.connect(self.clustering_progress.emit)
        self.worker.clustering_finished.connect(self.clusters_ready.emit)
        self.worker.start()

    @pyqtSlot(dict)
    def on_clusters_ready(self, clusters: Dict[int, List[Dict]]):
        """
        Handles the completion of the clustering process.
        """
        self.clusters = clusters
        logging.info(f"Clustering finished with {len(clusters)} clusters.")

        # Prepare cluster_info dict
        cluster_info = {}
        for cluster_id, annotations in clusters.items():
            image_indices = set(anno['image_index'] for anno in annotations)
            num_annotations = len(annotations)
            num_images = len(image_indices)
            cluster_info[cluster_id] = {
                'num_annotations': num_annotations,
                'num_images': num_images
            }
        # Sort clusters by number of images in descending order
        sorted_cluster_info = dict(sorted(cluster_info.items(), key=lambda item: item[1]['num_images'], reverse=True))
        # Get the first cluster ID to display
        first_cluster_id = list(sorted_cluster_info.keys())[0] if sorted_cluster_info else None
        # Populate the cluster selection ComboBox
        self.view.populate_cluster_selection(sorted_cluster_info, selected_cluster_id=first_cluster_id)
        if first_cluster_id is not None:
            self.sample_and_display_crops(cluster_id=first_cluster_id)
        self.view.hide_progress_bar()

    @pyqtSlot(str)
    def on_cluster_file_selected(self, filename):
        """
        Handles loading of a cluster file.
        """
        self.load_cluster_labels(filename)


    @pyqtSlot(int, int)
    def on_sampling_parameters_changed(self, _, crops_per_cluster: int):
        """
        Handles changes in sampling parameters (number of crops per cluster).
        Implements debouncing to prevent excessive sampling.

        :param _: Placeholder for cluster_id which is no longer used.
        :param crops_per_cluster: Number of crops to sample per cluster.
        """
        logging.info(f"Sampling parameters updated: {crops_per_cluster} crops per cluster.")
        self.crops_per_cluster = crops_per_cluster

        # Restart the debounce timer
        if self.debounce_timer.isActive():
            self.debounce_timer.stop()
        self.debounce_timer.start()

    @pyqtSlot()
    def on_save_annotations(self):
        """
        Handles the action of saving annotations to a file.
        """
        # Collect all annotations
        all_annotations = []
        annotations_without_class = False
        for cluster_id, annotations in self.clusters.items():
            for anno in annotations:
                class_id = anno.get('class_id', None)
                if class_id is not None:
                    # Convert data types to native Python types
                    image_index = int(anno['image_index'])
                    cluster_id_int = int(cluster_id)
                    class_id_int = int(class_id)
                    coord = anno['coord']
                    if isinstance(coord, np.ndarray):
                        coord = coord.tolist()
                    else:
                        coord = [float(c) for c in coord]
                    # Build the annotation dictionary
                    all_annotations.append({
                        'image_index': image_index,
                        'coord': coord,
                        'class_id': class_id_int,
                        'cluster_id': cluster_id_int
                    })
                else:
                    annotations_without_class = True

        if annotations_without_class:
            reply = QMessageBox.question(
                self.view,
                "Annotations Without Class Labels",
                "Some annotations do not have class labels assigned. Do you want to proceed with saving?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                logging.info("User canceled saving due to missing class labels.")
                return

        if not all_annotations:
            QMessageBox.warning(self.view, "No Annotations", "There are no annotations to save.")
            return

        # Ask the user for a file path
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self.view, "Save Annotations", "", "JSON Files (*.json);;All Files (*)", options=options
        )
        if filename:
            # Save annotations to the selected file
            try:
                import json
                with open(filename, 'w') as f:
                    json.dump(all_annotations, f, indent=4)
                logging.info(f"Annotations saved to {filename}")
                QMessageBox.information(self.view, "Success", f"Annotations saved to {filename}")
            except Exception as e:
                logging.error(f"Error saving annotations: {e}")
                QMessageBox.critical(self.view, "Error", f"Failed to save annotations: {e}")
        else:
            logging.debug("Save annotations action was canceled.")

    @pyqtSlot(int, int)
    def on_class_selected(self, cluster_id, class_id):
        """
        Handles the event when a class is selected for a cluster.
        """
        logging.info(f"Class {class_id} selected for cluster {cluster_id}.")
        # Assign the class label to the cluster
        if not hasattr(self, 'cluster_class_labels'):
            self.cluster_class_labels = {}
        self.cluster_class_labels[cluster_id] = class_id

        # Update the cluster labels
        class_name = CLASS_COMPONENTS.get(class_id, f"Class {class_id}")
        if not hasattr(self, 'cluster_labels'):
            self.cluster_labels = {}
        self.cluster_labels[cluster_id] = class_name

        # Update the cluster selection to reflect the class label
        self.update_cluster_selection()

        # Assign the class label to the annotations in the cluster
        annotations = self.clusters.get(cluster_id, [])
        for anno in annotations:
            anno['class_id'] = class_id

    @pyqtSlot()
    def handle_sampling_parameters_changed(self):
        """
        Handles the sampling parameters after debouncing.
        """
        logging.info("Handling debounced sampling parameter changes.")
        # Assuming there's a currently selected cluster
        selected_cluster_id = self.view.get_selected_cluster_id()
        if selected_cluster_id is not None:
            self.sample_and_display_crops(selected_cluster_id)
        else:
            logging.warning("No cluster is currently selected.")
            self.view.display_sampled_crops([])

    def sample_and_display_crops(self, cluster_id: int):
        """
        Samples a subset of annotations from a single cluster to display as zoomed-in crops.

        :param cluster_id: The ID of the cluster to sample from.
        """
        if cluster_id not in self.clusters:
            logging.warning(f"Cluster ID {cluster_id} does not exist.")
            self.view.display_sampled_crops([])
            return

        annotations = self.clusters[cluster_id]
        if not annotations:
            logging.warning(f"No annotations found in cluster {cluster_id}.")
            self.view.display_sampled_crops([])
            return

        # Sample annotations within the cluster
        num_samples = min(self.crops_per_cluster, len(annotations))
        try:
            sampled_annotations = np.random.choice(annotations, size=num_samples, replace=False)
            logging.debug(f"Sampled {len(sampled_annotations)} annotations from cluster {cluster_id}.")
        except ValueError as e:
            logging.error(f"Error sampling annotations: {e}")
            self.view.display_sampled_crops([])
            return

        # Prepare annotations for processing
        processed_annotations = []
        for anno in sampled_annotations:
            processed_annotations.append({
                'cluster_id': cluster_id,
                'image_index': anno['image_index'],
                'coord': anno['coord']
            })

        # Initialize Image Processing Worker
        self.image_worker = ImageProcessingWorker(
            sampled_annotations=processed_annotations,
            model=self.model,
            image_processor=self.image_processor
        )
        self.image_thread = QThread()
        self.image_worker.moveToThread(self.image_thread)
        self.image_thread.started.connect(self.image_worker.process_images)
        self.image_worker.processing_finished.connect(self.on_image_processing_finished)
        self.image_worker.processing_finished.connect(self.image_thread.quit)
        self.image_worker.processing_finished.connect(self.image_worker.deleteLater)
        self.image_thread.finished.connect(self.image_thread.deleteLater)
        self.image_thread.start()

    @pyqtSlot(list)
    def on_image_processing_finished(self, sampled_crops):
        """
        Receives processed crops and updates the view.
        """
        logging.info(f"Received {len(sampled_crops)} sampled crops from processing worker.")
        self.view.display_sampled_crops(sampled_crops)
        logging.info(f"Displayed {len(sampled_crops)} sampled crops.")

    @pyqtSlot(int)
    def on_sample_cluster(self, cluster_id: int):
        """
        Handles requests to display crops from a specific cluster.

        :param cluster_id: The ID of the cluster to sample.
        """
        logging.info(f"Sampling crops from cluster {cluster_id}.")
        self.sample_and_display_crops(cluster_id)

    def update_cluster_selection(self):
        """
        Updates the cluster selection ComboBox to include labels.
        """
        cluster_info = {}
        for cluster_id, annotations in self.clusters.items():
            image_indices = set(anno['image_index'] for anno in annotations)
            num_annotations = len(annotations)
            num_images = len(image_indices)
            label = self.cluster_labels.get(cluster_id, '')
            class_id = self.cluster_class_labels.get(cluster_id, None)
            if class_id is not None:
                class_name = CLASS_COMPONENTS.get(class_id, f"Class {class_id}")
                label = f"{label} [{class_name}]"
            if label:
                display_text = (f"Cluster {cluster_id} - '{label}' ({num_annotations} annotations from {num_images} "
                                f"images)")
            else:
                display_text = f"Cluster {cluster_id} ({num_annotations} annotations from {num_images} images)"
            cluster_info[cluster_id] = {
                'num_annotations': num_annotations,
                'num_images': num_images,
                'label': label
            }
        # Sort clusters by number of images in descending order
        sorted_cluster_info = dict(sorted(cluster_info.items(), key=lambda item: item[1]['num_images'], reverse=True))
        # Get current selected cluster ID
        selected_cluster_id = self.view.get_selected_cluster_id()
        self.view.populate_cluster_selection(sorted_cluster_info, selected_cluster_id)
