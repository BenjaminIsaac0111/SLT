# test_uncertainty_service.py

import numpy as np
import pytest
from PyQt5.QtCore import QObject

from GUI.controllers.MainController import MainController
from GUI.models.Annotation import Annotation
from GUI.models.UncertaintyPropagator import UncertaintyPropagator


# dummy_components.py
# dummy_components.py
# dummy_components.py


class DummySignal:
    """A simple dummy signal to mimic PyQt signals for testing purposes."""

    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in self._callbacks:
            callback(*args, **kwargs)


class DummyView(QObject):
    def __init__(self):
        super().__init__()
        # Define all signals expected by MainController
        self.request_clustering = DummySignal()
        self.sample_cluster = DummySignal()
        self.sampling_parameters_changed = DummySignal()
        self.annotation_method_changed = DummySignal()
        self.bulk_label_changed = DummySignal()
        self.crop_label_changed = DummySignal()
        self.save_project_state_requested = DummySignal()
        self.export_annotations_requested = DummySignal()
        self.load_project_state_requested = DummySignal()
        self.save_project_requested = DummySignal()
        self.save_project_as_requested = DummySignal()

    def get_selected_cluster_id(self):
        # For testing, simply return a dummy cluster id.
        return 1

    def populate_cluster_selection(self, cluster_info, selected_cluster_id=None):
        pass

    def hide_clustering_progress_bar(self):
        pass

    def show_clustering_progress_bar(self):
        pass

    def update_clustering_progress_bar(self, progress: int):
        pass

    def update_annotation_progress_bar(self, progress: int):
        pass

    def hide_crop_loading_progress_bar(self):
        pass

    def show_crop_loading_progress_bar(self):
        pass

    def hide_progress_bar(self):
        pass


class DummyGMM:
    """
    A dummy GaussianMixture-like class for testing purposes.
    It implements predict_proba to return a constant probability distribution
    for each sample.
    """

    def __init__(self, n_components: int, constant_probs: np.ndarray):
        """
        Parameters:
            n_components: The number of components.
            constant_probs: A 1D NumPy array of length n_components representing the probability
                            distribution for each sample. Must sum to 1.
        """
        if len(constant_probs) != n_components:
            raise ValueError("Length of constant_probs must equal n_components.")
        if not np.allclose(np.sum(constant_probs), 1.0):
            raise ValueError("constant_probs must sum to 1.")
        self.n_components = n_components
        self.constant_probs = constant_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns a matrix of shape (n_samples, n_components) where each row is the same probability distribution.
        """
        n_samples = X.shape[0]
        return np.tile(self.constant_probs, (n_samples, 1))


def test_propagation_constant_probability():
    """
    Test that the propagator correctly updates uncertainties when the dummy GMM returns a constant distribution.
    """
    # Create a dummy feature matrix. The actual values are irrelevant in this test.
    feature_matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    uncertainties = np.array([0.8, 0.6, 0.4])

    # Define a dummy GMM with two components. For every sample, predict_proba returns [0.3, 0.7].
    constant_probs = np.array([0.3, 0.7])
    dummy_gmm = DummyGMM(n_components=2, constant_probs=constant_probs)

    alpha = 0.7
    propagator = UncertaintyPropagator(gmm=dummy_gmm, alpha=alpha)

    # Use component index 1 as the labeled component; therefore, p_labeled should be 0.7 for all rows.
    updated_uncertainties = propagator.propagate(feature_matrix, uncertainties, labeled_component=1)

    # Expected update: uncertainty * (1 - alpha * 0.7)
    expected_factor = 1 - alpha * 0.7  # 1 - 0.49 = 0.51
    expected_uncertainties = uncertainties * expected_factor
    np.testing.assert_allclose(updated_uncertainties, expected_uncertainties, rtol=1e-6)


def test_propagation_with_alpha_zero():
    """
    Test that if alpha is set to 0, the original uncertainties are returned unchanged.
    """
    feature_matrix = np.array([[1, 2]], dtype=np.float32)
    uncertainties = np.array([0.5])
    constant_probs = np.array([0.3, 0.7])
    dummy_gmm = DummyGMM(n_components=2, constant_probs=constant_probs)

    propagator = UncertaintyPropagator(gmm=dummy_gmm, alpha=0.0)  # No propagation
    updated_uncertainties = propagator.propagate(feature_matrix, uncertainties, labeled_component=1)

    np.testing.assert_allclose(updated_uncertainties, uncertainties, rtol=1e-6)


def test_propagation_component_index_out_of_range():
    """
    Test that providing a labeled_component index that is out of range
    results in an IndexError.
    """
    feature_matrix = np.array([[1, 2]], dtype=np.float32)
    uncertainties = np.array([0.5])
    constant_probs = np.array([0.5, 0.5])
    dummy_gmm = DummyGMM(n_components=2, constant_probs=constant_probs)

    propagator = UncertaintyPropagator(gmm=dummy_gmm, alpha=0.7)

    # Using a labeled_component index of 3 should be out-of-range since there are only 2 components.
    with pytest.raises(IndexError):
        propagator.propagate(feature_matrix, uncertainties, labeled_component=3)


class DummyView:
    """A minimal view that supports the get_selected_cluster_id() call."""

    def get_selected_cluster_id(self):
        return 1  # For testing, we assume cluster id 1 is selected.


class DummyClusteringController:
    """
    A dummy clustering controller that returns a fixed dictionary of clusters.
    In our test, we will have a single cluster (id 1) with the dummy annotations.
    """

    def __init__(self, annotations):
        self.annotations = annotations

    def get_clusters(self):
        return {1: self.annotations}


class DummyImageProcessingController:
    """A stub that satisfies the interface requirements."""

    def set_clusters(self, clusters):
        pass

    # For label propagation tests, we assume a fixed number of crops per cluster.
    crops_per_cluster = 10


class DummyProjectStateController:
    """A stub controller for project state actions."""

    def autosave_project_state(self, state):
        pass


class DummyModel:
    """A dummy model placeholder."""
    pass


def test_integration_user_labelling_updates_uncertainty():
    """
    This integration test simulates a user labeling a cluster of annotations.
    It then calls propagate_labeling_changes on the main controller to update uncertainties.
    Finally, it verifies that the uncertainties are updated according to the expected rule.
    """
    # Create several dummy Annotation objects.
    # For simplicity, all annotations share the same dummy logit_features.
    annotations = []
    initial_uncertainties = [0.8, 0.6, 0.4]
    for i, uncertainty in enumerate(initial_uncertainties):
        # Use a simple logit feature vector; values are arbitrary.
        logit_features = np.array([float(i), float(i + 1)])
        anno = Annotation(
            image_index=i,
            filename="dummy.jpg",
            coord=(10, 10),
            logit_features=logit_features,
            uncertainty=uncertainty,
            class_id=-1,  # Initially, unlabeled.
            cluster_id=1
        )
        annotations.append(anno)

    # Create a dummy GMM that returns a constant probability distribution.
    # Here, every sample gets predict_proba = [0.2, 0.8].
    constant_probs = np.array([0.2, 0.8])
    dummy_gmm = DummyGMM(n_components=2, constant_probs=constant_probs)

    # Create dummy instances for the view, model, and controllers.
    dummy_view = DummyView()
    dummy_model = DummyModel()
    dummy_clustering_controller = DummyClusteringController(annotations)
    dummy_image_processing_controller = DummyImageProcessingController()
    dummy_project_state_controller = DummyProjectStateController()

    # Instantiate MainController using the dummy view and model.
    main_controller = MainController(model=dummy_model, view=dummy_view)
    # Override the subcontrollers with our dummy controllers.
    main_controller.clustering_controller = dummy_clustering_controller
    main_controller.image_processing_controller = dummy_image_processing_controller
    main_controller.project_state_controller = dummy_project_state_controller

    # Set the dummy GMM in the main controller.
    main_controller.gmm = dummy_gmm

    # Simulate a user labeling the data by assigning a new class (e.g., class_id 1)
    for anno in annotations:
        anno.class_id = 1

    # Capture original uncertainties before propagation.
    original_uncertainties = np.array([anno.uncertainty for anno in annotations])

    # Simulate the propagation of uncertainty changes (as would be triggered by a user label event).
    main_controller.propagate_labeling_changes()

    # Given the dummy GMM returns p_labeled = 0.8 for each annotation,
    # the updated uncertainty should be:
    # new_uncertainty = original * (1 - 0.7 * 0.8) = original * 0.44
    expected_factor = 1 - 0.7 * 0.8  # equals 0.44

    # Verify that each annotation's adjusted uncertainty was updated correctly.
    for anno, orig in zip(annotations, original_uncertainties):
        expected_uncertainty = orig * expected_factor
        np.testing.assert_allclose(
            anno.adjusted_uncertainty, expected_uncertainty, rtol=1e-6,
            err_msg=f"Expected adjusted uncertainty {expected_uncertainty} but got {anno.adjusted_uncertainty}"
        )
