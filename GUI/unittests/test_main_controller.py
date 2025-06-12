import sys
import types
import pytest

# Provide lightweight stubs for heavy optional dependencies so that
# MainController and its imports can be loaded without installing the
# full stack. Only the minimal attributes used during the test are stubbed.
for name in [
    "h5py",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.colors",
    "cachetools",
]:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
        if name == "cachetools":
            class DummyCache(dict):
                def __init__(self, maxsize=128):
                    pass
            sys.modules[name].LRUCache = DummyCache
        if name == "matplotlib.colors":
            sys.modules[name].Colormap = object

from GUI.controllers.MainController import MainController
from GUI.unittests.test_persistence import example_state


class DummyMenu:
    def __init__(self):
        self.checked = None

    def set_checked_annotation_method(self, method: str):
        self.checked = method


class DummyView:
    def __init__(self):
        self.menu = DummyMenu()
        self.calls = []

    def window(self):
        return self

    def menuBar(self):
        return self.menu

    def populate_cluster_selection(self, info, selected_cluster_id=None):
        self.calls.append(("populate", info, selected_cluster_id))

    def hide_progress_bar(self):
        self.calls.append(("hide",))

    def update_labeling_statistics(self, stats):
        self.calls.append(("stats", stats))


class DummyClusteringController:
    def __init__(self):
        self.clusters = None

    def generate_cluster_info(self):
        return {1: {"num_annotations": 1}}

    def compute_labeling_statistics(self):
        return {"dummy": True}


class DummyImageProcessingController:
    def __init__(self):
        self.clusters = None

    def set_clusters(self, clusters):
        self.clusters = clusters


def build_controller(view):
    ctrl = MainController.__new__(MainController)
    ctrl.view = view
    ctrl.clustering_controller = DummyClusteringController()
    ctrl.image_processing_controller = DummyImageProcessingController()
    ctrl._initialize_model_if_needed = lambda state: None
    ctrl.on_label_generator_method_changed = lambda method: setattr(ctrl, "method", method)
    ctrl.on_select_cluster = lambda cid, force=False: view.calls.append(("select", cid, force))
    ctrl._clusters_from_state = MainController._clusters_from_state.__get__(ctrl)
    return ctrl


def test_on_project_loaded_sets_state(example_state: 'ProjectState'):
    view = DummyView()
    ctrl = build_controller(view)

    ctrl._on_project_loaded(example_state)

    clusters = ctrl._clusters_from_state(example_state)
    assert list(ctrl.clustering_controller.clusters.keys()) == list(clusters.keys())
    for cid in clusters:
        ann_exp = [a.to_dict() for a in clusters[cid]]
        ann_got = [a.to_dict() for a in ctrl.clustering_controller.clusters[cid]]
        assert ann_got == ann_exp
    assert ctrl.image_processing_controller.clusters.keys() == clusters.keys()
    assert view.menu.checked == example_state.annotation_method
    assert ("populate", ctrl.clustering_controller.generate_cluster_info(), example_state.selected_cluster_id) in view.calls
    assert ("select", example_state.selected_cluster_id, True) in view.calls
    assert ("hide",) in view.calls
    assert ("stats", ctrl.clustering_controller.compute_labeling_statistics()) in view.calls
    assert getattr(ctrl, "method") == example_state.annotation_method
