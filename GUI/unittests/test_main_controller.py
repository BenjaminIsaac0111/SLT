import sys
import types

# Provide lightweight stubs for heavy optional dependencies so that
# MainController and its imports can be loaded without installing the
# full stack. Only the minimal attributes used during the test are stubbed.
for name in [
    "h5py",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.colors",
    "cachetools",
    "sklearn",
    "sklearn.utils",
    "sklearn.utils.class_weight",
    "sklearn.cluster",
    "sklearn.mixture",
    "numba",
    "sklearn.model_selection",
    "numba",
    "pandas",
    "yaml",
    "PIL",
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
        if name == "sklearn.utils.class_weight":
            sys.modules[name].compute_class_weight = lambda *a, **k: []
        if name == "sklearn.cluster":
            sys.modules[name].AgglomerativeClustering = object
            sys.modules[name].MiniBatchKMeans = object
        if name == "sklearn.mixture":
            sys.modules[name].GaussianMixture = object
        if name == "numba":
            sys.modules[name].njit = lambda *a, **k: (lambda f: f)

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
        self.data_path = None

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

    def set_data_path(self, path: str):
        self.data_path = path


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


def test_reorder_clusters_preserves_order():
    clusters = {1: [1], 2: [2], 3: [3]}
    ordered = MainController._reorder_clusters([2, 1], clusters)
    assert list(ordered.keys()) == [2, 1, 3]


def test_clusters_from_state_honours_order(example_state: 'ProjectState'):
    state = example_state.copy()
    state.clusters = {"1": state.clusters["1"], "2": state.clusters["1"]}
    state.cluster_order = [2, 1]
    view = DummyView()
    ctrl = build_controller(view)
    clusters = ctrl._clusters_from_state(state)
    assert list(clusters.keys()) == [2, 1]


def test_label_generator_switches(monkeypatch):
    class Timer:
        def __init__(self):
            self.started = False
            self.ms = None

        def start(self, ms):
            self.started = True
            self.ms = ms

    view = DummyView()
    ctrl = MainController.__new__(MainController)
    ctrl.view = view
    ctrl.clustering_controller = types.SimpleNamespace(annotation_generator=None)
    ctrl._idle_timer = Timer()
    ctrl._dirty = False

    ctrl.on_label_generator_method_changed("Image Centre")
    from GUI.models.PointAnnotationGenerator import CenterPointAnnotationGenerator
    assert isinstance(ctrl.annotation_generator, CenterPointAnnotationGenerator)
    assert ctrl._dirty is True
    assert ctrl._idle_timer.started


def test_visible_crops_complete():
    from GUI.models.annotations import PointAnnotation

    view = DummyView()
    anno1 = PointAnnotation(0, "a", (0, 0), [], 0.5, class_id=1)
    anno2 = PointAnnotation(0, "b", (1, 1), [], 0.5, class_id=-1)
    view.selected_crops = [{"annotation": anno1}]
    ctrl = build_controller(view)
    assert ctrl._visible_crops_complete()
    view.selected_crops.append({"annotation": anno2})
    assert not ctrl._visible_crops_complete()


def test_propagate_labeling_changes(monkeypatch):
    from GUI.models.annotations import PointAnnotation

    view = DummyView()
    ctrl = build_controller(view)
    ann = PointAnnotation(0, "a", (0, 0), [], 0.5)
    ctrl.clustering_controller.get_clusters = lambda: {1: [ann]}
    called = {}

    def fake_prop(arg):
        called["annos"] = arg

    monkeypatch.setattr(
        "GUI.controllers.MainController.propagate_for_annotations", fake_prop
    )

    ctrl.propagate_labeling_changes()
    assert called["annos"] == [ann]


def test_start_new_project_sets_model(monkeypatch):
    view = DummyView()
    ctrl = build_controller(view)
    dummy_model = types.SimpleNamespace(data_path="data.h5")
    recorded = {}

    def fake_cidm(state):
        recorded["state"] = state
        return dummy_model

    monkeypatch.setattr(
        "GUI.controllers.MainController.create_image_data_model",
        fake_cidm,
    )
    ctrl.set_model = lambda m: recorded.setdefault("model", m)

    ctrl.start_new_project("data.h5")
    assert recorded["model"] is dummy_model
    assert recorded["state"].data_path == "data.h5"


def test_set_model_updates_view():
    view = DummyView()
    ctrl = types.SimpleNamespace(
        image_data_model=None,
        clustering_controller=types.SimpleNamespace(model=None),
        image_processing_controller=types.SimpleNamespace(model=None),
        io=types.SimpleNamespace(_tag=None),
        view=view,
    )
    MainController.set_model(ctrl, types.SimpleNamespace(data_path="foo.h5"))
    assert view.data_path == "foo.h5"
