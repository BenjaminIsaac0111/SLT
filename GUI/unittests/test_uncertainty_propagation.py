import numpy as np
from scipy.spatial import cKDTree
import pytest

from GUI.models.annotations import PointAnnotation
from GUI.models.UncertaintyPropagator import auto_lambda, propagate_for_annotations


def make_annotation(x, uncertainty=1.0, class_id=-1):
    return PointAnnotation(
        image_index=0,
        filename="img",
        coord=(0, 0),
        logit_features=np.array([x], dtype=np.float32),
        uncertainty=uncertainty,
        class_id=class_id,
    )


def test_auto_lambda_known_value():
    feats = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    lam = auto_lambda(feats, k=1)
    assert np.isclose(lam, 0.25)


def test_auto_lambda_default_k():
    feats = np.arange(0, 16, 2, dtype=np.float32).reshape(-1, 1)
    lam = auto_lambda(feats)
    k = int(np.ceil(np.log2(feats.shape[0])))
    dists, _ = cKDTree(feats).query(feats, k=k + 1)
    sigma = np.median(dists[:, -1])
    expected = 1.0 / sigma ** 2
    assert np.isclose(lam, expected)


def test_propagate_for_empty_list():
    propagate_for_annotations([])


def test_propagate_no_labels():
    annos = [make_annotation(x, uncertainty=0.5) for x in [0.0, 1.0, 2.0]]
    propagate_for_annotations(annos, lambda_param=1.0)
    for ann in annos:
        assert ann.adjusted_uncertainty == pytest.approx(ann.uncertainty)


def test_propagate_with_single_label():
    annos = [
        make_annotation(0.0, class_id=1),
        make_annotation(1.0),
        make_annotation(3.0),
    ]
    propagate_for_annotations(annos, lambda_param=0.5)
    assert annos[0].adjusted_uncertainty == 0.0
    expected1 = 1.0 * (1.0 - np.exp(-0.5 * 1.0 ** 2))
    expected2 = 1.0 * (1.0 - np.exp(-0.5 * 3.0 ** 2))
    assert annos[1].adjusted_uncertainty == pytest.approx(expected1)
    assert annos[2].adjusted_uncertainty == pytest.approx(expected2)


def test_unlabeling_resets_uncertainty():
    annos = [
        make_annotation(0.0, uncertainty=0.6, class_id=1),
        make_annotation(2.0, uncertainty=0.6),
    ]
    propagate_for_annotations(annos, lambda_param=1.0)
    annos[0].class_id = -1
    propagate_for_annotations(annos, lambda_param=1.0)
    for ann in annos:
        assert ann.adjusted_uncertainty == pytest.approx(0.6)


def test_threshold_ignores_distant_labels():
    annos = [
        make_annotation(0.0, class_id=1),
        make_annotation(10.0),
        make_annotation(2.0),
    ]
    propagate_for_annotations(annos, lambda_param=1.0, threshold=5.0)
    expected_near = 1.0 * (1.0 - np.exp(-1.0 * 2.0 ** 2))
    assert annos[1].adjusted_uncertainty == pytest.approx(1.0)
    assert annos[2].adjusted_uncertainty == pytest.approx(expected_near)


def test_lambda_controls_decay():
    annos_low = [make_annotation(0.0, uncertainty=0.8, class_id=1), make_annotation(1.0, uncertainty=0.8)]
    propagate_for_annotations(annos_low, lambda_param=0.1)
    u_low = annos_low[1].adjusted_uncertainty

    annos_high = [make_annotation(0.0, uncertainty=0.8, class_id=1), make_annotation(1.0, uncertainty=0.8)]
    propagate_for_annotations(annos_high, lambda_param=5.0)
    u_high = annos_high[1].adjusted_uncertainty

    assert u_high > u_low
