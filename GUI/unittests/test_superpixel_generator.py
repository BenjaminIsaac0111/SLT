import numpy as np
from GUI.models.PointAnnotationGenerator import SLICSuperpixelAnnotationGenerator
from GUI.models.annotations import MaskAnnotation


def test_superpixel_generator_outputs_masks():
    gen = SLICSuperpixelAnnotationGenerator(n_segments=4, compactness=0.5, edge_buffer=0)
    uncertainty = np.zeros((10, 10), dtype=np.float32)
    uncertainty[2:7, 2:7] = 1.0
    logits = np.random.rand(10, 10, 2).astype(np.float32)
    annos = gen.generate_annotations(uncertainty, logits)
    assert annos
    assert isinstance(annos[0], MaskAnnotation)
    assert annos[0].mask.shape == (10, 10)
