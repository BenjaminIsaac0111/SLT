import numpy as np

from GUI.models.Annotation import Annotation
from GUI.models.ImageProcessor import ImageProcessor


def test_create_annotation_overlay_draws_color():
    proc = ImageProcessor()
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    ann = Annotation(
        image_index=0,
        filename="a",
        coord=(10, 10),
        logit_features=np.array([0.0]),
        uncertainty=0.5,
        class_id=1,
    )
    out = proc.create_annotation_overlay(img, [ann])
    arr = np.array(out)
    expected = proc.class_color_map[1]
    assert tuple(arr[10, 10]) == expected
