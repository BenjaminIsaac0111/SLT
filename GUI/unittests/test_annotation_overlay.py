import numpy as np
from PIL import Image, ImageDraw, ImageFont

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.Annotation import Annotation
from GUI.models.ImageProcessor import ImageProcessor


def test_create_annotation_overlay_draws_crosshair():
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
    out = proc.create_annotation_overlay(img, [ann], radius=2, show_labels=False)
    arr = np.array(out.convert("RGB"))
    expected = proc.class_color_map[1]
    assert tuple(arr[10, 6]) == expected
    assert tuple(arr[10, 14]) == expected
    assert tuple(arr[6, 10]) == expected
    assert tuple(arr[14, 10]) == expected


def test_create_annotation_overlay_draws_label_box():
    proc = ImageProcessor()
    img = np.ones((80, 80, 3), dtype=np.uint8) * 255
    ann = Annotation(
        image_index=0,
        filename="a",
        coord=(20, 20),
        logit_features=np.array([0.0]),
        uncertainty=0.5,
        class_id=1,
    )
    out = proc.create_annotation_overlay(
        img, [ann], radius=3, crosshair=False, show_labels=True
    )
    arr = np.array(out.convert("RGB"))
    font = ImageFont.load_default()
    label = CLASS_COMPONENTS[1]
    tb = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    lx = 20 + 3 + 4
    ly = 20 - th // 2
    cx = lx + tw // 2
    cy = ly + th // 2
    assert arr[cy, cx, 0] < 255

