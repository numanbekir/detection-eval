import numpy as np
import pytest

from detection_eval.boxes import box_iou, cxcywh_to_xyxy, xywh_to_xyxy


def test_format_conversions():
    # The same 4x2 box at top-left (10, 20) in three formats.
    assert np.allclose(xywh_to_xyxy([[10, 20, 4, 2]]), [[10, 20, 14, 22]])
    assert np.allclose(cxcywh_to_xyxy([[12, 21, 4, 2]]), [[10, 20, 14, 22]])


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([0, 0, 2, 2], [0, 0, 2, 2], 1.0),      # identical
        ([0, 0, 2, 2], [1, 0, 3, 2], 2 / 6),    # half overlap: inter 2, union 4+4-2
        ([0, 0, 2, 2], [1, 1, 3, 3], 1 / 7),    # corner overlap: inter 1, union 7
        ([0, 0, 2, 2], [2, 0, 4, 2], 0.0),      # touching edges
        ([0, 0, 2, 2], [5, 5, 6, 6], 0.0),      # far apart
        ([0, 0, 4, 4], [1, 1, 3, 3], 4 / 16),   # one inside the other
    ],
)
def test_iou_hand_computed(a, b, expected):
    assert box_iou([a], [b])[0, 0] == pytest.approx(expected)


def test_iou_matrix_shape_and_symmetry():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, size=(7, 2))
    a = np.hstack([xy, xy + rng.uniform(1, 50, size=(7, 2))])
    b = a[:5] + 3
    iou = box_iou(a, b)
    assert iou.shape == (7, 5)
    assert np.allclose(iou, box_iou(b, a).T)
    assert ((iou >= 0) & (iou <= 1)).all()


def test_zero_area_boxes_do_not_give_nan():
    assert box_iou([[1, 1, 1, 1]], [[1, 1, 1, 1]])[0, 0] == 0.0
