import numpy as np

from detection_eval.nms import batched_nms, nms

# A and B are the same 10 x 10 object, B shifted 1 px: IoU = 90 / 110 = 0.82.
# C is somewhere else.
A = [0, 0, 10, 10]
B = [1, 0, 11, 10]
C = [20, 20, 30, 30]


def test_duplicate_is_removed():
    keep = nms([A, B, C], [0.9, 0.8, 0.7], 0.7)
    assert keep.tolist() == [0, 2]


def test_returns_highest_score_first():
    keep = nms([C, A, B], [0.7, 0.9, 0.8], 0.7)
    assert keep.tolist() == [1, 0]


def test_lower_threshold_removes_more():
    # A and B: IoU 0.82. Kept at threshold 0.9, removed at 0.5.
    assert nms([A, B], [0.9, 0.8], 0.9).tolist() == [0, 1]
    assert nms([A, B], [0.9, 0.8], 0.5).tolist() == [0]


def test_equal_iou_is_kept():
    # IoU of these two is exactly 0.5; only "above the threshold" is removed.
    assert nms([[0, 0, 10, 10], [0, 0, 10, 5]], [0.9, 0.8], 0.5).tolist() == [0, 1]


def test_removed_box_does_not_remove_others():
    # B overlaps A and D; D does not overlap A. B is removed by A, so D stays.
    D = [2, 0, 12, 10]  # IoU(A, D) = 80 / 120 = 0.67, IoU(B, D) = 0.82
    keep = nms([A, B, D], [0.9, 0.8, 0.7], 0.7)
    assert keep.tolist() == [0, 2]


def test_ties_keep_input_order():
    assert nms([B, A], [0.8, 0.8], 0.7).tolist() == [0]


def test_empty():
    keep = nms(np.zeros((0, 4)), np.zeros(0), 0.7)
    assert len(keep) == 0


def test_no_overlap_keeps_all():
    boxes = [[i * 20, 0, i * 20 + 10, 10] for i in range(5)]
    scores = [0.1, 0.5, 0.3, 0.9, 0.7]
    assert nms(boxes, scores, 0.7).tolist() == [3, 4, 1, 2, 0]


def test_batched_keeps_overlapping_boxes_of_different_classes():
    # A car and a van box on the same object: both stay.
    keep = batched_nms([A, B], [0.9, 0.8], [3, 4], 0.7)
    assert keep.tolist() == [0, 1]


def test_batched_removes_duplicates_inside_a_class():
    keep = batched_nms([A, B, A], [0.9, 0.8, 0.6], [3, 3, 4], 0.7)
    assert keep.tolist() == [0, 2]


def test_batched_empty():
    assert len(batched_nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))) == 0


def test_max_keep_is_the_same_as_cutting():
    boxes = [[i * 20, 0, i * 20 + 10, 10] for i in range(5)] + [A, B]
    scores = [0.1, 0.5, 0.3, 0.9, 0.7, 0.6, 0.55]
    full = nms(boxes, scores, 0.7)
    assert nms(boxes, scores, 0.7, max_keep=3).tolist() == full[:3].tolist()
