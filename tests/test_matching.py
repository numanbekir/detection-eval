import numpy as np
import pytest

from detection_eval.matching import match_predictions, precision_recall

GT_A, GT_B = [0, 0, 10, 10], [50, 50, 60, 60]


def test_duplicate_and_background():
    # The 0.75 box is a second box on GT-A, the 0.60 box is on empty background.
    boxes = [[0, 0, 10, 9], [2, 0, 12, 10], [100, 100, 110, 110]]
    matched = match_predictions(boxes, [0.92, 0.75, 0.60], [GT_A, GT_B])
    assert matched.tolist() == [0, -1, -1]
    assert precision_recall(matched, num_gt=2) == pytest.approx((1 / 3, 1 / 2))


def test_result_keeps_input_order():
    # Same predictions in a shuffled order: results follow the input order.
    boxes = [[1, 1, 11, 11], [51, 50, 61, 60], [5, 0, 15, 10]]
    matched = match_predictions(boxes, [0.65, 0.88, 0.70], [GT_A, GT_B])
    assert matched.tolist() == [0, 1, -1]


def test_score_decides_who_matches_first():
    # The lower-IoU box has the higher score, so it takes GT-A first.
    boxes = [[0, 0, 10, 7], [0, 0, 10, 9]]   # IoU 0.7 and 0.9 with GT-A
    matched = match_predictions(boxes, [0.92, 0.75], [GT_A])
    assert matched.tolist() == [0, -1]


def test_iou_exactly_at_threshold_is_a_match():
    matched = match_predictions([[0, 0, 10, 5]], [0.9], [GT_A], iou_threshold=0.5)
    assert matched.tolist() == [0]


def test_no_predictions():
    matched = match_predictions(np.zeros((0, 4)), [], [GT_A, GT_B])
    assert matched.tolist() == []
    assert precision_recall(matched, num_gt=2) == (0.0, 0.0)


def test_no_ground_truth():
    matched = match_predictions([GT_A], [0.9], np.zeros((0, 4)))
    assert matched.tolist() == [-1]
    assert precision_recall(matched, num_gt=0) == (0.0, 0.0)
