import math

import numpy as np
import pytest

from detection_eval.ap import (
    average_precision,
    average_precision_coco,
    envelope,
    mean_ap,
    pr_curve,
)


def curve(results, num_gt):
    """Build a PR curve from a TP/FP string list that is already in score order."""
    scores = np.linspace(0.95, 0.10, len(results))
    is_tp = np.array([r == "TP" for r in results], dtype=bool)
    return pr_curve(scores, is_tp, num_gt)


# 5 ground truth boxes, 8 predictions.
EXAMPLE = ["TP", "TP", "FP", "TP", "FP", "TP", "FP", "FP"]


# ---- pr_curve ---------------------------------------------------------------

def test_pr_curve_values():
    precisions, recalls = curve(EXAMPLE, num_gt=5)
    assert precisions == pytest.approx([1, 1, 2 / 3, 3 / 4, 3 / 5, 4 / 6, 4 / 7, 4 / 8])
    assert recalls == pytest.approx([0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 0.8])


def test_pr_curve_sorts_by_score():
    # Same predictions as EXAMPLE, given in a shuffled order.
    scores = [0.60, 0.95, 0.40, 0.85, 0.90, 0.50, 0.70, 0.80]
    is_tp = [True, True, False, False, True, False, False, True]
    precisions, recalls = pr_curve(scores, is_tp, num_gt=5)
    assert precisions == pytest.approx([1, 1, 2 / 3, 3 / 4, 3 / 5, 4 / 6, 4 / 7, 4 / 8])
    assert recalls == pytest.approx([0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 0.8])


def test_pr_curve_ties_keep_input_order():
    precisions, recalls = pr_curve([0.9, 0.9], [False, True], num_gt=1)
    assert precisions == pytest.approx([0.0, 0.5])
    assert recalls == pytest.approx([0.0, 1.0])


def test_pr_curve_no_predictions():
    precisions, recalls = pr_curve([], [], num_gt=3)
    assert len(precisions) == 0 and len(recalls) == 0


def test_pr_curve_no_ground_truth_raises():
    with pytest.raises(ValueError):
        pr_curve([0.9], [False], num_gt=0)


# ---- envelope ---------------------------------------------------------------

def test_envelope_removes_dips():
    assert envelope([1.0, 0.5, 2 / 3, 0.75, 0.6]) == pytest.approx([1.0, 0.75, 0.75, 0.75, 0.6])


def test_envelope_never_increases():
    env = envelope(curve(EXAMPLE, num_gt=5)[0])
    assert env == pytest.approx([1, 1, 0.75, 0.75, 2 / 3, 2 / 3, 4 / 7, 0.5])
    assert np.all(np.diff(env) <= 0)


def test_envelope_empty():
    assert len(envelope([])) == 0


# ---- average_precision (all points) and average_precision_coco ------------

@pytest.mark.parametrize(
    "results, num_gt, ap_all, ap_coco",
    [
        (EXAMPLE, 5, 0.68333, 0.68647),
        (["TP", "FP", "TP", "TP", "FP"], 4, 0.625, 0.62871),
        (["TP", "TP", "TP", "FP", "TP", "FP", "FP", "FP"], 5, 0.76, 0.76238),  # FP moved later
        (["TP"] * 6, 10, 0.6, 0.60396),       # no FP, 4 boxes missed
        (["FP", "FP", "FP"], 3, 0.0, 0.0),    # no TP
        (["TP"] * 3, 3, 1.0, 1.0),            # perfect
    ],
)
def test_average_precision(results, num_gt, ap_all, ap_coco):
    precisions, recalls = curve(results, num_gt)
    assert average_precision(precisions, recalls) == pytest.approx(ap_all, abs=1e-5)
    assert average_precision_coco(precisions, recalls) == pytest.approx(ap_coco, abs=1e-5)


def test_coco_thresholds_match_pycocotools():
    # np.linspace(0, 1, 101)[70] is 0.7000000000000001, just above 0.7, so a
    # recall of exactly 0.7 does not reach that threshold: 70 of 101, not 71.
    precisions, recalls = curve(["TP"] * 7, num_gt=10)
    assert average_precision_coco(precisions, recalls) == pytest.approx(70 / 101)


def test_average_precision_empty():
    assert average_precision([], []) == 0.0
    assert average_precision_coco([], []) == 0.0


# ---- mean_ap ----------------------------------------------------------------

def test_mean_ap_plain_mean():
    # VisDrone YOLO11s 640 baseline, AP50 per class (Ultralytics val).
    aps = [0.419, 0.305, 0.124, 0.784, 0.439, 0.360, 0.267, 0.135, 0.534, 0.432]
    assert mean_ap(aps) == pytest.approx(0.3799)


def test_mean_ap_skips_nan():
    assert mean_ap([0.8, float("nan"), 0.4]) == pytest.approx(0.6)


def test_mean_ap_nothing_left_is_nan():
    assert math.isnan(mean_ap([float("nan")]))
    assert math.isnan(mean_ap([]))
