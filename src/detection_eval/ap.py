"""Average precision (AP) and mean average precision (mAP).

The input for one class is a list of predictions with their scores and a
TP/FP flag (from match_predictions), plus the number of ground truth boxes.
Predictions are sorted by score, high to low. Each row of the sorted list then
stands for one threshold: keeping that row and every row above it. Cumulative
TP and FP counts give precision and recall at every threshold in one pass.

AP summarises the whole precision-recall curve as one number. Two variants:
- all points: the area under the precision envelope (Pascal VOC 2010+).
- COCO: the mean of the envelope sampled at 101 recall thresholds.
"""
import numpy as np


def pr_curve(scores, is_tp, num_gt):
    """Precision and recall at every threshold.

    Args:
        scores: (N,) confidence scores, in any order.
        is_tp: (N,) bool, True for a true positive. For example
            match_predictions(...) >= 0.
        num_gt: number of ground truth boxes, must be > 0.

    Returns:
        (precisions, recalls): two (N,) float arrays in score order, high to
        low. Ties keep their input order (stable sort), as in pycocotools.

    Raises:
        ValueError: if num_gt is 0 (recall is undefined).
    """
    if num_gt == 0:
        raise ValueError("num_gt must be > 0")

    scores = np.asarray(scores, dtype=float)
    is_tp = np.asarray(is_tp, dtype=bool)

    order = np.argsort(-scores, kind="stable")

    is_tp = is_tp[order]

    tp = np.cumsum(is_tp)
    fp = np.cumsum(~is_tp)

    precisions = tp / (tp + fp)
    recalls = tp / num_gt

    return precisions, recalls


def envelope(precisions):
    """Precision envelope: at each row, the highest precision from that row on.

    Precision can go up again after a false positive; the envelope removes
    those dips, so it never increases from left to right.
    """
    z = np.array(precisions, dtype=float)

    for i in range(len(z) - 2, -1, -1):
        if z[i] < z[i + 1]:
            z[i] = z[i + 1]

    return z


def average_precision(precisions, recalls):
    """All-points AP: the area under the precision envelope.

    Each row adds a strip as wide as its recall increase (0 for a false
    positive) and as tall as its envelope value. Returns 0.0 for empty input.
    """
    recalls = np.asarray(recalls, dtype=float)
    z = envelope(precisions)
    widths = np.diff(recalls, prepend=0.0)
    return float(np.sum(widths * z))


def average_precision_coco(precisions, recalls):
    """COCO AP: the mean of the envelope at 101 recall thresholds.

    The thresholds are np.linspace(0.0, 1.0, 101), exactly as in pycocotools.
    For each threshold, take the envelope at the first row whose recall reaches
    it; a threshold that no row reaches contributes 0. Returns 0.0 for empty
    input.
    """
    recalls = np.asarray(recalls, dtype=float)
    z = envelope(precisions)

    threshold = np.linspace(0.0, 1.0, 101)
    total = 0.0
    for t in threshold:
        for i in range(len(recalls)):
            if recalls[i] >= t:
                total = total + z[i]
                break
    return total / 101


def mean_ap(aps):
    """mAP: the plain mean of per-class APs, every class weighted equally.

    Classes with no ground truth boxes should be passed as NaN; they are
    skipped. Returns NaN if no class is left.
    """
    aps = np.asarray(aps, dtype=float)
    mask = ~np.isnan(aps)
    aps = aps[mask]

    if len(aps) == 0:
        return float("nan")

    return np.mean(aps)
