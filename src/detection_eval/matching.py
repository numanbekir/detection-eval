"""Matching predictions to ground truth boxes, and precision / recall.

Matching is greedy and score-ordered, as in Pascal VOC and COCO evaluation:
predictions are visited from the highest score down, and each one takes the
unmatched ground truth box it overlaps most, if that IoU reaches the
threshold. A ground truth box can be matched at most once.
"""
import numpy as np

from detection_eval.boxes import box_iou


def match_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """Match each prediction to at most one ground truth box.

    Args:
        pred_boxes: (N, 4) predicted boxes, xyxy.
        pred_scores: (N,) confidence scores.
        gt_boxes: (M, 4) ground truth boxes, xyxy.
        iou_threshold: minimum IoU for a match.

    Returns:
        (N,) int array in the input order: the index of the matched ground
        truth box for a true positive, or -1 for a false positive.
        A prediction takes the free ground truth box with the highest IoU;
        if several have the same IoU, the last of them.
    """
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    n = len(pred_scores)

    # IoU of every prediction with every ground truth box, shape (N, M).
    ious = box_iou(pred_boxes, gt_boxes)

    # Visit predictions from the highest score down. A stable sort keeps ties
    # in input order, which is what pycocotools does.
    order = np.argsort(-pred_scores, kind="stable")

    # One "taken" flag per ground truth box; -1 means unmatched (false positive).
    gt_taken = np.zeros(ious.shape[1], dtype=bool)
    matched_gt = np.full(n, -1)

    if ious.shape[1] == 0:
        return matched_gt

    for i in order:
        # Taken ground truth boxes get IoU -1 so argmax can never pick them.
        row = np.where(gt_taken, -1.0, ious[i])

        # On equal IoU the last ground truth box wins, as in pycocotools
        # (its loop only skips a box whose IoU is lower, so a tie replaces).
        j = len(row) - 1 - row[::-1].argmax()
        if row[j] >= iou_threshold:
            gt_taken[j] = True
            matched_gt[i] = j

    return matched_gt


def precision_recall(matched_gt, num_gt):
    """Precision and recall from the output of match_predictions.

    Returns 0.0 for precision when there are no predictions, and 0.0 for
    recall when there are no ground truth boxes.
    """
    matched_gt = np.asarray(matched_gt)
    tp = int((matched_gt >= 0).sum())
    fp = len(matched_gt) - tp
    fn = num_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall
