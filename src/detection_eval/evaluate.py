"""COCO-style mAP50 from our own matching and AP code, to check against pycocotools.

Input is the same pair of files pycocotools reads: the ground truth dict
(labels_to_coco) and the list of detection dicts (detections_to_coco).

pycocotools, per class:
1. In every image, sort that class's detections by score (stable) and keep
   the top max_dets (100).
2. Match them to that image's ground truth boxes, highest score first.
3. Put the kept detections of all images in one list, image by image, and
   build one precision-recall curve from it.
4. AP = mean of the precision envelope at 101 recall thresholds.
mAP = mean AP over the classes that have ground truth boxes.
"""
import numpy as np

from detection_eval.ap import average_precision_coco, mean_ap, pr_curve
from detection_eval.boxes import xywh_to_xyxy
from detection_eval.matching import match_predictions


def evaluate_class(images, iou_threshold=0.5, max_dets=100):
    """Score and TP flag of every kept detection of one class, over all images.

    Args:
        images: list with one tuple per image, in image id order:
            (gt_boxes, det_boxes, det_scores)
            gt_boxes: (M, 4) xyxy, det_boxes: (N, 4) xyxy, det_scores: (N,).
            M or N can be 0.
        iou_threshold: minimum IoU for a true positive.
        max_dets: keep at most this many detections per image (highest scores).

    Returns:
        (scores, is_tp, num_gt)
        scores: (K,) float array, is_tp: (K,) bool array. Image by image, and
            inside one image from the highest score down.
        num_gt: total number of ground truth boxes over all images.
    """
    all_scores = []
    all_tp = []
    num_gt = 0

    for gt_boxes, det_boxes, det_scores in images:
        # Objects in images without detections count too: they were missed.
        num_gt += len(gt_boxes)

        # Stable sort keeps tied scores in input order, as pycocotools does.
        # The max_dets cut happens before matching.
        order = np.argsort(-det_scores, kind="stable")[:max_dets]
        kept_boxes = det_boxes[order]
        kept_scores = det_scores[order]

        matched = match_predictions(kept_boxes, kept_scores, gt_boxes, iou_threshold)
        all_scores.extend(kept_scores.tolist())
        all_tp.extend((matched >= 0).tolist())

    return np.array(all_scores, dtype=float), np.array(all_tp, dtype=bool), num_gt


# ---- from COCO dicts to per-class input, and the final mAP -----------------

def split_by_class(gt, detections):
    """{category_id: [(gt_boxes, det_boxes, det_scores) per image, in image id order]}."""
    image_ids = sorted(im["id"] for im in gt["images"])
    out = {}
    for cat in gt["categories"]:
        c = cat["id"]
        gts = {i: [] for i in image_ids}
        dets = {i: [] for i in image_ids}
        for a in gt["annotations"]:
            if a["category_id"] == c:
                gts[a["image_id"]].append(a["bbox"])
        for d in detections:
            if d["category_id"] == c:
                dets[d["image_id"]].append((d["bbox"], d["score"]))
        out[c] = [
            (
                xywh_to_xyxy(gts[i]) if gts[i] else np.zeros((0, 4)),
                xywh_to_xyxy([b for b, _ in dets[i]]) if dets[i] else np.zeros((0, 4)),
                np.array([s for _, s in dets[i]], dtype=float),
            )
            for i in image_ids
        ]
    return out


def coco_map50(gt, detections, max_dets=100):
    """(mAP50, {category_id: AP50}) the way pycocotools computes AP at IoU 0.50."""
    aps = {}
    for c, images in split_by_class(gt, detections).items():
        scores, is_tp, num_gt = evaluate_class(images, 0.5, max_dets)
        if num_gt == 0:
            aps[c] = float("nan")  # pycocotools skips classes without ground truth
            continue
        precisions, recalls = pr_curve(scores, is_tp, num_gt)
        aps[c] = average_precision_coco(precisions, recalls)
    return mean_ap(list(aps.values())), aps
