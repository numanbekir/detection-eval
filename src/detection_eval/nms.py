"""Non-maximum suppression (NMS): remove duplicate boxes of the same object.

A detector gives several overlapping boxes for one object. NMS keeps the box
with the highest score and removes every other box that overlaps it by more
than an IoU threshold, then repeats with the next highest box that is left.
"""
import numpy as np

from detection_eval.boxes import box_iou


def nms(boxes, scores, iou_threshold=0.7, max_keep=None):
    """Indices of the boxes to keep, from the highest score down.

    Args:
        boxes: (N, 4) xyxy.
        scores: (N,).
        iou_threshold: a box is removed if its IoU with a kept box is above this.
        max_keep: stop after this many kept boxes (None = no limit). The same
            as cutting the full result, since boxes are kept best first.

    Returns:
        (K,) int array of indices into boxes, highest score first.
        Ties keep their input order (stable sort).
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")  # candidates, best first
    keep = []

    while len(order) > 0 and (max_keep is None or len(keep) < max_keep):
        i = order[0]  # best box that is left
        keep.append(i)

        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]  # one IoU per remaining box
        order = rest[ious <= iou_threshold]  # drop the duplicates of box i

    return np.array(keep, dtype=int)


def batched_nms(boxes, scores, classes, iou_threshold=0.7, max_keep=None):
    """NMS inside each class: boxes of different classes never remove each other.

    Every class is shifted to its own far-away area, so boxes of two classes
    cannot overlap, and then one nms call handles all classes together.

    Returns:
        (K,) int array of indices into boxes, highest score first.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    offset = boxes.max() + 1
    shifted = boxes + (np.asarray(classes)[:, None] * offset)
    return nms(shifted, scores, iou_threshold, max_keep)
