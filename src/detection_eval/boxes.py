"""Bounding box formats and Intersection over Union (IoU).

All functions work on NumPy arrays of shape (N, 4). The internal format is
xyxy: (x1, y1, x2, y2) = top-left and bottom-right corners, in pixels.
"""
import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """COCO format (x, y, width, height), (x, y) = top-left corner -> xyxy."""
    boxes = np.asarray(boxes, dtype=np.float64)
    x, y, w, h = boxes.T
    return np.stack([x, y, x + w, y + h], axis=1)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """YOLO/Darknet format (center x, center y, width, height) -> xyxy.

    Darknet label files store these values normalized to [0, 1]; multiply by
    the image width/height before calling this to get pixels.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    cx, cy, w, h = boxes.T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Area of each xyxy box. Boxes with x2 < x1 or y2 < y1 get area 0."""
    boxes = np.asarray(boxes, dtype=np.float64)
    w = np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return w * h


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between every box in `a` (N, 4) and every box in `b` (M, 4).

    Returns an (N, M) matrix where out[i, j] = IoU(a[i], b[j]).
    IoU = intersection area / union area, a value in [0, 1].
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)

    # Intersection rectangle: the larger of the top-left corners and the
    # smaller of the bottom-right corners. a[:, None] has shape (N, 1, 4) and
    # b[None] has shape (1, M, 4), so NumPy broadcasts them to (N, M).
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    # Two zero-area boxes have union 0; define their IoU as 0 instead of NaN.
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
