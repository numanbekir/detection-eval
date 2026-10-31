import contextlib
import io

import numpy as np
import pytest

from detection_eval.evaluate import coco_map50, evaluate_class


def box(x1, y1, x2, y2):
    return [x1, y1, x2, y2]


def image(gts, dets, scores):
    """One (gt_boxes, det_boxes, det_scores) tuple from plain lists."""
    return (
        np.array(gts, dtype=float).reshape(-1, 4),
        np.array(dets, dtype=float).reshape(-1, 4),
        np.array(scores, dtype=float),
    )


# ---- evaluate_class -----------------------------------------------------------

def test_one_perfect_detection():
    scores, is_tp, num_gt = evaluate_class([image([box(0, 0, 10, 10)], [box(0, 0, 10, 10)], [0.9])])
    assert scores.tolist() == [0.9]
    assert is_tp.tolist() == [True]
    assert num_gt == 1


def test_returns_numpy_arrays():
    scores, is_tp, _ = evaluate_class([image([box(0, 0, 10, 10)], [box(0, 0, 10, 10)], [0.9])])
    assert isinstance(scores, np.ndarray) and isinstance(is_tp, np.ndarray)
    assert is_tp.dtype == bool


def test_detection_in_image_without_gt_is_false_positive():
    scores, is_tp, num_gt = evaluate_class([image([], [box(0, 0, 10, 10)], [0.7])])
    assert is_tp.tolist() == [False]
    assert num_gt == 0


def test_gt_in_image_without_detections_still_counts():
    # The second image has 2 objects and no detections: both are missed.
    scores, is_tp, num_gt = evaluate_class([
        image([box(0, 0, 10, 10)], [box(0, 0, 10, 10)], [0.9]),
        image([box(0, 0, 10, 10), box(20, 20, 30, 30)], [], []),
    ])
    assert num_gt == 3
    assert len(scores) == 1


def test_empty_image():
    scores, is_tp, num_gt = evaluate_class([image([], [], [])])
    assert len(scores) == 0 and len(is_tp) == 0 and num_gt == 0


def test_sorted_by_score_inside_each_image():
    scores, is_tp, _ = evaluate_class([
        image([box(0, 0, 10, 10)], [box(50, 50, 60, 60), box(0, 0, 10, 10)], [0.3, 0.8]),
    ])
    assert scores.tolist() == [0.8, 0.3]
    assert is_tp.tolist() == [True, False]


def test_images_stay_in_input_order():
    # Not one global sort: image 1 first even though image 2 has a higher score.
    scores, _, _ = evaluate_class([
        image([], [box(0, 0, 10, 10)], [0.2]),
        image([], [box(0, 0, 10, 10)], [0.9]),
    ])
    assert scores.tolist() == [0.2, 0.9]


def test_max_dets_keeps_highest_scores():
    dets = [box(i * 20, 0, i * 20 + 10, 10) for i in range(5)]
    scores, _, _ = evaluate_class([image([], dets, [0.1, 0.5, 0.3, 0.9, 0.7])], max_dets=3)
    assert scores.tolist() == [0.9, 0.7, 0.5]


def test_max_dets_is_per_image():
    dets = [box(i * 20, 0, i * 20 + 10, 10) for i in range(3)]
    scores, _, _ = evaluate_class([image([], dets, [0.1, 0.2, 0.3])] * 2, max_dets=2)
    assert len(scores) == 4


def test_max_dets_cut_happens_before_matching():
    # The low-score detection would be a TP, but it is cut before matching.
    scores, is_tp, num_gt = evaluate_class(
        [image([box(0, 0, 10, 10)], [box(50, 50, 60, 60), box(0, 0, 10, 10)], [0.9, 0.1])],
        max_dets=1,
    )
    assert is_tp.tolist() == [False]
    assert num_gt == 1


def test_ties_keep_input_order():
    scores, is_tp, _ = evaluate_class([
        image([box(0, 0, 10, 10)], [box(0, 0, 10, 10), box(50, 50, 60, 60)], [0.5, 0.5]),
    ])
    assert is_tp.tolist() == [True, False]


def test_iou_threshold():
    # IoU of these two boxes is 0.6.
    img = image([box(0, 0, 10, 10)], [box(0, 0, 10, 6)], [0.9])
    assert evaluate_class([img], iou_threshold=0.5)[1].tolist() == [True]
    assert evaluate_class([img], iou_threshold=0.75)[1].tolist() == [False]


# ---- toy example: 2 images, 4 GT, 5 detections -------------------------------

TOY_GT = {
    "images": [{"id": 1, "file_name": "a.png", "width": 200, "height": 100},
               {"id": 2, "file_name": "b.png", "width": 200, "height": 100}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [20, 20, 40, 40], "area": 1600, "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [120, 30, 60, 60], "area": 3600, "iscrowd": 0},
        {"id": 3, "image_id": 2, "category_id": 1, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0},
        {"id": 4, "image_id": 2, "category_id": 1, "bbox": [100, 40, 60, 40], "area": 2400, "iscrowd": 0},
    ],
    "categories": [{"id": 1, "name": "car"}],
}
TOY_DETS = [
    {"image_id": 1, "category_id": 1, "bbox": [22, 22, 40, 40], "score": 0.95},
    {"image_id": 1, "category_id": 1, "bbox": [25, 25, 40, 40], "score": 0.85},
    {"image_id": 1, "category_id": 1, "bbox": [125, 35, 60, 60], "score": 0.80},
    {"image_id": 2, "category_id": 1, "bbox": [100, 40, 60, 40], "score": 0.70},
    {"image_id": 2, "category_id": 1, "bbox": [60, 60, 30, 30], "score": 0.50},
]


def test_toy_example_matches_hand_calculation():
    m, aps = coco_map50(TOY_GT, TOY_DETS)
    assert aps[1] == pytest.approx(0.6287, abs=1e-4)
    assert m == pytest.approx(0.6287, abs=1e-4)


# ---- against pycocotools on random data --------------------------------------

def random_dataset(seed, n_images=30, n_classes=3):
    rng = np.random.default_rng(seed)
    images, anns, dets = [], [], []
    ann_id = 1
    for img_id in range(1, n_images + 1):
        images.append({"id": img_id, "file_name": f"{img_id}.jpg", "width": 200, "height": 200})
        for _ in range(rng.integers(0, 8)):
            x, y = rng.uniform(0, 150, 2)
            w, h = rng.uniform(5, 50, 2)
            c = int(rng.integers(1, n_classes + 1))
            anns.append({"id": ann_id, "image_id": img_id, "category_id": c,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
            if rng.random() < 0.8:  # a noisy detection of this object
                dets.append({"image_id": img_id, "category_id": c,
                             "bbox": [x + rng.normal(0, 4), y + rng.normal(0, 4), w, h],
                             "score": round(float(rng.random()), 2)})  # rounding makes ties
        for _ in range(rng.integers(0, 5)):  # false alarms
            x, y = rng.uniform(0, 150, 2)
            dets.append({"image_id": img_id, "category_id": int(rng.integers(1, n_classes + 1)),
                         "bbox": [x, y, 20.0, 20.0], "score": round(float(rng.random()), 2)})
    cats = [{"id": c, "name": f"c{c}"} for c in range(1, n_classes + 1)]
    return {"images": images, "annotations": anns, "categories": cats}, dets


def pycocotools_map50(gt, dets, max_dets=100):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        # pycocotools sorts maxDets inside evaluate(), so look the index up afterwards.
        ev.params.maxDets = sorted({1, 10, max_dets})
        ev.evaluate()
        ev.accumulate()
    m = ev.params.maxDets.index(max_dets)
    p = ev.eval["precision"][0, :, :, 0, m]  # IoU 0.50, area all, our max_dets
    return float(np.mean([p[:, k].mean() for k in range(p.shape[1]) if (p[:, k] > -1).any()]))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_same_as_pycocotools_on_random_data(seed):
    pytest.importorskip("pycocotools")
    gt, dets = random_dataset(seed)
    ours, _ = coco_map50(gt, dets)
    assert ours == pytest.approx(pycocotools_map50(gt, dets), abs=1e-12)


def test_same_as_pycocotools_with_small_max_dets():
    pytest.importorskip("pycocotools")
    gt, dets = random_dataset(7, n_images=10)
    ours, _ = coco_map50(gt, dets, max_dets=2)
    assert ours == pytest.approx(pycocotools_map50(gt, dets, max_dets=2), abs=1e-12)
