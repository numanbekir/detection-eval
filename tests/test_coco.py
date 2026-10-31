import contextlib
import io
import json

import numpy as np
import pytest

from detection_eval.coco import (
    detections_to_coco,
    labels_to_coco,
    read_yolo_dataset,
    yolo_line_to_coco,
)


# ---- yolo_line_to_coco ------------------------------------------------------

def test_line_basic():
    # 100 x 50 image: w = 0.2 * 100 = 20, h = 0.4 * 50 = 20,
    # x = 50 - 10 = 40, y = 25 - 10 = 15.
    category_id, bbox = yolo_line_to_coco("3 0.5 0.5 0.2 0.4", 100, 50)
    assert category_id == 4
    assert bbox == pytest.approx([40, 15, 20, 20])


def test_line_class_zero_becomes_one():
    category_id, _ = yolo_line_to_coco("0 0.5 0.5 0.1 0.1", 100, 100)
    assert category_id == 1


def test_line_width_uses_image_width_height_uses_image_height():
    # Not centred, not square: mixing up width and height breaks this one.
    _, bbox = yolo_line_to_coco("1 0.25 0.75 0.1 0.2", 200, 100)
    # w = 20, h = 20, x = 50 - 10 = 40, y = 75 - 10 = 65
    assert bbox == pytest.approx([40, 65, 20, 20])


def test_line_box_at_top_left_corner():
    _, bbox = yolo_line_to_coco("2 0.05 0.1 0.1 0.2", 640, 480)
    assert bbox == pytest.approx([0, 0, 64, 96])


def test_line_ignores_surrounding_whitespace():
    category_id, bbox = yolo_line_to_coco("  3 0.5 0.5 0.2 0.4 \n", 100, 50)
    assert category_id == 4
    assert bbox == pytest.approx([40, 15, 20, 20])


def test_line_values_are_python_numbers():
    category_id, bbox = yolo_line_to_coco("3 0.5 0.5 0.2 0.4", 100, 50)
    assert type(category_id) is int
    assert all(type(v) is float for v in bbox)


# ---- labels_to_coco ---------------------------------------------------------

NAMES = ["pedestrian", "people", "bicycle", "car"]

IMAGES = [
    {"file_name": "a.jpg", "width": 100, "height": 50,
     "lines": ["3 0.5 0.5 0.2 0.4", "0 0.1 0.2 0.1 0.2"]},
    {"file_name": "b.jpg", "width": 200, "height": 100, "lines": []},
    {"file_name": "c.jpg", "width": 200, "height": 100,
     "lines": ["1 0.25 0.75 0.1 0.2", "", "3 0.5 0.5 0.5 0.5"]},
]


def build():
    return labels_to_coco(IMAGES, NAMES)


def test_top_level_keys():
    assert set(build()) == {"images", "annotations", "categories"}


def test_every_image_is_listed_even_without_labels():
    images = build()["images"]
    assert [im["file_name"] for im in images] == ["a.jpg", "b.jpg", "c.jpg"]
    assert images[1] == {"id": 2, "file_name": "b.jpg", "width": 200, "height": 100}


def test_image_ids_start_at_one():
    assert [im["id"] for im in build()["images"]] == [1, 2, 3]


def test_annotation_count_skips_blank_lines():
    assert len(build()["annotations"]) == 4


def test_annotation_ids_start_at_one_and_keep_counting():
    # pycocotools stores the matched GT id and treats 0 as "no match",
    # so an annotation with id 0 could never be matched.
    assert [a["id"] for a in build()["annotations"]] == [1, 2, 3, 4]


def test_annotation_image_ids():
    assert [a["image_id"] for a in build()["annotations"]] == [1, 1, 3, 3]


def test_annotation_fields():
    ann = build()["annotations"][2]
    assert ann["category_id"] == 2
    assert ann["bbox"] == pytest.approx([40, 65, 20, 20])
    assert ann["area"] == pytest.approx(400)
    assert ann["iscrowd"] == 0


def test_area_is_width_times_height():
    for ann in build()["annotations"]:
        assert ann["area"] == pytest.approx(ann["bbox"][2] * ann["bbox"][3])


def test_categories():
    assert build()["categories"] == [
        {"id": 1, "name": "pedestrian"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "bicycle"},
        {"id": 4, "name": "car"},
    ]


def test_ground_truth_is_json_serialisable():
    json.dumps(build())


# ---- detections_to_coco -----------------------------------------------------

def test_detections_basic():
    boxes = np.array([[10, 20, 50, 40], [0, 0, 5, 5]], dtype=np.float32)
    scores = np.array([0.9, 0.3], dtype=np.float32)
    classes = np.array([3, 0])
    out = detections_to_coco(7, boxes, scores, classes)

    assert len(out) == 2
    assert out[0]["image_id"] == 7
    assert out[0]["category_id"] == 4
    assert out[0]["bbox"] == pytest.approx([10, 20, 40, 20])
    assert out[0]["score"] == pytest.approx(0.9)
    assert out[1]["category_id"] == 1
    assert out[1]["bbox"] == pytest.approx([0, 0, 5, 5])


def test_detections_empty():
    out = detections_to_coco(1, np.zeros((0, 4)), np.zeros(0), np.zeros(0))
    assert out == []


def test_detections_are_json_serialisable():
    # NumPy float32 / int64 values make json.dumps fail.
    boxes = np.array([[10, 20, 50, 40]], dtype=np.float32)
    out = detections_to_coco(1, boxes, np.array([0.5], dtype=np.float32), np.array([2]))
    json.dumps(out)
    assert type(out[0]["category_id"]) is int
    assert type(out[0]["score"]) is float


# ---- read_yolo_dataset ------------------------------------------------------

def test_read_yolo_dataset(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    Image.new("RGB", (64, 32)).save(tmp_path / "images" / "b.jpg")
    Image.new("RGB", (10, 20)).save(tmp_path / "images" / "a.png")
    (tmp_path / "labels" / "b.txt").write_text("1 0.5 0.5 0.1 0.1\n")

    images = read_yolo_dataset(tmp_path / "images", tmp_path / "labels")
    assert images == [
        {"file_name": "a.png", "width": 10, "height": 20, "lines": []},
        {"file_name": "b.jpg", "width": 64, "height": 32, "lines": ["1 0.5 0.5 0.1 0.1"]},
    ]


# ---- end to end with pycocotools --------------------------------------------

def test_perfect_detections_give_ap_one_in_pycocotools():
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = build()
    detections = []
    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        detections += detections_to_coco(
            ann["image_id"], np.array([[x, y, x + w, y + h]]), np.array([0.9]),
            np.array([ann["category_id"] - 1]),
        )

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    assert ev.stats[0] == pytest.approx(1.0)  # mAP50-95
    assert ev.stats[1] == pytest.approx(1.0)  # mAP50
