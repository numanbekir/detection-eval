"""Convert YOLO labels and model detections to the COCO JSON format.

pycocotools reads two things:
- a ground truth file: one dict with "images", "annotations" and "categories".
- a results list: one dict per detection with "image_id", "category_id",
  "bbox" and "score".

COCO boxes are [x, y, w, h] in pixels, (x, y) = top-left corner.
YOLO labels are "class cx cy w h", normalised to 0-1, (cx, cy) = centre.

Category ids are YOLO class + 1, the same choice Ultralytics makes when it
writes COCO JSON. pycocotools only needs ground truth and detections to use
the same ids.
"""
import json
from pathlib import Path


def yolo_line_to_coco(line, img_w, img_h):
    """One line of a YOLO label file -> (category_id, [x, y, w, h]) in pixels.

    Example: "3 0.5 0.5 0.2 0.4" in a 100 x 50 image -> (4, [40.0, 15.0, 20.0, 20.0]).
    """
    parts = line.split()
    cls = int(parts[0])
    cx, cy, w, h = (float(v) for v in parts[1:5])

    cx, w = cx * img_w, w * img_w
    cy, h = cy * img_h, h * img_h

    x = cx - w / 2
    y = cy - h / 2
    return cls + 1, [x, y, w, h]


def labels_to_coco(images, class_names):
    """Build the COCO ground truth dict.

    Args:
        images: list of dicts with "file_name", "width", "height" and "lines"
            (the lines of that image's YOLO label file, may be empty).
        class_names: list of class names, index = YOLO class.

    Returns:
        {"images": [...], "annotations": [...], "categories": [...]}
        Image ids and annotation ids start at 1. Annotation ids keep counting
        across images. Every image is listed, even one without labels.
    """
    coco_images = []
    annotations = []
    ann_id = 1  # keeps counting across images, never reset
    for image_id, im in enumerate(images, start=1):
        coco_images.append({
            "id": image_id,
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
        })
        for line in im["lines"]:
            if not line.strip():
                continue
            category_id, bbox = yolo_line_to_coco(line, im["width"], im["height"])
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    return {"images": coco_images, "annotations": annotations, "categories": categories}


def detections_to_coco(image_id, boxes_xyxy, scores, classes):
    """Detections of one image -> list of COCO result dicts.

    Args:
        image_id: the id of this image in the ground truth file.
        boxes_xyxy: (N, 4) boxes in pixels, (x1, y1, x2, y2).
        scores: (N,) confidence scores.
        classes: (N,) YOLO class indices.

    Returns:
        A list of N dicts: {"image_id", "category_id", "bbox": [x, y, w, h], "score"}.
        Values are plain Python int/float so json.dump can write them.
    """
    results = []
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        results.append({  # plain int/float: json.dump cannot write NumPy numbers
            "image_id": int(image_id),
            "category_id": int(classes[i]) + 1,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(scores[i]),
        })
    return results


# ---- file helpers -----------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_yolo_dataset(images_dir, labels_dir):
    """Read image sizes and label lines into the list labels_to_coco expects.

    Images are sorted by file name, so image ids are stable between runs.
    An image without a label file gets an empty "lines" list.
    """
    from PIL import Image

    images = []
    paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    for path in paths:
        with Image.open(path) as im:
            width, height = im.size
        label_path = Path(labels_dir) / (path.stem + ".txt")
        lines = label_path.read_text().splitlines() if label_path.exists() else []
        images.append({"file_name": path.name, "width": width, "height": height, "lines": lines})
    return images


def save_json(obj, path):
    """Write obj as JSON, creating the parent folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
