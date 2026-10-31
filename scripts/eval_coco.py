"""Evaluate COCO results JSON against YOLO labels with pycocotools and our own code.

Builds the ground truth JSON from the YOLO label files (detection_eval.coco),
runs COCOeval and prints the standard summary, then AP50 per class from
pycocotools and from detection_eval.evaluate side by side.

Class names and image/label folders come from the Ultralytics data YAML, so the
same script works for any YOLO-format dataset.

Usage:
    python scripts/eval_coco.py --detections runs/coco_eval/yolo11s_640_val/detections.json
    python scripts/eval_coco.py --data data/other.yaml --split test --detections ...
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from detection_eval.coco import labels_to_coco, read_yolo_dataset, save_json  # noqa: E402
from detection_eval.evaluate import coco_map50  # noqa: E402


def read_data_yaml(path, split):
    """Class names and the image/label folders of one split from an Ultralytics data YAML."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())

    names = cfg["names"]
    if isinstance(names, dict):  # {0: "pedestrian", 1: "people", ...}
        names = [names[k] for k in sorted(names)]

    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = path.parent / root
    images = root / cfg[split]
    # Ultralytics rule: the labels folder is the images folder with "images" -> "labels"
    parts = list(images.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return names, images, Path(*parts)


parser = argparse.ArgumentParser(description="pycocotools evaluation of YOLO detections.")
parser.add_argument("--detections", required=True, help="COCO results JSON")
parser.add_argument("--data", default=str(ROOT / "data/VisDrone.yaml"), help="Ultralytics data YAML")
parser.add_argument("--split", default="val", help="split key in the YAML (val, test)")
parser.add_argument("--images", help="override the image folder from the YAML")
parser.add_argument("--labels", help="override the label folder from the YAML")
args = parser.parse_args()

names, images_dir, labels_dir = read_data_yaml(args.data, args.split)
images_dir = args.images or images_dir
labels_dir = args.labels or labels_dir
print(f"{len(names)} classes, images: {images_dir}, labels: {labels_dir}")

gt = labels_to_coco(read_yolo_dataset(images_dir, labels_dir), names)
gt_path = Path(args.detections).with_name("ground_truth.json")
save_json(gt, gt_path)
print(f"ground truth: {len(gt['images'])} images, {len(gt['annotations'])} boxes -> {gt_path}")

with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints loading messages
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(args.detections))
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()

# precision has shape [iou thresholds, recall points, classes, area ranges, maxDets].
# iou index 0 = 0.50, area index 0 = all, maxDets index 2 = 100. -1 means "no GT".
precision = ev.eval["precision"][0, :, :, 0, 2]
coco_aps = {}
for k, cat_id in enumerate(ev.params.catIds):
    p = precision[:, k]
    coco_aps[cat_id] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
coco_map = float(sum(v for v in coco_aps.values() if v == v) / sum(v == v for v in coco_aps.values()))

our_map, our_aps = coco_map50(gt, json.loads(Path(args.detections).read_text()))

print("\nAP50 (area=all, maxDets=100)")
print(f"  {'class':<16} {'pycocotools':>12} {'ours':>12} {'difference':>12}")
for cat_id, ap in coco_aps.items():
    name = coco_gt.cats[cat_id]["name"]
    print(f"  {name:<16} {ap:>12.6f} {our_aps[cat_id]:>12.6f} {our_aps[cat_id] - ap:>12.1e}")
print(f"  {'mAP50':<16} {coco_map:>12.6f} {our_map:>12.6f} {our_map - coco_map:>12.1e}")
