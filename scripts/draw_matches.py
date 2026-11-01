"""Draw true positives, false positives and missed objects on one image.

Reads ground_truth.json (written by eval_coco.py) and detections.json (COCO results) from one folder,
keeps detections above a score threshold, matches them class by class with
detection_eval.matching (IoU >= 0.5) and colours the boxes:
    green  = true positive    red = false positive    orange = missed (false negative)

Usage:
    python scripts/draw_matches.py --run runs/yolo11s_640_val --images data/VisDrone/images/val
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from detection_eval.boxes import xywh_to_xyxy  # noqa: E402
from detection_eval.matching import match_predictions  # noqa: E402

COLOURS = {"tp": (30, 200, 70), "fp": (230, 40, 40), "fn": (255, 150, 0)}

parser = argparse.ArgumentParser(description="Draw TP / FP / FN boxes on one image.")
parser.add_argument("--run", required=True, help="folder with ground_truth.json and detections.json")
parser.add_argument("--image", help="file name; default: the image with the most ground truth boxes")
parser.add_argument("--images", required=True, help="image folder of the evaluated split")
parser.add_argument("--score", type=float, default=0.3, help="keep detections with score >= this")
args = parser.parse_args()

run = Path(args.run)
gt = json.loads((run / "ground_truth.json").read_text())
dets = json.loads((run / "detections.json").read_text())

if args.image:
    image = next(im for im in gt["images"] if im["file_name"] == args.image)
else:
    counts = {}
    for a in gt["annotations"]:
        counts[a["image_id"]] = counts.get(a["image_id"], 0) + 1
    busiest = max(counts, key=counts.get)
    image = next(im for im in gt["images"] if im["id"] == busiest)

gt_anns = [a for a in gt["annotations"] if a["image_id"] == image["id"]]
img_dets = [d for d in dets if d["image_id"] == image["id"] and d["score"] >= args.score]

boxes_to_draw = []  # (xyxy, kind)
totals = {"tp": 0, "fp": 0, "fn": 0}
for cat in gt["categories"]:
    g = [a for a in gt_anns if a["category_id"] == cat["id"]]
    d = [x for x in img_dets if x["category_id"] == cat["id"]]
    g_xyxy = xywh_to_xyxy([a["bbox"] for a in g]) if g else np.zeros((0, 4))
    d_xyxy = xywh_to_xyxy([x["bbox"] for x in d]) if d else np.zeros((0, 4))
    matched = match_predictions(d_xyxy, [x["score"] for x in d], g_xyxy)
    for box, m in zip(d_xyxy, matched):
        kind = "tp" if m >= 0 else "fp"
        boxes_to_draw.append((box, kind))
        totals[kind] += 1
    for j, box in enumerate(g_xyxy):
        if j not in set(matched.tolist()):
            boxes_to_draw.append((box, "fn"))
            totals["fn"] += 1

im = Image.open(Path(args.images) / image["file_name"]).convert("RGB")
draw = ImageDraw.Draw(im)
for box, kind in boxes_to_draw:
    draw.rectangle([float(v) for v in box], outline=COLOURS[kind], width=2)

banner = (f"{image['file_name']}  score >= {args.score}   "
          f"TP {totals['tp']}  FP {totals['fp']}  missed {totals['fn']}  (GT {len(gt_anns)})")
draw.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
draw.text((6, 5), banner, fill=(255, 255, 255))

out = run / f"matches_{Path(image['file_name']).stem}.jpg"
im.save(out, quality=92)
print(banner)
print(f"-> {out}")
