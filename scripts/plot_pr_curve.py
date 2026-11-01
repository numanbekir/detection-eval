"""Plot the precision-recall curve of some classes: this repo next to pycocotools.

For each class: the raw precision-recall curve and its precision envelope from
detection_eval, and the precision pycocotools stores at its 101 recall
thresholds (every 5th shown). If the two implementations agree, the orange
points sit exactly on the blue envelope.

Usage (needs the plot dependency group, i.e. matplotlib):
    python scripts/plot_pr_curve.py --data data/VisDrone.yaml \
        --detections runs/yolo11s_640_val/detections.json --classes car bicycle --out docs/pr_curve.png
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from detection_eval.ap import average_precision_coco, envelope, pr_curve  # noqa: E402
from detection_eval.coco import labels_to_coco, read_data_yaml, read_yolo_dataset  # noqa: E402
from detection_eval.evaluate import evaluate_class, split_by_class  # noqa: E402

OURS = "#2a78d6"
COCO_COLOUR = "#eb6834"
RAW = "#b4b2aa"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"


parser = argparse.ArgumentParser(description="PR curves: this repo vs pycocotools.")
parser.add_argument("--data", required=True, help="Ultralytics data YAML")
parser.add_argument("--split", default="val", help="split key in the YAML")
parser.add_argument("--detections", required=True, help="COCO results JSON")
parser.add_argument("--classes", nargs="+", required=True, help="class names to plot")
parser.add_argument("--title", default="", help="text after the class name, e.g. 'YOLO11s 640, VisDrone val'")
parser.add_argument("--out", required=True, help="output PNG")
args = parser.parse_args()

names, images_dir, labels_dir = read_data_yaml(args.data, args.split)
gt = labels_to_coco(read_yolo_dataset(images_dir, labels_dir), names)
detections = json.loads(Path(args.detections).read_text())

# pycocotools, from dicts in memory (no files written)
with contextlib.redirect_stdout(io.StringIO()):
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(detections)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
# [iou 0.50, 101 recall points, class, area all, maxDets 100]
coco_precision = ev.eval["precision"][0, :, :, 0, 2]
rec_thresholds = ev.params.recThrs

per_class = split_by_class(gt, detections)
cat_by_name = {c["name"]: c["id"] for c in gt["categories"]}

plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, len(args.classes), figsize=(4.6 * len(args.classes), 4.2),
                         sharey=True, facecolor=SURFACE)
axes = np.atleast_1d(axes)

for ax, name in zip(axes, args.classes):
    cat_id = cat_by_name[name]
    scores, is_tp, num_gt = evaluate_class(per_class[cat_id], 0.5, 100)
    precisions, recalls = pr_curve(scores, is_tp, num_gt)
    ours_ap = average_precision_coco(precisions, recalls)

    k = ev.params.catIds.index(cat_id)
    p = coco_precision[:, k]
    coco_ap = float(p[p > -1].mean())

    ax.set_facecolor(SURFACE)
    ax.plot(recalls, precisions, color=RAW, linewidth=1, label="raw curve (this repo)")
    ax.plot(recalls, envelope(precisions), color=OURS, linewidth=2, label="envelope (this repo)")
    step = slice(0, None, 5)
    ax.plot(rec_thresholds[step], p[step], linestyle="none", marker="o", markersize=7,
            markerfacecolor=COCO_COLOUR, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="pycocotools, 101 recall points (every 5th)")

    ax.set_title(f"{name}{', ' + args.title if args.title else ''}\n"
                 f"AP50  pycocotools {coco_ap:.4f}  ·  this repo {ours_ap:.4f}",
                 fontsize=10, loc="left", color=TEXT)
    ax.set_xlabel("recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

axes[0].set_ylabel("precision")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=150, facecolor=SURFACE)
print(f"-> {args.out}")
