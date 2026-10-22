"""Train a YOLO11 baseline on VisDrone2019-DET.

Usage (from any directory):
    python scripts/train.py
"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # repo root

# Start from COCO-pretrained weights (transfer learning)
model = YOLO(ROOT / "data/weights/yolo11s.pt")

model.train(
    data=str(ROOT / "data/VisDrone.yaml"),
    imgsz=640,
    epochs=50,
    batch=16,
    workers=4,
    project=str(ROOT / "runs"),
    name="visdrone_yolo11s_640_baseline",
)
