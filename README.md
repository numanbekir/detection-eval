# detection-eval

Object detection evaluation written from scratch (IoU, matching, AP, COCO-style mAP)
and checked against pycocotools, then used to answer one question on a real model:

**Same model, same images: how much mAP do you get depending on who measures it, and why?**

The model is YOLO11s trained on [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset)
(drone images, 10 classes, many objects under 32 pixels) at two input sizes, 640 and 1280.

## Results

`best.pt` of each run, selected on val. Two splits: val (548 images, 38,759 objects) and
test-dev (1,610 images, 75,102 objects), which was not used for training or model selection.

| split | mAP | measured by | imgsz 640 | imgsz 1280 |
|---|---|---|---|---|
| val | mAP50 | Ultralytics `yolo val` | 0.380 | 0.547 |
| val | mAP50 | pycocotools | 0.347 | 0.503 |
| val | mAP50 | **this repo** | **0.347** | **0.503** |
| test-dev | mAP50 | pycocotools | 0.288 | 0.411 |
| test-dev | mAP50 | **this repo** | **0.288** | **0.411** |
| val | mAP50-95 | Ultralytics `yolo val` | 0.222 | 0.335 |
| val | mAP50-95 | pycocotools | 0.207 | 0.315 |
| test-dev | mAP50-95 | pycocotools | 0.169 | 0.246 |

- **This repo equals pycocotools.** AP50 is the same in every class, at both sizes, on both
  splits; the largest difference is 2.2e-16 (floating point rounding). The test suite also
  compares the two on random data with tied scores and small `maxDets`.
- **Who measures:** on val, Ultralytics reports 0.033 (640) and 0.044 (1280) more mAP50 than
  pycocotools from the same weights. The known differences are listed below; how much
  each one contributes is not measured yet.
- **Which images:** test-dev is 0.06 (640) and 0.09 (1280) mAP50 lower than val. It is not the
  detection limit (raising `maxDets` to 1000 adds 0.002) and not crowding (46.6 objects per image
  vs 70.7 on val): the test-dev scenes are harder for this model, people and pedestrians most of all.
- **1280 beats 640 on both splits:** +0.156 mAP50 on val, +0.123 on test-dev.

### AP50 per Class

pycocotools and this repo give the same values.

| class | val 640 | val 1280 | test-dev 640 | test-dev 1280 |
|---|---|---|---|---|
| pedestrian | 0.405 | 0.620 | 0.259 | 0.418 |
| people | 0.279 | 0.476 | 0.109 | 0.252 |
| bicycle | 0.115 | 0.319 | 0.075 | 0.195 |
| car | 0.775 | 0.862 | 0.695 | 0.797 |
| van | 0.340 | 0.472 | 0.256 | 0.370 |
| truck | 0.315 | 0.470 | 0.366 | 0.506 |
| tricycle | 0.242 | 0.403 | 0.185 | 0.280 |
| awning-tricycle | 0.100 | 0.179 | 0.142 | 0.233 |
| bus | 0.489 | 0.609 | 0.518 | 0.604 |
| motor | 0.411 | 0.621 | 0.278 | 0.452 |

### AP by Object Size

pycocotools, AP at IoU 0.50:0.95, object area in original image pixels.

| size | val 640 | val 1280 | test-dev 640 | test-dev 1280 |
|---|---|---|---|---|
| small (< 32²) | 0.111 | 0.223 | 0.078 | 0.141 |
| medium | 0.320 | 0.436 | 0.262 | 0.361 |
| large (> 96²) | 0.443 | 0.529 | 0.408 | 0.483 |

At 640 a 1360 × 765 image is scaled down to 640 × 360, so a 20-pixel pedestrian becomes about
9 pixels. Small objects gain the most from 1280 (AP almost doubles on both splits). The cost is
speed: 3.4 ms vs 10.3 ms per image on the GPU (Ultralytics val, RTX 3060 Laptop).

### One Image

![True positives, false positives and missed objects at 640 and 1280](docs/matches_640_vs_1280.jpg)

Part of val image `0000295_02400_d_0000033` (317 objects), detections with score ≥ 0.3,
IoU ≥ 0.5. Green: true positive, red: false positive, orange: missed. Counts are for the whole
image. At 1280 the model finds 63 more objects with 10 more false positives.

### Why Ultralytics Reports More

From the Ultralytics 8.4.175 source:

- **Matching order.** pycocotools matches detections from the highest score down. Ultralytics
  matches by IoU, highest IoU first.
- **AP at 101 points.** pycocotools reads the precision envelope as a step function
  (`searchsorted`). Ultralytics interpolates linearly (`np.interp`) and integrates with
  `np.trapezoid`.
- **Inference path.** `yolo val` runs batched with rectangular letterboxing and FP16;
  `scripts/predict_coco.py` runs one image at a time.
- **Not the cause:** detection limits. pycocotools keeps 100 detections per image and class;
  raising it to 1000 changes mAP50 at 640 from 0.347 to 0.348.

## How It Works

```
src/detection_eval/
  boxes.py      box format conversions, pairwise IoU (NumPy broadcasting)
  matching.py   greedy score-ordered matching, precision / recall
  ap.py         precision-recall curve, AP (all points and COCO 101-point), mAP
  coco.py       YOLO labels and detections -> COCO JSON
  evaluate.py   COCO-style mAP50 per class from the pieces above
scripts/
  train.py         train YOLO11 on VisDrone (Ultralytics)
  predict_coco.py  run a model on a split, save detections as COCO results JSON
  eval_coco.py     ground truth from YOLO labels, pycocotools and this repo side by side
  draw_matches.py  draw TP / FP / missed boxes on one image
```

Per class, the same steps as pycocotools: in each image sort detections by score (stable sort,
ties keep input order) and keep the top 100, match them to that image's ground truth, put all
images into one list, build one precision-recall curve and average the precision envelope at
101 recall thresholds.

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # numpy, pillow, pyyaml + pycocotools, pytest
uv run pytest -q         # 74 tests
```

Training and prediction need Ultralytics (`uv sync --group train`). The dataset is not included:
download and convert VisDrone with the Ultralytics
[VisDrone.yaml](https://docs.ultralytics.com/datasets/detect/visdrone/), then put the YAML at
`data/VisDrone.yaml` with `path` pointing to the dataset.

```bash
uv run python scripts/train.py --imgsz 640
uv run python scripts/predict_coco.py --weights runs/visdrone_yolo11s_640_baseline/weights/best.pt \
    --imgsz 640 --out runs/coco_eval/yolo11s_640_val/detections.json
uv run python scripts/eval_coco.py --detections runs/coco_eval/yolo11s_640_val/detections.json
uv run python scripts/draw_matches.py --run runs/coco_eval/yolo11s_640_val
```

For test-dev, add `--images data/VisDrone/images/test` to `predict_coco.py` and `--split test`
to `eval_coco.py`.

Training settings: YOLO11s from COCO-pretrained weights, 50 epochs, seed 0, batch 16 at 640 and
4 at 1280 (`nbs=64` keeps the update size the same), RTX 3060 Laptop 6 GB: about 1 h 50 min and
7 h 20 min.

## Limitations

- **Ignored regions.** VisDrone marks some areas as "ignored". The Ultralytics converter drops
  them, so detections there count as false positives in every row of the tables above.
- **One seed per run.** Differences below about 0.005 mAP may be noise.
- **Ultralytics on test-dev not run.** The "who measures" gap is shown on val only.

## Next

- Split the Ultralytics / pycocotools gap into its parts.
- Ignored regions as crowd boxes.
- NMS (classic, class-aware, Soft-NMS), then a C++ version with a benchmark.
- Error analysis: classification, localisation, duplicate, background and missed errors.

## License

Code: MIT. VisDrone is licensed for academic use only and is not part of this repository.
