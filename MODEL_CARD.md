# Model Card

## Model

WCA-Det is a YOLO11-based commodity detector with EMA feature enhancement, ECA
channel attention, and a training-only weak-class augmentation strategy.

## Intended Use

The model is intended for research on:

- fine-grained commodity detection;
- visually similar retail product recognition;
- weak-class robustness in object detection;
- lightweight attention modules for object detection.

## Not Intended For

The model and dataset are not intended for commercial deployment without
additional validation and permission from the authors.

## Evaluation

Input size: 640.

| Model | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: |
| YOLO11_EMA | ~0.986 | 0.84334 |
| WCA-Det | 0.98640 | 0.85535 |

The final model improves mAP@0.5:0.95 by 1.20 percentage points over the
YOLO11_EMA baseline.

