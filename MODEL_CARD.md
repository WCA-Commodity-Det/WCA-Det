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

## Stability

The final weak-class-aware fine-tuning stage was repeated with three random
seeds. The mean result was 0.98619 mAP@0.5 and 0.85501 mAP@0.5:0.95, with
standard deviations of 0.00055 and 0.00035, respectively.

## External Validation

On the public single-product RPC subset, WCA-Det obtained:

| Dataset | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: |
| RPC single-product subset | 0.99500 | 0.92893 |

These results are reported as an external validation of the released code and
configuration, while the main ablation analysis remains based on the self-built
12-class fine-grained commodity dataset.

## Limitations

- The method is designed for fine-grained commodity categories with visually
  similar packaging and weak-class imbalance.
- Performance on large-scale multi-object retail scenes depends on the dataset
  distribution and annotation protocol.
- The released weights are not included; users should train models from the
  provided configuration files.
