# Dataset Description

This repository releases a 12-class fine-grained commodity detection dataset in
YOLO format. The dataset is intended for academic research on commodity
recognition, intelligent retail, and fine-grained object detection.

## Classes

| ID | Class name |
| ---: | --- |
| 0 | bsk1 |
| 1 | bsk1_con |
| 2 | bsk1_wt |
| 3 | bss_large |
| 4 | fd_can |
| 5 | hs_can |
| 6 | kkkl_can |
| 7 | mnd_can |
| 8 | nfsq |
| 9 | xb_can |
| 10 | xb_wt |
| 11 | xb |

The weak classes used for the training-only augmentation strategy are:

- bsk1_wt
- bss_large
- hs_can
- nfsq

## Directory Layout

```text
data/
  images/
    train/
    val/
  labels/
    train/
    val/
  train_weak_aug.txt
  train_weighted_simam.txt
```

Each label file follows the YOLO format:

```text
class_id x_center y_center width height
```

The coordinates are normalized to the image width and height.

## Dataset YAML Files

- `configs/dataset_merged.yaml`: standard training and validation split.
- `configs/dataset_merged_weak_aug.yaml`: training list with weak-class
  augmentation samples and the same validation split.

## Usage Notes

The released images are provided for academic research and reproducibility.
Please do not use them for commercial purposes unless separate permission is
obtained from the authors.

