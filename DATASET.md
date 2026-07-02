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

## Dataset Scale and Split

The released self-built dataset contains 12 commodity categories and follows the
YOLO detection format. The standard split used in the manuscript contains:

- training images: 600;
- validation images: 1020;
- categories: 12;
- one annotated object instance per image in the standard validation split.

The validation split is used for ablation, comparison, weak-class analysis, and
visualization in the manuscript. The weak-class augmentation samples are
constructed only from the training split.

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
- `configs/rpc_single_product_subset.yaml`: configuration template for the
  public RPC single-product subset used for external validation.

## Public RPC Single-Product Subset

For external validation, we used a 12-category single-product subset derived
from the public Retail Product Checkout (RPC) dataset. The selected categories
are:

| ID | RPC class name |
| ---: | --- |
| 0 | 196_stationery |
| 1 | 131_chocolate |
| 2 | 190_tissue |
| 3 | 197_stationery |
| 4 | 173_personal_hygiene |
| 5 | 124_chocolate |
| 6 | 172_personal_hygiene |
| 7 | 177_tissue |
| 8 | 148_candy |
| 9 | 194_stationery |
| 10 | 158_seasoner |
| 11 | 63_dessert |

The subset follows the YOLO directory structure:

```text
rpc_single_product_subset_yolo/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

The public subset was used only as an external generalization check. The
primary ablation and weak-class analysis in the manuscript are based on the
self-built 12-class commodity dataset.

## Usage Notes

The released images are provided for academic research and reproducibility.
Please do not use them for commercial purposes unless separate permission is
obtained from the authors.

If the GitHub repository is cloned without large image files or if a compressed
dataset archive is provided separately, place the extracted dataset under the
same directory layout described above and update the corresponding YAML `path`
field before training or validation.
