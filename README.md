# WCA-Det: Weak-Class-Aware Dual-Attention Commodity Detection

This repository provides the public code, configuration files, and commodity
detection data used in the Electronics manuscript:

**WCA-Det: Weak-Class-Aware Dual-Attention Enhancement for Fine-Grained
Commodity Detection**.

WCA-Det is built on a compact YOLO11-based detector. It integrates EMA
multi-scale feature enhancement, ECA channel recalibration, and a training-only
weak-class augmentation strategy. The weak-class augmentation is used only for
training data construction and does not add inference parameters or GFLOPs.

## Public Release Scope

This repository is intended to support manuscript review and research
reproducibility. It includes:

- model YAML files for YOLO11, YOLO11_EMA, YOLO11_ECA, and WCA-Det;
- training, validation, ablation, and dataset-construction scripts;
- the released 12-class commodity detection dataset in YOLO format;
- configuration for the public single-product RPC subset used for external
  validation;
- result tables used in the manuscript and reviewer response.

Large trained weights are not included in the repository. They can be reproduced
from the provided configuration files and scripts.

## Main Results

All results below use an input size of 640 on the 12-class commodity dataset.

| Method | EMA | ECA | Weak-class augmentation | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- | ---: | ---: |
| YOLO11 | No | No | No | ~0.970 | ~0.78350 |
| YOLO11_EMA | Yes | No | No | ~0.986 | 0.84334 |
| YOLO11_ECA | No | Yes | No | 0.96051 | 0.77967 |
| YOLO11_EMA_ECA | Yes | Yes | No | 0.98570 | 0.85152 |
| WCA-Det / YOLO11_EMA_ECA_WeakAug | Yes | Yes | Yes | 0.98640 | 0.85535 |

Compared with YOLO11_EMA, WCA-Det improves mAP@0.5:0.95 by 1.20 percentage
points while keeping the inference-side model compact.

### Seed-Repeat Stability

To reduce the influence of random initialization and data-ordering effects, the
final weak-class-aware fine-tuning stage was repeated with three random seeds.

| Run | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| Seed 0 | 0.97512 | 0.94880 | 0.98640 | 0.85535 |
| Seed 1 | 0.97512 | 0.94976 | 0.98557 | 0.85503 |
| Seed 2 | 0.97511 | 0.94922 | 0.98661 | 0.85466 |
| Mean | 0.97512 | 0.94926 | 0.98619 | 0.85501 |
| Std. | 0.00001 | 0.00048 | 0.00055 | 0.00035 |

### Public RPC Single-Product Subset

We also provide validation results on a public single-product subset derived
from the Retail Product Checkout (RPC) dataset. The subset contains 12 selected
commodity categories with train/validation/test splits. See
[docs/public_rpc_single_results.md](docs/public_rpc_single_results.md).

| Method | Best epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8m | 100 | 0.97229 | 0.99496 | 0.99440 | 0.97083 |
| YOLOv10m | 88 | 0.95404 | 0.96531 | 0.98959 | 0.95981 |
| YOLO11 | 97 | 0.97758 | 0.99880 | 0.99460 | 0.92070 |
| YOLO11_EMA | 96 | 0.98527 | 0.98150 | 0.99322 | 0.91165 |
| YOLO11_ECA | 89 | 0.99109 | 0.99065 | 0.99480 | 0.91220 |
| WCA-Det | 94 | 0.99369 | 0.99513 | 0.99500 | 0.92893 |

## Repository Structure

```text
wca-det-public/
  configs/                 Model and dataset YAML files
  data/                    YOLO-format commodity dataset
    images/train/
    images/val/
    labels/train/
    labels/val/
    train_weak_aug.txt
  scripts/                 Training, validation, and dataset preparation scripts
  docs/                    Experiment logs and manuscript result tables
```

## Installation

Create a Python environment with PyTorch and Ultralytics installed.

```bash
pip install -r requirements.txt
```

The experiments were run with:

- Python 3.10
- Ultralytics 8.4.51
- PyTorch CUDA environment
- NVIDIA GeForce RTX 5060 Ti

## Reproducibility Settings

Unless otherwise specified, the main experiments use:

- input size: 640;
- epochs: 200;
- batch size: 8;
- optimizer: SGD;
- initial learning rate: 0.01;
- weight decay: 0.0005;
- mosaic: 1.0;
- mixup: 0.15;
- copy-paste: 0.15;
- HSV value augmentation: 0.4;
- close-mosaic: 15.

The final weak-class-aware fine-tuning stage is initialized from trained
YOLO11_EMA_ECA weights and optimized for 5 epochs with head-only updating,
SGD, learning rate `5e-6`, and mosaic/mixup/copy-paste disabled.

## Training

Place `yolo11m.pt` in the repository root, or provide its path with
`--weights`.

```bash
python scripts/train_wcadet.py ^
  --mode train ^
  --data configs/dataset_merged_weak_aug.yaml ^
  --model-yaml configs/yolo11-ema-eca-perfect.yaml ^
  --weights yolo11m.pt ^
  --epochs 200 ^
  --batch 8 ^
  --imgsz 640 ^
  --workers 4 ^
  --name wca_det_yolo11_ema_eca
```

## Validation

```bash
python scripts/train_wcadet.py ^
  --mode val ^
  --data configs/dataset_merged.yaml ^
  --model-yaml configs/yolo11-ema-eca-perfect.yaml ^
  --weights runs/detect/wca_det_yolo11_ema_eca/weights/best.pt ^
  --batch 8 ^
  --imgsz 640 ^
  --name wca_det_val
```

## Dataset

The dataset contains 12 fine-grained commodity classes in YOLO detection format.
See [DATASET.md](DATASET.md) for details.

## Public Dataset Configuration

For external validation on the public RPC single-product subset, place or
extract the subset locally and update the `path` field in:

```text
configs/rpc_single_product_subset.yaml
```

The local dataset should follow the YOLO directory layout:

```text
rpc_single_product_subset_yolo/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

## GitHub Update Workflow

After editing files locally, update the public repository with:

```powershell
cd D:\spsb\github_release\wca-det-public
D:\Git\cmd\git.exe status --short
D:\Git\cmd\git.exe add .
D:\Git\cmd\git.exe commit -m "Update public data, code, and reproducibility materials"
D:\Git\cmd\git.exe push origin main
```

## Citation

If this code or dataset helps your research, please cite the related manuscript
or use the metadata in [CITATION.cff](CITATION.cff).
