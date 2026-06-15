# WCA-Det: Weak-Class-Aware Dual-Attention Commodity Detection

This repository contains the code, configuration files, and 12-class commodity
detection dataset used in our Electronics manuscript.

WCA-Det is built on a YOLO11-based detector with EMA feature enhancement, ECA
channel attention, and a training-only weak-class augmentation strategy. The
weak-class augmentation branch is used only during training and does not add
inference parameters or GFLOPs.

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

## Citation

If this code or dataset helps your research, please cite the related manuscript
or use the metadata in [CITATION.cff](CITATION.cff).

