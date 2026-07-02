# Public Release Checklist

This checklist records the materials that should be kept synchronized with the
manuscript and reviewer response.

## Code

- `scripts/train_wcadet.py`: main training and validation entry point.
- `scripts/run_ablation.py`: ablation experiment runner.
- `scripts/make_weak_aug_dataset.py`: weak-class augmentation data generation.
- `scripts/merge_datasets.py`: dataset merging utility.

## Configurations

- `configs/yolo11-baseline.yaml`: YOLO11 baseline.
- `configs/yolo11-ema.yaml`: YOLO11 with EMA enhancement.
- `configs/yolo11-ema-eca-perfect.yaml`: WCA-Det inference model.
- `configs/dataset_merged.yaml`: self-built 12-class dataset.
- `configs/dataset_merged_weak_aug.yaml`: weak-class training data.
- `configs/rpc_single_product_subset.yaml`: public RPC subset validation.

## Data

- `data/images/` and `data/labels/`: released self-built 12-class commodity
  dataset in YOLO format.
- `data/train_weak_aug.txt`: training list for weak-class-aware augmentation.

## Reproducibility Documents

- `README.md`: repository overview, main results, commands, and environment.
- `DATASET.md`: dataset description, class names, and split protocol.
- `MODEL_CARD.md`: intended use, evaluation, limitations, and external
  validation.
- `docs/seed_repeat_stability.md`: three-seed stability results.
- `docs/public_rpc_single_results.md`: public RPC single-product subset results.

## Before Pushing to GitHub

1. Confirm no single file exceeds the GitHub 100 MB file limit.
2. Confirm no temporary files are included, such as `.cache`, `runs/`, `logs/`,
   `.pt`, `.pth`, `.onnx`, or `.engine`.
3. Run:

```powershell
D:\Git\cmd\git.exe status --short
```

4. Commit with a descriptive message, for example:

```powershell
D:\Git\cmd\git.exe add .
D:\Git\cmd\git.exe commit -m "Update public WCA-Det reproducibility materials"
D:\Git\cmd\git.exe push origin main
```
