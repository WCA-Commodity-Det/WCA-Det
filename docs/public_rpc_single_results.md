# Public RPC Single-Product Subset Validation

This document reports external validation results on a 12-category
single-product subset derived from the public Retail Product Checkout (RPC)
dataset. The subset is used to evaluate whether the released training pipeline
can be applied beyond the self-built commodity dataset.

## Dataset

The public subset follows the YOLO detection format:

```text
rpc_single_product_subset_yolo/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

The selected categories are:

```text
196_stationery, 131_chocolate, 190_tissue, 197_stationery,
173_personal_hygiene, 124_chocolate, 172_personal_hygiene,
177_tissue, 148_candy, 194_stationery, 158_seasoner, 63_dessert
```

## Results

All models were trained and evaluated under the same public-subset protocol.

| Method | Best epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8m | 100 | 0.97229 | 0.99496 | 0.99440 | 0.97083 |
| YOLOv10m | 88 | 0.95404 | 0.96531 | 0.98959 | 0.95981 |
| YOLO11 | 97 | 0.97758 | 0.99880 | 0.99460 | 0.92070 |
| YOLO11_EMA | 96 | 0.98527 | 0.98150 | 0.99322 | 0.91165 |
| YOLO11_ECA | 89 | 0.99109 | 0.99065 | 0.99480 | 0.91220 |
| YOLO11_EMA_ECA / WCA-Det | 94 | 0.99369 | 0.99513 | 0.99500 | 0.92893 |

## Notes

The RPC single-product subset is easier than crowded checkout scenes because
each image contains a single centered commodity instance. Therefore, these
results should be interpreted as an external reproducibility and generalization
check rather than as the primary weak-class analysis. The main contribution and
ablation study in the manuscript are based on the self-built 12-class
fine-grained commodity dataset.
