# Paper Result Tables

## Table 1. Main Ablation Results at 640

Main metric: `mAP50-95`.

| Method | Input size | EMA | ECA | Training data | mAP50 | mAP50-95 | Gain over YOLO11 | Gain over YOLO11_EMA | Status |
|---|---|---|---|---|---:|---:|---:|---:|---|
| YOLO11 | 640 | No | No | Original | ~0.970 | ~0.78350 | - | - | Done |
| YOLO11_EMA | 640 | Yes | No | Original | ~0.986 | 0.84334 | +0.05984 | - | Done |
| YOLO11_SimAM | 640 | No | No | Original | ~0.968 | 0.81356 | +0.03006 | -0.02978 | Done, reference only |
| YOLO11_ECA | 640 | No | Yes | Original | 0.97529 | 0.81833 | +0.03483 | -0.02501 | Done |
| YOLO11_EMA_ECA | 640 | Yes | Yes | Original | 0.98570 | 0.85152 | +0.06802 | +0.00818 | Done |
| YOLO11_EMA_ECA (ours) | 640 | Yes | Yes | Weak-class augmentation | 0.98640 | 0.85535 | +0.07185 | +0.01201 | Final candidate |

## Table 2. Improvement Breakdown at 640

| Comparison | mAP50-95 before | mAP50-95 after | Absolute gain | Percentage-point gain |
|---|---:|---:|---:|---:|
| YOLO11 -> YOLO11_EMA | ~0.78350 | 0.84334 | +0.05984 | +5.98 |
| YOLO11 -> YOLO11_ECA | ~0.78350 | 0.81833 | +0.03483 | +3.48 |
| YOLO11_EMA -> YOLO11_EMA_ECA | 0.84334 | 0.85152 | +0.00818 | +0.82 |
| YOLO11_EMA_ECA original data -> YOLO11_EMA_ECA weak-class augmentation | 0.85152 | 0.85535 | +0.00383 | +0.38 |
| YOLO11_EMA -> YOLO11_EMA_ECA (ours) | 0.84334 | 0.85535 | +0.01201 | +1.20 |
| YOLO11 -> YOLO11_EMA_ECA (ours) | ~0.78350 | 0.85535 | +0.07185 | +7.19 |

## Table 3. Final Model Per-Class Results at 640

Final model:

`YOLO11_EMA_ECA (ours)`

Checkpoint:

`D:/spsb/runs/detect/runs/train/yolo11m_EMA_ECA_L22_A05_WeakAug_FromBest_LR5e6_E5/weights/best.pt`

| Class ID | Class name | mAP50-95 | Note |
|---:|---|---:|---|
| 0 | bsk1 | 0.872 | - |
| 1 | bsk1_con | 0.937 | - |
| 2 | bsk1_wt | 0.752 | Weak-augmented class |
| 3 | bss_large | 0.773 | Weak-augmented class |
| 4 | fd_can | 0.925 | - |
| 5 | hs_can | 0.793 | Weak-augmented class |
| 6 | kkkl_can | 0.858 | - |
| 7 | mnd_can | 0.864 | - |
| 8 | nfsq | 0.727 | Weak-augmented class |
| 9 | xb_can | 0.942 | - |
| 10 | xb_wt | 0.884 | - |
| 11 | xb | 0.935 | - |
| all | all classes | 0.85535 | Final average |

## Table 4. Runs to Use in the Paper

| Purpose | Run directory | Use in paper |
|---|---|---|
| YOLO11 baseline | `D:/spsb/runs/detect/runs/ablation/yolo11m_01_Baseline` | Main comparison |
| YOLO11_EMA | `D:/spsb/runs/detect/runs/ablation/yolo11m_02_EMA_Only` | Main comparison |
| YOLO11_SimAM | `D:/spsb/runs/detect/runs/ablation/yolo11m_03_SimAM_Only` | Reference comparison |
| YOLO11_ECA | `D:/spsb/runs/detect/runs/train/yolo11m_04_ECA_Only_Fair_E200` | ECA-only ablation |
| YOLO11_EMA_ECA | `D:/spsb/runs/detect/runs/train/yolo11m_EMA_ECAInject_L22_A05_B16_FocalG05_FromBest_LR5e6_NoWarm_E3` | EMA+ECA ablation |
| YOLO11_EMA_ECA (ours, weak-class augmentation training) | `D:/spsb/runs/detect/runs/train/yolo11m_EMA_ECA_L22_A05_WeakAug_FromBest_LR5e6_E5` | Final model |

## ECA-Only Note

`YOLO11_ECA` reached `mAP50-95 = 0.81833` after fair 200-epoch training from `yolo11m.pt`, improving over the YOLO11 baseline by about `+3.48` percentage points. The earlier short fine-tune run from the baseline checkpoint is excluded because it was not a fair standalone ECA training protocol.
