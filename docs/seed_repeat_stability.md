# Seed-Repeat Stability Results

To reduce the influence of random initialization and data-ordering effects, the
final weak-class-aware fine-tuning stage was repeated with three random seeds.

| Run | Best epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WCA_SeedRepeat_S0 | 3 | 0.97512 | 0.94880 | 0.98640 | 0.85535 |
| WCA_SeedRepeat_S1 | 1 | 0.97512 | 0.94976 | 0.98557 | 0.85503 |
| WCA_SeedRepeat_S2 | 1 | 0.97511 | 0.94922 | 0.98661 | 0.85466 |
| Mean | - | 0.97512 | 0.94926 | 0.98619 | 0.85501 |
| Std. | - | 0.00001 | 0.00048 | 0.00055 | 0.00035 |

The repeated runs show that the final improvement is stable under small changes
in random seed during the fine-tuning stage.
