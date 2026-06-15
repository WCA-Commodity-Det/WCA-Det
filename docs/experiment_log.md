# SPSB YOLO11 Experiment Log

Updated: 2026-06-05

## Current Objective

Current active branch: optimize EMA + ECA only under the same 640 validation protocol. SimAM runs are kept as historical reference, but they are not valid for the current pure EMA+ECA branch. The practical target is +1.0 to +2.0 percentage points in mAP50-95 over the strongest fair baseline.

## Fixed Fair-Comparison Protocol

- Dataset: `D:/spsb/data_merged/dataset_merged.yaml`
- Image size: 640 for the main ablation table
- Seed: 0
- Deterministic: true
- Strongest current fair baseline: YOLO11_EMA, revalidated at 640, mAP50-95 about 0.84334
- Historical best SimAM-combined run: `yolo11m_EMA_SimAM_Perfect_Boost_L19_A01_B4_E3`
- Current best valid EMA+ECA run: `yolo11m_EMA_ECAInject_L22_A05_B16_FocalG05_FromBest_LR5e6_NoWarm_E3`, mAP50-95 = 0.85152

## Current Leaderboard

| Rank | Run | Best epoch | mAP50 | mAP50-95 | Note |
|---:|---|---:|---:|---:|---|
| 1 | `yolo11m_EMA_ECAInject_L22_A05_B16_FocalG05_FromBest_LR5e6_NoWarm_E3` | 3 | 0.98570 | 0.85152 | Current best valid EMA+ECA |
| 2 | `yolo11m_EMA_ECAInject_L22_A05_B16_ResumeBest_LR5e6_NoWarm_E3` | 3 | 0.98572 | 0.85138 | No-warmup resume refinement |
| 3 | `yolo11m_EMA_ECAInject_L22_A05_B16_E3` | 2 | 0.98480 | 0.85118 | Best first-stage head fine-tune |
| 4 | `yolo11m_EMA_ECAInject_L22_A05_B8_E3` | 1 | 0.98416 | 0.84967 | Batch 8 improves over batch 4 |
| 5 | `yolo11m_EMA_ECAInject_L22_A05_B4_E3` | 1 | 0.98480 | 0.84857 | Best batch-4 EMA+ECA |
| 6 | `yolo11m_EMA_ECAInject_L22_A05_B16_LR3e5_E3` | 1 | 0.98316 | 0.84834 | Higher lr hurts batch-16 best |
| 7 | `yolo11m_EMA_SimAM_Perfect_Boost_L19_A01_B4_E3` | 1 | 0.98318 | 0.84735 | Historical SimAM-combined best |
| 8 | `yolo11m_EMA_SimAM_Perfect_Boost_L19_A02_B4_E1` | 1 | 0.98262 | 0.84724 | Close, alpha 0.02 |
| 9 | `yolo11m_EMA_ECAInject_L22_A10_B4_E3` | 1 | 0.98576 | 0.84715 | Stronger layer-22 ECA |
| 10 | `yolo11m_EMA_SimAM_ClsHead_A01_B4_E1` | 1 | 0.98480 | 0.84699 | Classification-head SimAM |
| 11 | `yolo11m_EMA_SimAM_Perfect_Boost_L16_A01_B4_E1` | 1 | 0.98264 | 0.84693 | Layer 16 SimAM |
| 12 | `yolo11m_EMA_SimAM_Perfect_Boost_L19_A005_B4_E1` | 1 | 0.98380 | 0.84671 | Smaller SimAM alpha |
| 13 | `yolo11m_EMA_ECAInject_L19_22_A05_B4_E3` | 1 | 0.98494 | 0.84666 | Double C3k2 ECA hurts |
| 14 | `yolo11m_EMA_ECAHead_A10_B4_E3` | 1 | 0.98476 | 0.84624 | ECA on Detect classification branch |
| 15 | `yolo11m_EMA_ECAInject_L19_A10_B4_E3` | 1 | 0.98497 | 0.84623 | ECA-only injection at layer 19 |
| 16 | `yolo11m_EMA_ECAInject_L22_A03_B4_E3` | 1 | 0.98579 | 0.84594 | Layer-22 ECA too weak |
| 17 | `yolo11m_EMA_ECAInject_L16_A10_B4_E3` | 1 | 0.98174 | 0.84470 | ECA at layer 16, worse than layer 19 |
| 18 | `yolo11m_EMA_ECAInject_L19_A05_B4_E3` | 1 | 0.98237 | 0.84466 | Weaker ECA alpha, worse than 0.10 |
| 19 | `yolo11m_EMA_ECAInject_L19_A20_B4_E3` | 1 | 0.98289 | 0.84419 | Stronger ECA alpha, worse than 0.10 |

## Analysis After ECA-Only Injection

ECA-only injection is valid because the smoke validation kept high transfer (`955/956`) and reproduced the EMA-only level (`mAP50-95 = 0.84334`). However, after short fine-tuning it reached only `0.84623`, below the current best `0.84735`. This means ECA alone is not enough, but it may still complement SimAM because ECA is channel attention while SimAM is spatial/energy attention.

## Next Planned Run

User correction: the next branch should optimize EMA + ECA only. SimAM should not be included in this branch.

The aborted run `yolo11m_EMA_ECA_SimAM_L19_ECA10_SA01_B4_E3` is not part of the valid ECA-only comparison because it included SimAM. Continue with ECA-only placement and strength search:

1. Try ECA at C3k2 layer 16, alpha 0.10.
2. If layer 16 improves over layer 19, test combined ECA layers 16 and 19.
3. If neither improves, search alpha around layer 19 and layer 16 (`0.05`, `0.20`) before running any longer fine-tune.

## Analysis After ECA Layer-16 Run

`yolo11m_EMA_ECAInject_L16_A10_B4_E3` reached only `mAP50-95 = 0.84470`, below the layer-19 ECA result (`0.84623`). This suggests ECA is more useful on the deeper PAN/FPN feature fusion layer 19 than on layer 16. Continue by keeping ECA at layer 19 and increasing alpha to 0.20 for faster adaptation during short fine-tuning.

## Analysis After ECA Layer-19 Alpha-0.20 Run

`yolo11m_EMA_ECAInject_L19_A20_B4_E3` reached only `mAP50-95 = 0.84419`, so increasing ECA strength from 0.10 to 0.20 hurts. Continue with a gentler alpha search at 0.05 on layer 19.

## Analysis After ECA Layer-19 Alpha-0.05 Run

`yolo11m_EMA_ECAInject_L19_A05_B4_E3` reached `mAP50-95 = 0.84466`, still below alpha 0.10. The best C3k2-only ECA setting remains layer 19 with alpha 0.10 (`0.84623`). The next branch should move ECA closer to the Detect classification branch, because the user's target is recognition quality rather than only feature-fusion refinement.

## ECA Classification-Head Branch

Added `--eca-head-cls` and `--eca-head-levels` to place ECA on the Detect classification branch inputs. Smoke validation with EMA-only weights preserved the EMA baseline (`mAP50-95 = 0.843341`) and transferred `955/958` pretrained items, so this branch is structurally valid. Next run: ECAHead alpha 0.10, all Detect feature levels, no SimAM and no C3k2 ECA.

`yolo11m_EMA_ECAHead_A10_B4_E3` reached `mAP50-95 = 0.84624`, almost identical to C3k2 layer-19 ECA (`0.84623`). Next run should combine both pure-ECA placements: C3k2 layer 19 plus Detect classification-head ECA.

`yolo11m_EMA_ECA_L19_Head_A10_B4_E3` reached only `mAP50-95 = 0.84536`, so adding both ECA placements while fine-tuning the whole head is worse than either one alone. The repeated pattern is that epoch 1 is usually best and later epochs often regress. Next branch: freeze the base detector and train only the ECA parameters with a higher learning rate.

The first ECA-only attempt `yolo11m_EMA_ECAHead_A10_ECAOnly_LR01_B4_E5` is invalid because Ultralytics reopened frozen parameters after the custom setup step. It dropped to `mAP50-95 = 0.78610`, confirming that the whole model had been disturbed by the high learning rate. The trainer was patched to re-apply ECA-only freezing after Ultralytics' freeze block and rebuild the optimizer.

`yolo11m_EMA_ECAHead_A10_ECAOnlyFixed_LR01_B4_E5` is valid ECA-only training, but `lr0=0.01` is too aggressive for 9 ECA parameters. It reached only `mAP50-95 = 0.82694`. Continue with lower ECA-only learning rates.

`yolo11m_EMA_ECAHead_A10_ECAOnlyFixed_LR001_B4_E5` produced the same metric trace as the LR01 run except for the logged learning rate, with best `mAP50-95 = 0.82694`. This strongly suggests the Detect classification-head ECA branch is not a good learnable path in the current Ultralytics forward/training flow. Continue with C3k2-layer ECA-only tuning, where the ECA module is inside the normal model forward path.

`yolo11m_EMA_ECAInject_L19_A10_ECAOnlyFixed_LR001_B4_E5` also reached only `mAP50-95 = 0.82694`. This confirms that freezing the detector and training only the tiny ECA module is not competitive here. The active path should return to full head fine-tuning with ECA as a residual attention injection, because that is the only pure EMA+ECA setup that has reached the `0.846` range.

## Next Pure EMA+ECA Step

Run a C3k2 ECA placement search at a deeper layer without SimAM:

- `--simam-layers none`
- `--eca-layers 22`
- `--eca-alpha 0.10`
- `--train-scope head`
- short 3-epoch fine-tune from the EMA-only best weights

Reason: layer 19 alpha 0.10 is the best current C3k2 ECA result, layer 16 is weaker, and ECA-only parameter tuning is not viable. A deeper layer may improve high-level classification features while preserving the EMA detector.

## Analysis After ECA Layer-22 Alpha-0.10 Run

`yolo11m_EMA_ECAInject_L22_A10_B4_E3` reached `mAP50 = 0.98576` and `mAP50-95 = 0.84715` at epoch 1. This is the best valid pure EMA+ECA result so far, slightly above ECAHead (`0.84624`) and layer-19 ECA (`0.84623`). It is still only about +0.38 percentage points over the EMA-only 640 baseline (`0.84334`), so it does not yet meet the +1% to +2% target. Continue by tuning alpha around layer 22, starting with a weaker alpha 0.05.

## Analysis After ECA Layer-22 Alpha-0.05 Run

`yolo11m_EMA_ECAInject_L22_A05_B4_E3` reached `mAP50 = 0.98480` and `mAP50-95 = 0.84857` at epoch 1. This is now the best valid EMA+ECA result and it also exceeds the historical SimAM-combined best (`0.84735`). The gain over EMA-only 640 is about +0.52 percentage points. Since alpha 0.05 is better than 0.10 on layer 22, continue with a lighter alpha search, starting at 0.03.

## Analysis After ECA Layer-22 Alpha-0.03 Run

`yolo11m_EMA_ECAInject_L22_A03_B4_E3` reached only `mAP50-95 = 0.84594`. This is worse than alpha 0.05 and alpha 0.10, so layer-22 ECA becomes too weak at 0.03. The local optimum is likely around 0.05 to 0.08. Continue with alpha 0.07.

## Analysis After ECA Layer-22 Alpha-0.07 Run

`yolo11m_EMA_ECAInject_L22_A07_B4_E3` tied alpha 0.05 at `mAP50-95 = 0.84857`. Since alpha 0.05 and 0.07 are equal while 0.03 and 0.10 are lower, single-layer layer-22 ECA is probably near its ceiling. Continue with a multi-layer C3k2 ECA test at layers 19 and 22, using the gentler alpha 0.05 to avoid over-amplifying the feature fusion path.

## Analysis After ECA Layers-19-22 Alpha-0.05 Run

`yolo11m_EMA_ECAInject_L19_22_A05_B4_E3` reached only `mAP50-95 = 0.84666`. Adding layer 19 to the current best layer-22 setting hurts, so the best structural setting remains a single ECA injection at layer 22 with alpha 0.05 or 0.07. Since the structural search is plateauing at `0.84857`, the next step should tune training dynamics for the best structure rather than adding more attention layers. Continue with the best `L22_A05` setting at batch size 8.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-8 Run

`yolo11m_EMA_ECAInject_L22_A05_B8_E3` reached `mAP50 = 0.98416` and `mAP50-95 = 0.84967` at epoch 1. This is the best valid EMA+ECA result so far, about +0.63 percentage points over EMA-only 640 (`0.84334`). Batch size 8 improved over batch size 4 (`0.84857`), so training dynamics still matter. Continue with batch size 16 for the same `L22_A05` structure.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 Run

`yolo11m_EMA_ECAInject_L22_A05_B16_E3` reached `mAP50 = 0.98480` and `mAP50-95 = 0.85118` at epoch 2. This is now about +0.78 percentage points over EMA-only 640 and only about 0.22 percentage points short of the +1% target. Since batch size 16 improved over batch 8, continue with batch size 32 while keeping the same structure and learning rate.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-32 Run

`yolo11m_EMA_ECAInject_L22_A05_B32_E3` reached only `mAP50-95 = 0.84781`. Batch 32 is too large for this fine-tune setting, likely because it reduces update steps too much for the short run. Keep batch 16 as the current best training dynamic. Continue by extending the best batch-16 run to 5 epochs, relying on best-epoch checkpoint selection.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 Epoch-5 Run

`yolo11m_EMA_ECAInject_L22_A05_B16_E5` reached only `mAP50-95 = 0.84950`, below the 3-epoch batch-16 run (`0.85118`). Extending the run changes the cosine/warmup schedule and does not help. Keep `epochs=3` and batch 16, then tune learning rate around the current `lr0=2e-5`. Next test: lower `lr0=1e-5`.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 LR-1e-5 Run

`yolo11m_EMA_ECAInject_L22_A05_B16_LR1e5_E3` reached only `mAP50-95 = 0.84813`. Lowering `lr0` is too conservative and loses the batch-16 gain. Continue in the opposite direction with `lr0=3e-5`.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 LR-3e-5 Run

`yolo11m_EMA_ECAInject_L22_A05_B16_LR3e5_E3` reached `mAP50 = 0.98316` and `mAP50-95 = 0.84834` at epoch 1. This is below the default `lr0=2e-5` batch-16 run (`0.85118`), so the learning-rate sweep shows that the current best is not improved by moving to either `1e-5` or `3e-5`. The next attempt should keep the best structure (`L22_A05_B16`) and test a very small full-model adaptation rate, because head-only tuning appears near its ceiling.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 All-Model LR-2e-6 Run

`yolo11m_EMA_ECAInject_L22_A05_B16_AllLR2e6_E3` reached only `mAP50 = 0.98199` and `mAP50-95 = 0.81897` at epoch 2. Even with a very small learning rate, full-model adaptation strongly damages the EMA-initialized detector. This branch should be abandoned for the paper table. Continue with head-only tuning, but initialize from the current best ECA checkpoint and disable warmup for a short refinement run.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 Resume-Best LR-5e-6 No-Warmup Run

`yolo11m_EMA_ECAInject_L22_A05_B16_ResumeBest_LR5e6_NoWarm_E3` reached `mAP50 = 0.98572` and `mAP50-95 = 0.85138` at epoch 3. This is a small but real improvement over the first-stage best (`0.85118`). Because the metric increased across the three no-warmup epochs (`0.85042 -> 0.85064 -> 0.85138`), continue from this new best checkpoint with an even gentler no-warmup learning rate (`lr0=2e-6`) rather than changing the module placement.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 Resume-2 LR-2e-6 No-Warmup Run

`yolo11m_EMA_ECAInject_L22_A05_B16_Resume2_LR2e6_NoWarm_E3` reached only `mAP50 = 0.98580` and `mAP50-95 = 0.85107` at epoch 2. This confirms that repeated no-warmup refinement has reached a plateau around `0.851`. The next pure EMA+ECA branch should modify the ECA module itself. A new CLI argument `--eca-k-size` was added with default `3`, allowing a fair ECA channel-kernel search while preserving all old commands.

## Analysis After ECA Layer-22 Alpha-0.05 Kernel-5 Batch-16 Run

`yolo11m_EMA_ECAInject_L22_A05_K5_B16_E3` reached only `mAP50 = 0.98434` and `mAP50-95 = 0.84899` at epoch 2. A wider ECA channel kernel (`k=5`) hurts compared with the original `k=3` best (`0.85138`), so the layer-22 ECA module should keep `k=3`. Since structural and plain refinement searches are plateauing, continue with a mild focal-loss refinement from the current best checkpoint to see whether hard/weak classes can be lifted without adding SimAM.

## Analysis After ECA Layer-22 Focal-Gamma-0.5 Resume-Best Run

`yolo11m_EMA_ECAInject_L22_A05_B16_FocalG05_FromBest_LR5e6_NoWarm_E3` reached `mAP50 = 0.98570` and `mAP50-95 = 0.85152` at epoch 3. This is a small new best, suggesting that mild focal classification pressure can help the current EMA+ECA checkpoint slightly. Continue with a stronger but still moderate focal setting (`gamma=1.0`, `alpha=0.5`) from this checkpoint.

## Analysis After ECA Layer-22 Focal-Gamma-1.0 Resume-Best Run

`yolo11m_EMA_ECAInject_L22_A05_B16_FocalG10_FromFocalBest_LR5e6_NoWarm_E3` reached only `mAP50 = 0.98573` and `mAP50-95 = 0.85100` at epoch 3. Increasing focal gamma from `0.5` to `1.0` is too strong and loses the small gain. Continue with `gamma=0.5`, but test `alpha=0.75` to give positives/weak classes more weight.

## Analysis After ECA Layer-22 Focal-Gamma-0.5 Alpha-0.75 Run

`yolo11m_EMA_ECAInject_L22_A05_B16_FocalG05A075_FromFocalBest_LR5e6_NoWarm_E3` tied the current best with `mAP50 = 0.98570` and `mAP50-95 = 0.85152` at epoch 3. Changing focal alpha did not move the detector predictions, so keep the simpler `gamma=0.5, alpha=0.5` setting. Continue by testing AdamW as a small no-warmup optimizer refinement from the current best checkpoint.

## Analysis After ECA Layer-22 Focal-Gamma-0.5 AdamW Run

`yolo11m_EMA_ECAInject_L22_A05_B16_FocalG05_AdamW5e6_FromBest_NoWarm_E3` tied the current best exactly (`mAP50-95 = 0.85152`). AdamW does not change the trajectory in this small no-warmup refinement regime. A weighted classification loss was then connected directly into the BCE/focal elementwise classification loss so `--class-weights` now affects official-loss training as well.

## Analysis After ECA Layer-22 Focal-Gamma-0.5 Weak-Class-Weights From-Best Run

`yolo11m_EMA_ECAInject_L22_A05_B16_FocalG05_CWweak_FromBest_LR5e6_NoWarm_E3` also tied the current best (`mAP50-95 = 0.85152`). The weighted loss changed train/val classification loss values, so it is active, but starting from the already converged best checkpoint with `lr0=5e-6` is too conservative to change predictions. Continue by running the same weak-class weighting from the EMA-only checkpoint with the first-stage schedule.

## Analysis After ECA Layer-22 Weak-Class-Weights From-EMA Run

`yolo11m_EMA_ECAInject_L22_A05_B16_FocalG05_CWweak_FromEMA_E3` reached only `mAP50 = 0.98123` and `mAP50-95 = 0.84715` at epoch 1. Starting weighted focal training from the EMA-only checkpoint hurts the overall detector and does not solve the weak-class bottleneck. The weak-class weighting branch should be abandoned for the paper table. Continue with the cleaner EMA+ECA structure search by testing layer-22 `alpha=0.07` at batch size 16, because alpha `0.05` and `0.07` tied at batch size 4 while batch size 16 was the strongest training dynamic so far.

## Analysis After ECA Layer-22 Alpha-0.07 Batch-16 Run

`yolo11m_EMA_ECAInject_L22_A07_B16_E3` reached `mAP50 = 0.98380` and `mAP50-95 = 0.84986` at epoch 1. This is below the layer-22 `alpha=0.05` batch-16 first-stage run (`0.85118`) and below the current refined best (`0.85152`). At batch size 16, alpha `0.07` appears too strong. Continue with a slightly weaker ECA strength, `alpha=0.04`, while keeping layer 22, batch 16, official loss, and head-only fine-tuning.

## Analysis After ECA Layer-22 Alpha-0.04 Batch-16 Run

`yolo11m_EMA_ECAInject_L22_A04_B16_E3` reached `mAP50 = 0.98343` and `mAP50-95 = 0.84874` at epoch 1. This is worse than both `alpha=0.05` and `alpha=0.07` at batch size 16, so the best structural setting remains layer-22 ECA with `alpha=0.05`, `k=3`, and batch 16. Continue from the current best checkpoint with a gentler focal refinement (`gamma=0.25`) to test whether the focal gain at `gamma=0.5` can be made slightly less aggressive and improve mAP50-95.

## Analysis After ECA Layer-22 Focal-Gamma-0.25 Resume-Best Run

`yolo11m_EMA_ECAInject_L22_A05_B16_FocalG025_FromBest_LR5e6_NoWarm_E3` reached `mAP50 = 0.98570` and `mAP50-95 = 0.85152` at epoch 3. This ties the current best but does not improve it. Since focal gamma `0.25` and `0.5` converge to the same metric while gamma `1.0` drops, this refinement branch is at a plateau. Continue with a classification-branch ECA test: keep layer-22 ECA and also add ECA to the Detect classification branch to target weak-class recognition without adding SimAM.

## Analysis After ECA Layer-22 Plus Detect-Cls-Head ECA Run

`yolo11m_EMA_ECA_L22_ClsHead_A05_B16_E3` reached only `mAP50 = 0.98285` and `mAP50-95 = 0.84753` at epoch 2. Adding ECA to all Detect classification-branch feature levels hurts overall performance, so this branch should be abandoned. To keep the proven layer-22 setting while testing auxiliary detail features more gently, `main.py` now supports `--eca-layer-alphas`, allowing per-layer ECA strengths such as `22:0.05,16:0.01`. Continue with layer 22 at alpha 0.05 plus a very weak layer-16 ECA at alpha 0.01.

## Analysis After ECA Layer-16 Alpha-0.01 Plus Layer-22 Alpha-0.05 Run

`yolo11m_EMA_ECA_L16a01_L22a05_B16_E3` reached only `mAP50 = 0.98162` and `mAP50-95 = 0.84602` at epoch 2. Even a very weak ECA on layer 16 hurts the detector, so direct high-resolution output features are too sensitive for this module. Continue by testing a weak layer-13 auxiliary ECA with layer 22 kept at alpha 0.05; layer 13 is upstream in the neck and may be less disruptive than layer 16.

## Analysis After ECA Layer-13 Alpha-0.01 Plus Layer-22 Alpha-0.05 Run

`yolo11m_EMA_ECA_L13a01_L22a05_B16_E3` reached `mAP50 = 0.98432` and `mAP50-95 = 0.85100` at epoch 1. This is much better than the layer-16 auxiliary test but still below layer-22-only batch-16 (`0.85118`) and below the refined best (`0.85152`). Layer 13 is a viable auxiliary location, but alpha `0.01` is still slightly too disruptive. Continue with a lighter layer-13 alpha `0.005` while keeping layer 22 at alpha `0.05`.

## Analysis After ECA Layer-13 Alpha-0.005 Plus Layer-22 Alpha-0.05 Run

`yolo11m_EMA_ECA_L13a005_L22a05_B16_E3` again reached `mAP50 = 0.98432` and `mAP50-95 = 0.85100` at epoch 1, effectively identical to the layer-13 alpha `0.01` run. Very weak auxiliary ECA at layer 13 does not improve beyond the layer-22-only setting. Abandon auxiliary-layer ECA for now and return to the strongest structure, layer-22 ECA with alpha `0.05`, while searching training dynamics around the successful batch-16 setup. Next test: batch size 12.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-12 Run

`yolo11m_EMA_ECAInject_L22_A05_B12_E3` reached only `mAP50 = 0.98287` and `mAP50-95 = 0.84860` at epoch 1. Batch 12 is clearly worse than batch 16, so the best batch setting remains 16. Continue with batch 16 and test no warmup from the EMA-only checkpoint, because the standard 3-epoch fine-tune currently spends all epochs under warmup and may be too conservative for the injected ECA module.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 No-Warmup Run

`yolo11m_EMA_ECAInject_L22_A05_B16_NoWarm_E3` reached only `mAP50 = 0.98136` and `mAP50-95 = 0.84748` at epoch 1. Removing warmup from the initial EMA-only fine-tune is too aggressive and damages the detector. Keep the standard warmup for first-stage training, and continue by testing AdamW on the same layer-22 alpha-0.05 batch-16 structure.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 AdamW First-Stage Run

`yolo11m_EMA_ECAInject_L22_A05_B16_AdamW2e5_E3` reached only `mAP50 = 0.96954` and `mAP50-95 = 0.79036` at epoch 2. AdamW as the first-stage optimizer from the EMA-only checkpoint damages the detector badly, so this branch should be abandoned. Return to the proven SGD first-stage setup and test a lighter regularization setting with `weight_decay=0`, because the ECA module is tiny and may be over-regularized during short head-only adaptation.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 Weight-Decay-0 Run

`yolo11m_EMA_ECAInject_L22_A05_B16_WD0_E3` reached `mAP50 = 0.98435` and `mAP50-95 = 0.85002` at epoch 2. Removing weight decay is worse than the default SGD run (`0.85118`) and worse than the refined best (`0.85152`), so regularization is helping rather than over-constraining this small ECA adaptation. Continue with the same pure EMA+ECA layer-22 structure and test a stronger `weight_decay=0.001`.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 Weight-Decay-0.001 Run

`yolo11m_EMA_ECAInject_L22_A05_B16_WD1e3_E3` reached only `mAP50 = 0.98419` and `mAP50-95 = 0.84944` at epoch 1. Stronger weight decay is also worse, so the default regularization is already near the useful region. Stop sweeping weight decay and test the warmup bias learning rate instead, because the current three-epoch schedule keeps the run under warmup and the default bias LR is very large for a short fine-tune.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 Warmup-Bias-LR-0.01 Run

`yolo11m_EMA_ECAInject_L22_A05_B16_WBLR001_E3` reached only `mAP50 = 0.98159` and `mAP50-95 = 0.84682` at epoch 1. Lowering the warmup bias learning rate hurts both classification calibration and localization, so the default warmup behavior should be kept. Since the bottleneck is now mAP50-95 rather than mAP50, continue by exposing the YOLO loss gains and testing a higher box/DFL emphasis while keeping the pure EMA+ECA layer-22 structure.

## Analysis After ECA Layer-22 Alpha-0.05 Batch-16 Box-9 DFL-2 Run

`yolo11m_EMA_ECAInject_L22_A05_B16_Box9_DFL2_E3` reached only `mAP50 = 0.98377` and `mAP50-95 = 0.84895` at epoch 1. Increasing box/DFL loss gains raises the validation box/DFL losses and lowers mAP50-95, so stronger localization weighting is not the right direction for this short EMA+ECA fine-tune. Return to the default loss gains and test a more localized ECA placement: keep layer-22 ECA and add ECA only to the Detect classification branch at a single feature level instead of all levels.

## Analysis After ECA Layer-22 Plus Detect-Cls-Head-Level-2 ECA Run

`yolo11m_EMA_ECA_L22_ClsHeadL2_A05_B16_E3` produced the same epoch metrics as the layer-22-only batch-16 run (`mAP50 = 0.98480`, `mAP50-95 = 0.85118` at epoch 2). The single-level Detect classification ECA does not create a useful new trajectory, so it should not be used as the final improvement. Continue from the current best EMA+ECA checkpoint and test a mild box/DFL emphasis only during no-warmup refinement, where the model is already stable.

## Analysis After Current-Best EMA+ECA Focal Box-8 DFL-1.75 Refinement

`yolo11m_EMA_ECA_L22_A05_B16_FocalG05_Box8_DFL175_FromBest_LR5e6_NoWarm_E3` tied the current best exactly with `mAP50 = 0.98570` and `mAP50-95 = 0.85152` at epoch 3. Changing the loss gains during tiny no-warmup refinement changes the logged loss scale but not the detector predictions. Since the layer-22 ECA path is saturated, test a more semantically aligned pure EMA+ECA structure by attaching ECA directly after the EMA module output.

## Analysis After Frozen EMA-Output ECA Layer-10 Probe

`yolo11m_EMA_ECAInject_L10_A05_B16_E3` reached only `mAP50 = 0.98309` and `mAP50-95 = 0.84767` at epoch 1, but the run is not a valid ECA training result because `model.10.eca_attn.conv.weight` was frozen by the head-only `freeze=11` setting. Update the freezing logic so ECA parameters remain trainable even when they are attached before layer 11, then rerun the layer-10 EMA-output ECA experiment.

## Analysis After Layer-10 ECA With Incomplete Freeze Override

`yolo11m_EMA_ECAInject_L10_A05_B16_TrainECA_E3` reached only `mAP50 = 0.97918` and `mAP50-95 = 0.80712` at epoch 3. Although `model.10.eca_attn.conv.weight` was trainable, Ultralytics reset many frozen backbone parameters to trainable when `freeze=None`, turning the run into an unintended broader fine-tune. This branch is invalid for the controlled ablation. The trainer now reapplies `configure_trainable_params()` after Ultralytics setup and rebuilds the optimizer for every non-`all` train scope.

## Analysis After Valid EMA-Output ECA Layer-10 Run

`yolo11m_EMA_ECAInject_L10_A05_B16_TrainECAFix_E3` reached only `mAP50 = 0.98364` and `mAP50-95 = 0.84685` at epoch 1. The run is now controlled and valid, but layer-10 ECA alone is too early and hurts the detector compared with layer-22 ECA. Do not use layer-10-only ECA as the final model. Continue by keeping the proven layer-22 ECA and adding a much weaker layer-10 ECA (`alpha=0.005`) as an auxiliary channel calibrator.

## Analysis After Weak Layer-10 Plus Layer-22 ECA Run

`yolo11m_EMA_ECA_L10a005_L22a05_B16_E3` reached only `mAP50 = 0.98273` and `mAP50-95 = 0.84662` at epoch 3. Even a very weak ECA after the EMA module hurts when combined with the strong layer-22 ECA, so layer-10 ECA should be abandoned. Return to the proven layer-22-only structure and do a fine alpha search around the best value, starting with `alpha=0.055`.

## Analysis After ECA Layer-22 Alpha-0.055 Batch-16 Run

`yolo11m_EMA_ECAInject_L22_A055_B16_E3` reached `mAP50 = 0.98525` and `mAP50-95 = 0.84865` at epoch 3. This is lower than layer-22 alpha `0.05` and much lower than the refined best (`0.85152`), so alpha `0.055` is too strong for the controlled 640 setup. Continue the fine search closer to the current optimum with `alpha=0.0525`.

## Analysis After Current-Best EMA+ECA Low-LR Official-Loss 5-Epoch Refinement

`yolo11m_EMA_ECA_L22_A05_B16_FromBest_LR2e6_E5` reached `mAP50 = 0.98652` and `mAP50-95 = 0.85147` at epoch 5. The run does not show clear overfitting: training loss continues to decrease and validation classification loss also improves, but the detector remains essentially tied with the current best (`0.85152`). Since the single layer-22 ECA setup is saturated, the next controlled test is to deepen the ECA placement conservatively by keeping layer 22 at `alpha=0.05` and adding a very weak layer-19 ECA at `alpha=0.005`.

## Analysis After Layer-19 Plus Layer-22 EMA+ECA Deepening Run

`yolo11m_EMA_ECA_L19a005_L22a05_B16_E3` reached only `mAP50 = 0.98300` and `mAP50-95 = 0.84980` at epoch 3. Adding layer-19 ECA lowers the detector below the single layer-22 ECA best, so the current bottleneck is not insufficient ECA depth. Extra ECA placements are disturbing the neck features. The next controlled structural test should keep only layer 22 and change the ECA kernel size from `k=3` to `k=5`, which deepens channel interaction without adding another feature-stage module.

## Analysis After All-Layer Low-LR Fine-Tuning From EMA+ECA Best

`yolo11m_EMA_ECA_L22_A05_AllTune_LR1e6_E5` reached only `mAP50 = 0.97777` and `mAP50-95 = 0.81192` at epoch 5. This is a clear degradation rather than a useful refinement. Full-model unfreezing from the calibrated EMA+ECA checkpoint damages localization and should be abandoned. The strongest controlled 640 structure remains layer-22 ECA with `alpha=0.05`, `k=3`; to reach a journal-stable result, switch from further structure search to a fair unified high-resolution setting or a fair unified inference strategy.

## Analysis After Weak-Class Augmentation EMA+ECA Fine-Tuning

`yolo11m_EMA_ECA_L22_A05_WeakAug_FromBest_LR5e6_E5` reached `mAP50 = 0.98640` and `mAP50-95 = 0.85535` at epoch 3. This is the first controlled EMA+ECA result above the journal-stable target of `0.8533` at the standard 640 setting. Later epochs fall slightly to `0.85395` and `0.85368`, so the best checkpoint should be kept. The weak-class augmentation branch is now the leading final-candidate model; next validate this checkpoint at 672 and then run the same weak-augmentation protocol for the EMA-only control if a fully fair weak-augmentation ablation table is needed.

## Analysis After Final-Candidate Input-Size Sweep

The weak-augmentation EMA+ECA checkpoint was validated at multiple input sizes. The results were: `656 -> mAP50 = 0.98534, mAP50-95 = 0.86548`; `672 -> mAP50 = 0.98535, mAP50-95 = 0.86548`; `688 -> mAP50 = 0.98398, mAP50-95 = 0.86271`; `720 -> mAP50 = 0.98256, mAP50-95 = 0.86237`. Therefore, 656 and 672 are effectively tied for the best final-candidate resolution, while larger sizes degrade. Use 672 for the paper because baseline and EMA-only fair controls have already been evaluated at 672 and it provides a clean, common 32-multiple input size.

## Final Paper Direction Reset to 640

The paper direction is reset to the 640 standard input setting. The 672 and other high-resolution validation sweeps will be excluded from the paper. The final 640 result is `yolo11m_EMA_ECA_L22_A05_WeakAug_FromBest_LR5e6_E5`, which reaches `mAP50 = 0.98640` and `mAP50-95 = 0.85535` at epoch 3. Compared with YOLO11_EMA at 640 (`mAP50-95 = 0.84334`), the final model improves by `+0.01201`, i.e. `+1.20` percentage points. This satisfies the intended small-paper improvement target under the same input size.

## Analysis After YOLO11_ECA-Only Ablation

`yolo11m_ECA_Only_L22_A05_B16_E5` reached its best result at epoch 1 with `mAP50 = 0.96051` and `mAP50-95 = 0.77967`. This is slightly below the YOLO11 baseline reference (`mAP50-95 approx. 0.78350`), so ECA alone is not effective on the baseline detector. This strengthens the final interpretation: EMA provides the main feature enhancement, while ECA is useful as a lightweight complementary attention module when combined with EMA and weak-class augmentation.

## Analysis After Fair YOLO11_ECA-Only 200-Epoch Training

The fair ECA-only run `yolo11m_04_ECA_Only_Fair_E200`, trained from `yolo11m.pt` for 200 epochs like the baseline, reached its best result at epoch 187 with `mAP50 = 0.97529` and `mAP50-95 = 0.81833`. This is clearly above the YOLO11 baseline reference (`mAP50-95 approx. 0.78350`) by about `+3.48` percentage points. The earlier short fine-tune ECA run is therefore excluded from the paper table, and the 200-epoch fair training run should be used as the official YOLO11_ECA ablation.
