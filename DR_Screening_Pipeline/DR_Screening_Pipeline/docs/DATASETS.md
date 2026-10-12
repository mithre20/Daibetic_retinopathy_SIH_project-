# Datasets to train/validate against

This repo ships no patient images (none should ever be committed to a code
repo without a data-sharing agreement and de-identification review). Point
`trainDRModel.m` / `evaluatePerformance.m` at one of these public,
research-license datasets instead:

| Dataset | Images | Labels | Notes |
|---|---|---|---|
| **IDRiD** (Indian Diabetic Retinopathy Image Dataset) | 516 | ICDR grade + pixel-level lesion masks (MA, HE, EX, SE) | Most directly relevant — Indian population, and the only major public set with lesion-level segmentation ground truth, useful for validating `detectMicroaneurysms.m` etc. directly, not just the final grade. |
| **Messidor-2** | 1,748 | ICDR grade + referable DR | Widely used as an external validation set in FDA/CE submissions. |
| **APTOS 2019 (Kaggle)** | ~3,662 labeled | ICDR grade | Larger, more label noise; good for CNN pretraining before fine-tuning on IDRiD. |
| **EyePACS (Kaggle DR Detection)** | ~88,000 | ICDR grade | Largest public set; heavily class-imbalanced (~73% Grade 0) — this is what `trainDRModel.m`'s inverse-frequency class weighting is calibrated against. |

## Recommended training recipe
1. Pretrain / fine-tune the CNN branch on EyePACS + APTOS (volume).
2. Fine-tune again on IDRiD + Messidor-2 (population/label-quality match to
   the deployment target).
3. Hold out IDRiD's own test split (or a fresh local pilot-site sample) for
   the final `evaluatePerformance.m` numbers reported to a clinical partner
   — never report metrics on data the model saw during any tuning step.
4. Once a pilot-site camera model is chosen, capture a small (~200-500
   image) locally-labeled set on *that exact camera* under real field
   conditions and re-validate `assessImageQuality.m`'s thresholds and the
   final sensitivity/specificity — public-dataset images are typically
   higher quality than what a portable field camera produces.

## CSV label format expected by this pipeline
```
image,grade
IDRiD_001.jpg,2
IDRiD_002.jpg,0
...
```
`grade` in {0,1,2,3,4} per the ICDR scale.
