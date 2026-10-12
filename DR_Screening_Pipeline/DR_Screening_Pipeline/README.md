# Explainable AI for Diabetic Retinopathy Screening in Rural India

A MATLAB/Simulink prototype for automated, explainable Diabetic Retinopathy (DR)
screening designed for deployment through primary health centres (PHCs) using
portable, non-mydriatic fundus cameras and low-bandwidth telemedicine links.

This is a **working software prototype**, not a finished clinical product. It is
built so every stage runs standalone (you can test/tune each module on its own)
and end-to-end (`main_pipeline.m` chains them). Where a full deep-learning model
would normally require a large labeled dataset (IDRiD, Messidor-2, APTOS 2019,
EyePACS) that isn't bundled here, the code is written against those datasets'
known structure so it runs unmodified once you point it at real data — see
`docs/DATASETS.md`.

---

## 1. Why this design

| Real-world failure mode of existing DR-AI | How this pipeline addresses it |
|---|---|
| Black-box CNN, no clinician trust | Grad-CAM + lesion-level evidence + calibrated confidence, packaged as a report reviewable in <30s |
| Breaks on poor-quality field images | Dedicated quality-gate stage rejects/enhances *before* any grading happens |
| Single-technique models miss small lesions (microaneurysms) | Hybrid architecture: CNN (global pattern) + classical lesion detectors (local, sub-pixel) fused at grading time |
| No sense of how many reviewers/bandwidth a district needs | Simulink/SimEvents-style workflow model + analytical queueing calculator sized to 100,000+ patients/year |
| Grading not clinically standardized | Output strictly follows the International Clinical DR (ICDR) severity scale, Levels 0–4 |

## 2. Pipeline architecture

```
 Fundus image (portable camera, field conditions)
        │
        ▼
 ┌───────────────────────────┐
 │ 1. Quality Assessment      │  assessImageQuality.m
 │    + Adaptive Enhancement  │  enhanceImage.m
 └───────────┬───────────────┘
             │ (reject → recapture message)
             ▼
 ┌───────────────────────────┐
 │ 2. Structure Segmentation  │  detectOpticDiscFovea.m
 │    - Optic disc / fovea    │  segmentVessels.m
 │    - Vessels               │  detectMicroaneurysms.m
 │    - Microaneurysms        │  segmentExudates.m
 │    - Exudates              │  detectHemorrhages.m
 │    - Hemorrhages           │  detectNeovascularization.m
 │    - Neovascularization    │
 └───────────┬───────────────┘
             ▼
 ┌───────────────────────────┐
 │ 3. DR Severity Grading     │  buildDRClassifier.m
 │    CNN (global) ⊕ lesion   │  trainDRModel.m
 │    rule-score (local),     │  gradeDRSeverity.m
 │    fused → ICDR grade 0-4  │
 └───────────┬───────────────┘
             ▼
 ┌───────────────────────────┐
 │ 4. Explainability          │  generateGradCAM.m
 │    Grad-CAM + lesion map + │  generateClinicalReport.m
 │    confidence + PDF report │
 └───────────┬───────────────┘
             ▼
 ┌───────────────────────────┐
 │ 5. Performance Validation  │  evaluatePerformance.m
 └───────────────────────────┘

 (separately) Telemedicine capacity planning:
 simulink/buildScreeningWorkflowModel.m   → Simulink block diagram
 simulink/capacityPlanningAnalysis.m      → analytical M/M/c sizing
```

## 3. Required toolboxes

- Image Processing Toolbox
- Computer Vision Toolbox
- Deep Learning Toolbox (with a pretrained network such as ResNet50/EfficientNet-b0 installed via Add-On Explorer)
- Statistics and Machine Learning Toolbox
- Simulink (+ SimEvents recommended, not required — see §6)
- Medical Imaging Toolbox is *not strictly required*; all image ops here use
  Image Processing Toolbox primitives so the pipeline still runs without it.

MATLAB R2021a or later (for built-in `gradCAM`).

## 4. Quick start

```matlab
addpath(genpath(pwd));

% Single image, full pipeline, generates report + prints grade
result = main_pipeline('data/sample_fundus.jpg');

% Batch evaluation against a labeled dataset (e.g. IDRiD test set)
metrics = evaluatePerformance('data/IDRiD_test/', 'data/IDRiD_test/labels.csv');
```

## 5. Folder guide

```
src/preprocessing/     Quality gate + enhancement
src/segmentation/      Anatomical + lesion segmentation
src/classification/    CNN+rule fusion grading (ICDR 0-4)
src/explainability/    Grad-CAM, clinical PDF report
src/evaluation/        Sensitivity/specificity/AUC vs. benchmarks
simulink/              Telemedicine throughput & capacity model
docs/                  Dataset notes, clinical mapping, validation plan
main_pipeline.m        End-to-end orchestration for one image
```

## 6. Simulink workflow simulation

Two complementary artifacts are provided (see `docs/SIMULINK_NOTES.md` for why):

1. `capacityPlanningAnalysis.m` — a closed-form/analytical queueing model
   (M/M/1 and M/M/c) you can run immediately in plain MATLAB, no Simulink
   license needed, to size reviewer headcount, camera count, and bandwidth
   for a target annual patient volume and referral rate.
2. `buildScreeningWorkflowModel.m` — programmatically builds a Simulink
   `.slx` model (acquisition source → bandwidth-limited transfer →
   AI-processing server → human-review queue → outcome sink) using the
   Simulink API, so it can be opened, edited, and extended with SimEvents
   blocks directly in the MATLAB GUI. Run it once inside MATLAB to produce
   `ScreeningWorkflow.slx`; it cannot be pre-built outside MATLAB.

## 7. Validation approach

`evaluatePerformance.m` computes sensitivity/specificity/AUC for **referable
DR (ICDR ≥ 2)** on any labeled folder you point it at, and prints a
side-by-side table against illustrative published benchmarks (IDx-DR pivotal
trial, Google ARDA, EyePACS baseline CNN — see `docs/VALIDATION_PLAN.md` for
sourcing and caveats: these are reference points for target-setting, not a
substitute for your own IRB-approved clinical validation study).

## 8. Honest limitations (read before deploying anything)

- No labeled training data ships with this repo — `trainDRModel.m` must be
  pointed at a real, ethically-sourced, de-identified dataset before the CNN
  branch is meaningful. Until trained, the pipeline still runs end-to-end
  using the lesion-based rule score alone (degraded but functional).
- Neovascularization detection is the weakest link scientifically — true NV
  detection needs FA imaging or a dedicated, well-validated model; the
  heuristic here (vessel-caliber irregularity near the disc) is a
  **screening flag for manual review only**, not a diagnostic claim.
- Sub-pixel microaneurysm detection quality depends heavily on camera
  resolution/compression; `detectMicroaneurysms.m` documents the assumptions.
- This is a decision-support / triage tool for a human-in-the-loop workflow,
  not an autonomous diagnostic device. Any real deployment needs regulatory
  clearance (in India: CDSCO medical device rules) and a clinical validation
  study on the actual target population and camera hardware.
