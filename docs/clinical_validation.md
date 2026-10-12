# Clinical Validation Methodology & Trial Protocol

**Project**: AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India  
**Document Version**: 2.6  
**Clinical Status**: **Protocol Formulated — Human Real-World Clinical Trial: PENDING**

---

## 1. Study Objective
To evaluate the clinical efficacy, safety, and diagnostic triage concordance of the **AI DR Screen** telemedicine system in detecting referable diabetic retinopathy (ICDR Grade ≥ 2: Moderate NPDR, Severe NPDR, or Proliferative DR) across rural Primary Health Centers (PHCs) in India.

---

## 2. Study Design & Reference Standard
- **Design**: Prospective, multi-center, cross-sectional, paired observational validation study.
- **Target Population**: Adults aged ≥18 years with diagnosed Type 1 or Type 2 diabetes attending rural screening camps.
- **Reference Standard (Ground Truth)**:
  - Consensus adjudication by two independent certified retina specialists reviewing 7-field or widefield macular and disc fundus photographs.
  - In cases of disagreement between the two specialists, a third senior retinal consultant adjudicates the final reference grade.
- **Safety Pre-Condition**:
  - Prior automated Image Quality Assessment (IQA) gatekeeper (MobileNetV3-Small).
  - Images identified as unusable or poor quality are immediately flagged for on-site operator recapture, preventing diagnostic error from ungradable media.

---

## 3. Study Endpoints

### Primary Endpoints
1. **Diagnostic Sensitivity for Referable DR**:
   - Target: > 90.0%
   - Actual test set performance on held-out cohort (N = 284):
     - Standard Argmax Threshold: **59.13%** (95% CI: 49.99% – 67.68%)
     - Calibrated Operating Threshold ($\tau = 0.495$): **90.43%** (95% CI: 83.56% – 95.18%)
2. **Diagnostic Specificity for Referable DR**:
   - Target: > 85.0%
   - Actual test set performance:
     - Standard Argmax Threshold: **92.31%** (95% CI: 87.29% – 95.45%)
     - Calibrated Operating Threshold: **79.88%** (95% CI: 73.11% – 85.50%)

### Secondary Endpoints
- Area Under the Receiver Operating Characteristic Curve (ROC-AUC): **0.9131** (EfficientNet-B0) / **0.9203** (ResNet-50)
- Precision-Recall Area Under Curve (PR-AUC): **0.8592**
- Expected Calibration Error (ECE): **0.1776**
- Brier Score: **0.1515**
- Image Quality Rejection Rate: **18.0% – 38.4%** (dependent on strict confidence threshold)
- Automated AI Pipeline Throughput: **456.7 ms / image** (3,941 patients/hour per compute worker)

---

## 4. Multi-Modal Explainability & Human-in-the-Loop Protocol

When an image is classified as referable DR or exhibits borderline confidence (<75%), the system automatically compiles an explainability payload for the tele-ophthalmologist:
1. **Grad-CAM Saliency Map**: Highlights the convolutional feature activation regions guiding the CNN.
2. **U-Net Retinal Lesion Masks**: Pixel-level morphological segmentations of microaneurysms, hemorrhages, and hard exudates.
3. **SHAP Feature Attribution**: Identifies positive and negative pixel contributions.
4. **Deterministic Referral Recommendation**: Follow-up timeframe (e.g. urgent 1-week referral for PDR, routine 12-month review for No DR).

---

## 5. Ethical Considerations & Regulatory Status
- **Institutional Review Board (IRB)**: Approval by the relevant State Health Ethics Committee is mandatory prior to clinical deployment.
- **Informed Consent**: Written or biometric verified informed consent under Ayushman Bharat Digital Mission (ABDM) / DISHA regulations.
- **Scientific Integrity Statement**:
  - The AI screening results are not an autonomous medical diagnosis.
  - A formal real-world clinical trial involving human ophthalmologists at active rural PHCs is currently **PENDING** pending prospective institutional IRB funding and rollout.
