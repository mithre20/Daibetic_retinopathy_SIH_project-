# Retinal Diabetic Retinopathy Risk & Referral Decision Engine

**AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India**

---

## 1. Purpose

The **Risk / Referral Decision Engine** is an auditable, deterministic, rule-based clinical triage layer for the AI DR Screen pipeline. Its primary objective is to convert validated Diabetic Retinopathy (DR) severity classifications (produced by the primary EfficientNet-B0 model) into clear, actionable clinical recommendations for healthcare workers and community ophthalmic assistants in rural Indian vision camps.

### Why Rule-Based and Not Another Neural Network?
1. **Clinical Transparency & Auditability**: Deep neural networks are statistical feature approximators and function as black boxes. In medical decision-making—particularly referral prioritization—decisions must follow explicit, inspectable logic that can be reviewed, verified, and audited by healthcare authorities (e.g., ICMR, AIOS, WHO).
2. **Deterministic Reproducibility**: Given a validated DR grade and confidence score, the decision engine will always produce the exact same clinical triage advice without probabilistic drift or hallucination.
3. **Zero Computational Overhead**: Triage decision execution requires less than 0.1 milliseconds on standard low-power edge laptops and mobile devices deployed in off-grid rural primary health centers (PHCs).
4. **Separation of Concerns**: Classification models detect retinal lesions and grade disease severity; clinical decision rules establish care pathways and referral urgency based on healthcare capacity and protocol.

---

## 2. System Architecture

The complete AI DR Screen workflow executes sequentially as follows:

```text
                         RETINAL FUNDUS IMAGE
                                  │
                                  ▼
                      ┌──────────────────────┐
                      │ IMAGE QUALITY MODEL  │
                      │   MobileNetV3-Small  │
                      └──────────┬───────────┘
                                 │
                     ┌───────────┴───────────┐
                     │                       │
                  ACCEPTED               REJECTED /
               Good Quality               UNUSABLE
            (Confidence >= 75%)              │
                     │                       ▼
                     │              SCREENING BLOCKED
                     │             (dr_prediction = null,
                     │              referral = null)
                     │                       │
                     ▼                       ▼
              EfficientNet-B0        Capture New Image
                     │
                     ▼
             Validated DR Grade
              (0, 1, 2, 3, 4)
                     │
                     ▼
          ┌─────────────────────┐
          │   REFERRAL ENGINE   │
          │     Rule-Based      │
          └──────────┬──────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
      Low Risk    Moderate    High Risk
         │          Risk         │
         ▼           ▼           ▼
      Routine    Eye Exam     Priority
     Follow-up  Recommended   Referral
```

The explainability modalities operate concurrently to assist human clinical verification:
```text
                       EfficientNet-B0
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
            Grad-CAM         SHAP          U-Net
         Heatmap Focus     Positive/      Retinal
           (Anatomy)       Negative     Lesion Maps
                           Features
```

---

## 3. Screening Referral Rules

The prototype screening rules map ICDR 5-class severity grades into standardized risk levels and actionable clinical steps:

| DR Grade | Classification Label | Risk Category | Referral Priority | Clinical Action & Recommendation |
| :---: | :---: | :---: | :---: | :--- |
| **0** | No DR | **Low Risk** | Routine | **Routine follow-up**<br>Annual routine retinal screening recommended as per diabetes standard of care. |
| **1** | Mild DR | **Moderate Risk** | Recommended | **Eye examination recommended**<br>Comprehensive eye examination by an optometrist or ophthalmologist within 6 to 12 months. |
| **2** | Moderate DR | **Moderate Risk** | Recommended | **Eye examination recommended**<br>Comprehensive ophthalmic examination within 3 to 6 months to evaluate progression and macular status. |
| **3** | Severe DR | **High Risk** | Priority | **Priority ophthalmologist referral**<br>Prompt referral to a retinal specialist/ophthalmologist within 2 to 4 weeks for clinical staging and intervention planning. |
| **4** | Proliferative DR | **High Risk** | Priority | **Priority ophthalmologist referral**<br>Urgent referral to a vitreoretinal specialist for immediate intervention (e.g., panretinal photocoagulation or anti-VEGF therapy). |

Explicit rule implementation:
```python
if dr_grade == 0:
    risk_level = "Low Risk"
    referral_priority = "Routine"
    recommendation = "Routine follow-up"
elif dr_grade in [1, 2]:
    risk_level = "Moderate Risk"
    referral_priority = "Recommended"
    recommendation = "Eye examination recommended"
elif dr_grade in [3, 4]:
    risk_level = "High Risk"
    referral_priority = "Priority"
    recommendation = "Priority ophthalmologist referral"
```

---

## 4. Confidence-Aware Safety Logic

Model uncertainty must be made transparent to the healthcare operator. The referral engine uses a configurable engineering threshold (`minimum_dr_confidence = 0.50`) to flag borderline or ambiguous predictions.

### Rule 4.1: Low-Confidence Non-Severe Predictions
If `dr_confidence < minimum_dr_confidence` for Grade 0, 1, or 2:
- `decision_status = "LOW_CONFIDENCE"`
- `recommendation = "Professional review recommended"`
- Explanation indicates that while the model tentatively predicted low/moderate severity, low confidence necessitates direct human clinician inspection.

### Rule 4.2: High-Severity Preservation Safeguard
Under no circumstances is a Grade 3 (Severe DR) or Grade 4 (Proliferative DR) prediction ever downgraded to low risk due to low confidence.
- Even if confidence is below 50% (e.g. 42%), the engine enforces:
  - `risk_level = "High Risk"`
  - `referral_priority = "Priority"`
  - `recommendation = "Priority ophthalmologist referral"`
- Note appended: *"Prediction confidence is low; professional review is especially important. High-severity screening signal is preserved and not downgraded."*

---

## 5. Strict Image Quality Gate Precedence

A fundamental clinical safety constraint is that substandard images must never produce a diagnostic recommendation.

1. If Image Quality Assessment (IQA) returns `Poor Quality` or `Unusable`:
   - EfficientNet-B0 forward pass is skipped completely.
   - `dr_prediction = null`
   - `risk_level = null`
   - `referral_priority = null`
   - `recommendation = "Please capture another retinal image."`
2. If IQA returns `Good Quality` but confidence is below the safety threshold ($< 75.0\%$):
   - Screening is blocked identically to prevent unconfident quality approvals.
3. This prevents the dangerous clinical failure mode where an unreadable, underexposed, or blurred fundus photograph is mistakenly classified as "No DR" (Grade 0), falsely reassuring a patient with sight-threatening retinopathy.

---

## 6. Transparency & Explainability

Every decision generated by the engine includes a clear, human-readable rationale:

```text
Why this recommendation?

Detected DR grade: Moderate DR (Grade 2)
Decision rule: DR Grade 1–2 → Moderate Risk
Recommendation: Eye examination recommended.
```

When coupled with Grad-CAM and SHAP visual explanations, the operator can see:
1. Which specific retinal structures (microaneurysms, hemorrhages, exudates, optic disc) triggered the classification.
2. The exact rule translating that finding into a referral priority.

### Optional Medical Report Context
Clinical lab parameters (Blood Glucose, HbA1c, Blood Pressure) extracted from medical records are displayed strictly as **supplementary clinical context**:
- The system **never** invents an unvalidated mathematical composite score (e.g. $DR + HbA1c + BP = 87\%$).
- Extracted values are presented alongside the DR grade for review by the consulting doctor, preserving deterministic reproducibility.

---

## 7. Limitations & Clinical Governance

1. **Screening Prototype**: This software is an assistive triage prototype intended for rural screening camps. It is not an autonomous diagnostic medical device.
2. **Clinical Protocol Variation**: Actual referral windows and clinical actions must be adapted to local ophthalmic guidelines (e.g., All India Ophthalmological Society guidelines, state health mission protocols).
3. **Mandatory Disclaimer**: Every user interface card, printed report, and API response carries the mandatory disclaimer:
   > *"AI screening results are not a medical diagnosis. Professional clinical evaluation by an ophthalmologist or optometrist is recommended."*
