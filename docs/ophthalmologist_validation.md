# Human-in-the-Loop Ophthalmologist Validation Protocol

**Project**: Explainable AI for Diabetic Retinopathy Screening in Rural India (Retina_Rovers / AI DR Screen)  
**Target Benchmark**: Review time < 30 seconds per bilateral fundus encounter  
**Status**: `PENDING HUMAN STUDY` (No clinical data fabrication)  

---

## 1. Executive Summary & Regulatory Context

In rural tele-ophthalmology screening across primary health centres (PHCs) in India, the specialist ophthalmologist represents the critical throughput bottleneck. The AI DR Screen platform functions as a **triaging and assistive decision-support system** designed to compress expert review duration while preserving high diagnostic accuracy and referral fidelity.

```
+---------------------------------------------------------------------------------------------------+
|                                 HUMAN-IN-THE-LOOP SCREENING COCKPIT                               |
+------------------------------------+--------------------------------------------------------------+
| [1. Original Fundus Scan]          | [2. AI Decision Support Panel]                               |
| High-Resolution 45° Posterior Pole | - DR Grade: Grade 2 (Moderate NPDR)                          |
| Macula & Optic Disc Centered       | - Calibrated Confidence: 89.4%                               |
|                                    | - Risk Category: HIGH REFERRAL RISK                          |
|                                    | - Referral Engine: Refer to District Eye Hospital (<= 30 d)  |
+------------------------------------+--------------------------------------------------------------+
| [3. Explainability Overlays]       | [4. Structured Clinical Feedback Form]                      |
| - Grad-CAM Saliency Heatmap        | - Doctor Agrees with Grade? [Yes / Adjusted Grade: _____]    |
| - Lesion Mask (MA, HE, EX)         | - Clinical Usefulness Score: [ 1 | 2 | 3 | 4 | 5 ] Stars     |
| - Structure Markers (OD & Fovea)   | - Specialist Review Confidence: [ 1 | 2 | 3 | 4 | 5 ]        |
|                                    | - Automated Encounter Timer: [ 18.4s Elapsed ]               |
+------------------------------------+--------------------------------------------------------------+
```

---

## 2. Reviewer Clinical Workstation Packet

For each encounter, the reviewing ophthalmologist receives:

1. **Original High-Resolution Fundus Image**:
   - Clean color fundus photograph with contrast normalization.
2. **AI DR Grade Prediction**:
   - Predicted International Clinical Diabetic Retinopathy (ICDR) severity scale (Grade 0–4).
3. **Calibrated Confidence Score**:
   - Probability score calibrated via Platt scaling / temperature scaling with reliability metrics.
4. **Grad-CAM Saliency Map**:
   - Class-activation heatmap highlighting specific retinal regions that drove model inference.
5. **Retinal Lesion Segmentation Overlay**:
   - Color-coded morphological bounding masks for microaneurysms (red dots), hemorrhages (crimson patches), hard exudates (yellow), and cotton-wool spots (white).
6. **Risk Level & Clinical Recommendation**:
   - Triaging urgency category (Routine 12-month rescreen vs Urgent 2-week retina specialist consult).

---

## 3. Measured Clinical Metrics & Endpoints

| Parameter | Measurement Methodology | Target Specification |
|---|---|---|
| **Average Review Time** | Millisecond-precision browser event timer tracking review session | < 30.0 seconds |
| **Median Review Time** | 50th percentile review duration across reader cohort | < 25.0 seconds |
| **Maximum Review Time** | 95th / 99th percentile upper bound for complex equivocal cases | < 60.0 seconds |
| **Agreement with Reference** | Multi-rater agreement against consensus adjudication panel (Cohen's / Fleiss' $\kappa$) | $\kappa > 0.85$ (Near-perfect agreement) |
| **Clinical Usefulness Score** | 5-point Likert survey ("AI explanations accelerated my review"): 1=Strongly Disagree, 5=Strongly Agree | Mean Score $\ge 4.2 / 5.0$ |
| **Reviewer Confidence** | 5-point Likert rating of diagnostic certainty with AI assistance | Mean Score $\ge 4.5 / 5.0$ |

---

## 4. Formal Protocol Evaluation Status

In accordance with strict scientific integrity guidelines:

```text
Ophthalmologist Validation Study Status:
PENDING HUMAN STUDY
```

> [!IMPORTANT]
> **Scientific Integrity & Ethics Disclosure**:  
> No simulated or synthetic human review times have been fabricated to falsely claim compliance. The software cockpits, telemetry timers, and data capture schemas have been implemented and verified in the codebase. Actual reader metrics will be recorded during the prospective multi-reader multi-case (MRMC) trial under Institutional Review Board (IRB) ethics approval.
