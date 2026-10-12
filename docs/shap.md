# SHAP Feature-Level Explainability for Diabetic Retinopathy Screening

## AI DR Screen — Explainable AI for Diabetic Retinopathy in Rural India

---

## 1. What is SHAP?

**SHAP (SHapley Additive exPlanations)** is an axiomatic, game-theoretic framework for explaining machine learning models. Derived from cooperative game theory (Lloyd Shapley, 1953), Shapley values allocate total model output among input features fairly according to their marginal contributions across all possible feature subsets.

When applied to deep convolutional neural networks (CNNs), SHAP computes pixel- and region-level attribution scores that explain how each spatial component of a retinal photograph influenced the network's classification logits.

Unlike purely heuristic or gradient-saliency techniques, SHAP possesses foundational theoretical guarantees:
- **Local Accuracy (Efficiency)**: The sum of feature attributions equals the difference between the model output for the input and the expected model output across reference baselines.
- **Missingness**: Features absent from the coalition receive zero attribution.
- **Consistency (Monotonicity)**: If a model changes so that the marginal contribution of a feature increases, its attribution cannot decrease.

---

## 2. Why SHAP is Used in AI DR Screen

In high-stakes medical screening—especially in resource-constrained primary health centres (PHCs) across rural India—clinicians cannot trust "black-box" predictions without interpretable rationale:

1. **Directional Attribution**: While attention heatmaps like Grad-CAM show *where* a CNN focused, they cannot distinguish whether a region *supported* or *opposed* a particular disease grade. SHAP explicitly bifurcates evidence into **positive** (supporting the predicted DR severity) and **negative** (opposing it or pulling toward normal retina).
2. **Pathology Validation**: By overlaying SHAP attributions onto retinal fundus photographs, ophthalmologists and screening technicians can verify whether the model is recognizing genuine pathological anomalies (such as microaneurysms, hemorrhages, and lipid exudates) rather than irrelevant camera artifacts, dust rings, or lighting gradients.
3. **Auditing False Positives & Negatives**: In borderline or referable screening cases (ICDR Grade 2+), SHAP provides transparent insight into why a specific risk tier was triggered.

---

## 3. How SHAP Works with EfficientNet-B0

AI DR Screen uses a fine-tuned **EfficientNet-B0** architecture for 5-class Diabetic Retinopathy classification:
- Grade 0: No Diabetic Retinopathy
- Grade 1: Mild Diabetic Retinopathy
- Grade 2: Moderate Diabetic Retinopathy
- Grade 3: Severe Diabetic Retinopathy
- Grade 4: Proliferative Diabetic Retinopathy

### Implementation Pipeline:
```text
Retinal Fundus Image (H × W × 3)
              │
              ▼
   Inference Preprocessing
   • Resize: (224, 224)
   • Tensor conversion: [0, 1]
   • ImageNet Normalization:
     mean = [0.485, 0.456, 0.406]
     std  = [0.229, 0.224, 0.225]
              │
              ▼
    EfficientNet-B0 Forward Pass
    • Backbone features [B, 1280, 7, 7]
    • Classifier Head [1280 → 5]
    • Output: Class Probabilities & Top Prediction
              │
              ▼
    SHAP GradientExplainer (Expected Gradients)
    • Model: DRModelWrapper (eval mode)
    • Baseline: Dark-field retinal zero-tensor reference
    • Path Integrations: Monte-Carlo path samples (nsamples=30)
              │
              ▼
    Raw Attribution Tensor [1, 3, 224, 224, 5]
              │
              ▼
    Spatial Aggregation & Decomposition
    • Sum across color channels (C=3)
    • Positive Map (+): np.maximum(attr_2d, 0)
    • Negative Map (−): np.maximum(-attr_2d, 0)
    • Diverging Map [−1, 1]: attr_2d / abs_max
              │
              ▼
    Multi-Resolution Overlay
    • Bicubic upsampling to original fundus size
    • Scientific Diverging Colorbar Legend
```

---

## 4. Multi-Modal Method Comparison

AI DR Screen incorporates multiple complementary analytical methods. None of these methods replace one another:

| Method | Role / Architecture | Primary Purpose | Clinical Interpretation |
| :--- | :--- | :--- | :--- |
| **EfficientNet-B0** | Primary CNN (5-Class) | DR Classification & Grading | **WHAT** severity grade is predicted |
| **ResNet50** | Comparative CNN | Secondary validation classifier | Verification of grading consistency |
| **U-Net** | Dual-Decoder Segmentation | Retinal lesion localization | **WHERE** specific lesions (MA, HE, EX, SE) reside |
| **Grad-CAM** | Gradient-Weighted Activation | Coarse CNN feature attention | **WHERE** the CNN focused its final receptive field |
| **SHAP** | Expected Gradients | Feature attribution decomposition | **WHICH** regions contributed positively vs negatively |

---

## 5. SHAP vs Grad-CAM

| Feature | Grad-CAM | SHAP (Expected Gradients) |
| :--- | :--- | :--- |
| **Resolution** | Coarse (7 × 7 feature map upsampled) | High (pixel- and region-level attribution) |
| **Attribution Sign** | Non-negative only (ReLU removes negative gradients) | Signed (both positive and negative attributions) |
| **Mathematical Basis** | Heuristic gradient-feature weighting | Axiomatic cooperative game theory |
| **Computation Time** | Real-time (~0.1 seconds) | On-demand (~10–15 seconds on CPU) |
| **Clinical Question** | "Where did the CNN look?" | "What pushed the model toward or away from this grade?" |

---

## 6. SHAP vs U-Net Lesion Segmentation

It is critical **NOT** to confuse SHAP with U-Net lesion segmentation:
- **SHAP explains the classifier's internal reasoning.** A highlighted pixel indicates that the model used that region to increase or decrease class confidence. It does not certify that a biological lesion exists.
- **U-Net performs anatomical segmentation.** It specifically localizes and classifies microaneurysms, hemorrhages, hard exudates, and cotton-wool spots based on segmentation training masks.

When combined, AI DR Screen performs **Lesion-Aware Regional Attribution**:
Computing empirical SHAP values *inside* U-Net segmented masks allows researchers to audit whether EfficientNet-B0 correctly prioritizes clinically verified lesions over background noise.

---

## 7. Positive vs Negative Contributions

### Positive Contributions (Warm Colors: Red / Yellow)
- Image regions that **increased** the logit score for the target DR grade.
- In referable DR images (Grade 2–4), positive attributions typically center on clusters of dark intraretinal hemorrhages, microaneurysm patterns, or bright exudate rings.
- In normal images (Grade 0), positive attributions reflect clean, uniform background retinal parenchyma and intact macula-foveal architecture.

### Negative Contributions (Cool Colors: Blue / Cyan)
- Image regions that **decreased** the logit score for the target DR grade.
- These regions pull the model away from the target class or push it toward alternative DR classifications.
- Example: Clear, lesion-free peripheral zones may push against a Severe DR classification.

### Neutral / Zero Regions (Dark / Transparent)
- Retinal background and uniform tissue that exerted minimal influence on the decision boundary.

---

## 8. How to Run SHAP

### 1. Command Line Interface (CLI)
```bash
# Standard explanation targeting model prediction
python inference/predict_shap.py --image fundus_scan.jpg

# Targeting specific DR class (e.g., Grade 2: Moderate DR)
python inference/predict_shap.py --image fundus_scan.jpg --target-class 2

# Specifying custom output directory and Monte-Carlo samples
python inference/predict_shap.py --image fundus_scan.jpg --output results/shap/ --samples 30

# Including complementary U-Net lesion overlap analysis
python inference/predict_shap.py --image fundus_scan.jpg --with-unet
```

### 2. Multi-Modal Comparison Script
Generate a side-by-side comparison across Original Fundus, Grad-CAM, SHAP, and U-Net:
```bash
python analysis/compare_explanations.py --image fundus_scan.jpg --output results/explanations_comparison.png
```

### 3. Backend REST API
Endpoint: `POST /api/explain/shap`
```bash
# Using multipart form-data upload
curl -X POST "http://127.0.0.1:5000/api/explain/shap?target_class=2&nsamples=30" \
     -F "image=@fundus_scan.jpg"
```

---

## 9. Output Files and Data Structures

When executed, `predict_shap.py` generates the following structured artifacts:

```text
results/
└── shap/
    ├── original.png              # Preprocessed fundus image
    ├── shap_positive.png         # Warm positive attribution heatmap
    ├── shap_negative.png         # Cool negative attribution heatmap
    ├── shap_overlay.png          # Diverging attribution overlay with scientific legend
    ├── shap_result.json          # Machine-readable prediction, metrics, and summary
    └── shap_values.npy           # Raw numerical attribution array [1, 3, 224, 224, 5]
```

### `shap_result.json` Schema:
```json
{
  "prediction_class": 0,
  "prediction_label": "No Diabetic Retinopathy",
  "prediction_probability": 0.686,
  "target_class": 0,
  "target_label": "No Diabetic Retinopathy",
  "explanation_method": "SHAP",
  "explainer": "shap.GradientExplainer (Expected Gradients)",
  "model": "EfficientNet-B0",
  "input_dimensions": [1024, 1024],
  "probabilities": {
    "no_dr": 0.686,
    "mild": 0.0603,
    "moderate": 0.0875,
    "severe": 0.0943,
    "proliferative": 0.0719
  },
  "shap_statistics": {
    "min": -1.132,
    "max": 0.952,
    "mean": 0.000014,
    "sparsity": 0.925,
    "computation_time_seconds": 10.80,
    "nsamples": 30
  },
  "attribution_dimensions": [1, 3, 224, 224, 5],
  "disclaimer": "SHAP visualizations explain which image regions influenced the AI classifier's prediction..."
}
```

---

## 10. Computational Optimization & Limitations

- **On-Demand Execution**: Running SHAP on every screening image automatically would introduce a 10–15s latency on CPU hardware. Normal triage classification remains instantaneous (<100ms), and SHAP is triggered on-demand via the **"⚡ Explain Prediction with SHAP"** button.
- **Monte-Carlo Path Sampling**: Configurable `nsamples` parameter balances speed and precision:
  - `nsamples=15`: Rapid preview (~5s on CPU)
  - `nsamples=30`: Standard clinical screening resolution (~10–12s on CPU)
  - `nsamples=50`: High-fidelity research publication resolution (~18s on CPU)
- **Zero-Tensor Baseline**: Fundus imaging features an inherent black background outside the circular retinal aperture; zero reference baselines ensure attributions reflect tissue structure relative to the unilluminated boundary.

---

## 11. Medical Safety & Clinical Disclaimer

> [!CAUTION]
> **RESEARCH PROTOTYPE — NOT AN INDEPENDENT MEDICAL DIAGNOSIS**
> 
> SHAP visualizations explain which image regions influenced the AI classifier's prediction. They are not a medical diagnosis and do not prove the presence or absence of a retinal lesion. All screening recommendations must be reviewed and validated by a licensed ophthalmologist or qualified eye-care professional.
