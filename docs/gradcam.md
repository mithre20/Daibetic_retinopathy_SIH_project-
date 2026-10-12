# Grad-CAM Explainable AI for Diabetic Retinopathy Screening

**Project**: Explainable AI for Diabetic Retinopathy Screening in Rural India  
**Architecture**: EfficientNet-B0 (Transfer Learning) + Grad-CAM Feature Attribution  
**Complementary Model**: U-Net Retinal Lesion Segmentation  

---

## 1. What Grad-CAM Is

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is an explainable artificial intelligence (XAI) technique that provides visual interpretability for convolutional neural network (CNN) predictions. 

Unlike black-box classifiers that output only a discrete disease grade (e.g., "Moderate DR"), Grad-CAM reveals **which spatial regions of a retinal fundus image most strongly influenced the model's decision**. It accomplishes this by flowing gradient information from the predicted disease class back into the final convolutional feature layer.

> [!NOTE]
> Grad-CAM is **not** a separate disease-classification model or a newly trained neural network. It is an interpretability algorithm applied directly to the trained EfficientNet-B0 classifier.

---

## 2. Why It Is Used in Rural Screening

In primary healthcare centers (PHCs) across rural India, AI screening tools face significant clinical adoption barriers if predictions cannot be audited or explained.

Key motivations for incorporating Grad-CAM into XAIRIS:
1. **Clinical Trust & Auditability**: Rural medical officers, ophthalmic assistants, and community health workers can immediately verify whether the AI focused on clinically relevant retinal landmarks or spurious image artifacts.
2. **Rapid Triage Support**: Highlighting suspicious areas accelerates secondary review by remote ophthalmologists during tele-ophthalmology sessions.
3. **Safety & False-Positive Verification**: If an image is flagged as "Referable DR", Grad-CAM allows the clinician to inspect the visual evidence before initiating costly patient travel to a tertiary hospital.

---

## 3. How Grad-CAM Works: Mathematical Workflow

The standard Grad-CAM formulation proceeds in 6 sequential stages:

```
Retinal Fundus Scan [1, 3, 224, 224]
              ↓
     EfficientNet-B0 Forward Pass
              ↓
Feature Activations A^k ∈ ℝ^{1280 × 7 × 7} (Layer: features[8][0])
              ↓
Class Logits Y = [Y_0, Y_1, Y_2, Y_3, Y_4]
              ↓
Select Target Class score Y^c (Predicted or Specified)
              ↓
Backpropagate Gradients: ∂Y^c / ∂A^k
              ↓
Global Average Pooling (GAP) across spatial dimensions (i, j):
    α_k^c = (1 / Z) ∑_i ∑_j (∂Y^c / ∂A_{i,j}^k)
              ↓
Linear Combination of Feature Maps:
    L_{Grad-CAM}^c = ReLU( ∑_k α_k^c A^k )
              ↓
Min-Max Normalization to [0, 1]
              ↓
Bicubic Upsampling to Original Scan Resolution (e.g., 2048 × 1536)
              ↓
Colormap (Jet / Turbo) + Alpha Blending (α = 0.4)
              ↓
Visual Explanation Overlay
```

1. **Feature Map Capture**: Forward hook captures activations $A^k \in \mathbb{R}^{C \times H \times W}$ at the target convolutional layer.
2. **Gradient Backpropagation**: The score for target class $c$ ($Y^c$) is backpropagated; backward hook captures gradients $\frac{\partial Y^c}{\partial A^k}$.
3. **Importance Weight Calculation**: Global Average Pooling collapses gradients into channel importance scalar weights $\alpha_k^c$:
   $$\alpha_k^c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\partial Y^c}{\partial A_{i,j}^k}$$
4. **Weighted Combination & Rectification**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k=1}^{C} \alpha_k^c A^k\right)$$
   The $\text{ReLU}$ non-linearity guarantees that only features with a positive influence on the target class are retained.
5. **Spatial Scaling**: The $7 \times 7$ heatmap is resized via bicubic interpolation to match original image dimensions.
6. **Alpha Overlay**:
   $$I_{\text{overlay}} = \alpha \cdot I_{\text{heatmap}} + (1 - \alpha) \cdot I_{\text{original}}$$

---

## 4. EfficientNet-B0 Target Convolutional Layer

In `torchvision.models.efficientnet_b0`, the feature extractor consists of 9 sequential blocks (`features[0]` to `features[8]`).

- **Selected Target Layer**: `backbone.features[8][0]`
- **Block Type**: `Conv2dNormActivation` -> `Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)`
- **Feature Map Dimensions**: `[Batch, 1280, 7, 7]`

### Rationale:
`features[8][0]` is the final convolutional layer of the network immediately preceding `AdaptiveAvgPool2d(1)` and the 5-class linear classifier head. It possesses the highest-level semantic retinal representations while preserving spatial 2D grid coordinates ($7 \times 7$).

---

## 5. How the Heatmap Is Generated

The Grad-CAM engine generates three visual artifacts:
1. **Raw Heatmap (`heatmap.npy`)**: A float32 2D array normalized strictly to $[0.0, 1.0]$ for quantitative scientific research.
2. **Colored Heatmap (`heatmap.png`)**: The normalized values mapped through a standard scientific colormap (`jet` or `turbo`):
   - **Blue / Cooler**: Low model attribution / background retina.
   - **Yellow**: Medium model importance.
   - **Red / Warmer**: Maximum model attention driving the target class prediction.
3. **Heatmap Overlay (`overlay.png`)**: Alpha-blended composite ($I_{\text{overlay}} = 0.4 \cdot I_{\text{heatmap}} + 0.6 \cdot I_{\text{fundus}}$) ensuring underlying retinal anatomy (optic disc, macula, blood vessels) remains clearly visible.

---

## 6. Difference Between Grad-CAM and U-Net

A critical scientific distinction must be maintained in all clinical interfaces:

| Characteristic | Grad-CAM (`EfficientNet-B0`) | U-Net Segmentation |
| :--- | :--- | :--- |
| **Model Type** | Classifier Attribution Method | Dense Semantic Segmentation Model |
| **Scientific Meaning** | **"Model Attention / Importance Map"** | **"Retinal Lesion Segmentation"** |
| **What It Shows** | Image regions that influenced the model's DR grade score | Anatomical boundaries of microaneurysms, hemorrhages, and exudates |
| **Resolution** | Coarse ($7 \times 7$ interpolated) | Pixel-level fine anatomy ($512 \times 512$) |
| **Clinical Interpretation** | *"Highlighted regions contributed strongly to the classification."* | *"Independently identifies predicted lesion clusters."* |

> [!IMPORTANT]
> Never claim that Grad-CAM "proves a lesion caused the disease". Grad-CAM and U-Net are **complementary explainability methods**: Grad-CAM explains classifier decision-making, while U-Net segments actual retinal lesions.

---

## 7. How to Run Inference

All inference scripts utilize the project virtual environment (`.venv`):

### A. Single Fundus Image Explanation
```bash
# Explain using model's predicted class (default)
python inference/gradcam_efficientnet.py --image path/to/fundus.jpg

# Explain with custom overlay opacity (e.g. 0.5)
python inference/gradcam_efficientnet.py --image path/to/fundus.jpg --alpha 0.5
```

### B. Specific Target Class Inspection
Inspect what image regions the model considers indicative of a specific severity grade:
```bash
# Target Class 0 = No DR
# Target Class 1 = Mild DR
# Target Class 2 = Moderate DR
# Target Class 3 = Severe DR
# Target Class 4 = Proliferative DR
python inference/gradcam_efficientnet.py --image path/to/fundus.jpg --target-class 3
```

### C. Combined U-Net + Grad-CAM Explanation
Runs both models, calculates quantitative spatial overlap (IoU, Dice, % lesion coverage), and generates a 5-panel combined visualization:
```bash
python inference/gradcam_efficientnet.py --image path/to/fundus.jpg --with-unet
```

### D. Batch Explanation on Directory of Images
```bash
python inference/gradcam_efficientnet.py --folder data/test --max-images 10
```

### E. Standalone Classification Inference
```bash
python inference/predict_efficientnet.py --image path/to/fundus.jpg
```

---

## 8. Limitations

1. **Coarse Spatial Resolution**: Because EfficientNet-B0 downsamples the $224 \times 224$ input by a factor of 32, feature maps at `features[8]` have a spatial resolution of $7 \times 7$. Consequently, Grad-CAM highlights general anatomical zones rather than individual sub-pixel microaneurysms.
2. **Sensitivity to Image Artifacts**: Poor illumination, dust on camera lenses, or severe cataract haze may attract model attribution if they correlate with training patterns.
3. **Attribution Does Not Imply Causality**: A highlighted region indicates statistical correlation with the predicted class, not medical causality.

---

## 9. Example Output

```text
========================================
GRAD-CAM EXPLAINABLE AI
========================================

Image:
fundus_scan.jpg

Prediction:
No Diabetic Retinopathy

Confidence:
68.6%

Class probabilities:
No DR: 68.6%
Mild DR: 6.0%
Moderate DR: 8.8%
Severe DR: 9.4%
Proliferative DR: 7.2%

Target class:
No Diabetic Retinopathy

Grad-CAM target layer:
backbone.features[8][0] (Conv2d: 320 -> 1280)

Heatmap min:
0.0
Heatmap max:
0.10847
Heatmap mean:
0.03632
Heatmap dimensions:
[7, 7]

Heatmap:
results/gradcam/heatmap.png

Overlay:
results/gradcam/overlay.png
========================================
```

---

## 10. Medical Disclaimer

> **Clinical Decision Support Notice**:  
> Grad-CAM provides an interpretability visualization of the model's prediction. Highlighted regions should not be interpreted as definitive clinical evidence or a diagnosis. All screening outputs must be confirmed by a licensed ophthalmologist or qualified eye-care professional.
