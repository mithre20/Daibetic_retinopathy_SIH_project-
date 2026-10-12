# Comparison of Explainability Methods: Grad-CAM vs. SHAP vs. U-Net

## Why are Grad-CAM, SHAP, and U-Net Different?
Users and clinicians often ask why the highlights shown in Grad-CAM, SHAP, and U-Net do not look identical. They differ fundamentally because they answer three different mathematical and clinical questions:

| Feature | Grad-CAM | SHAP (Expected Gradients) | U-Net Segmentation |
| :--- | :--- | :--- | :--- |
| **Question Answered** | "Where did the classifier focus its attention?" | "Which pixels pushed the score up vs. down?" | "Where are the physical boundaries of lesions?" |
| **Output Type** | Coarse, smooth gradient heatmap | Fine-grained positive/negative attribution map | Pixel-level multi-class lesion masks |
| **Spatial Resolution** | Low/Coarse (interpolated from 16x16 deep layers) | High (pixel-level input resolution) | High (pixel-level 512x512 resolution) |
| **Lesion Classes** | None (class-agnostic heat energy) | Directional (positive vs. negative) | 4 explicit classes (Microaneurysm, Hemorrhage, Hard Exudate, Soft Exudate) |
| **Clinical Meaning** | Heuristic model attention | Additive feature importance | Anatomical lesion localization |

## Why Complementary Rather Than Identical?
1. **Grad-CAM** gives an immediate "high-level glance" at the region of interest.
2. **SHAP** verifies whether subtle pixel groups provided confirmatory or contradictory evidence.
3. **U-Net** marks the exact physical micro-lesions, allowing doctors to count lesions and measure pixel area coverage.

Because Grad-CAM uses coarse deep-layer representations while U-Net segments fine morphological edges, they will not match pixel-for-pixel, and are designed to complement each other.
