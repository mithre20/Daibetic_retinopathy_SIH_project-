# Grad-CAM (Gradient-Weighted Class Activation Mapping)

## What is Grad-CAM?
Grad-CAM (Gradient-Weighted Class Activation Mapping) is a visual explainability technique that creates a coarse two-dimensional spatial heatmap over the retinal fundus image. It reveals which anatomical regions the convolutional neural network focused on when predicting a particular Diabetic Retinopathy grade.

## Which layer of the model does Grad-CAM inspect?
In our EfficientNet-B0 model, Grad-CAM targets the deepest convolutional feature extraction layer:
`backbone.features[8][0]` (Conv2d: 320 -> 1280 channels).
This layer retains high-level semantic abstractions combined with spatial resolution.

## Why is the heatmap shown?
Deep neural networks are often criticized as opaque "black boxes." Grad-CAM provides visual accountability so doctors, clinicians, and health workers can immediately verify whether the AI is focusing on actual retinal pathology (such as clusters of hemorrhages or exudates) or being misled by irrelevant peripheral imaging artifacts.

## What do the heatmap colors mean?
The heatmap uses the standard "jet" or "turbo" clinical colormap:
- **Red / Warm Highlights**: Regions with high positive gradients that contributed most strongly toward predicting the assigned DR severity grade.
- **Yellow / Green**: Moderate attribution regions.
- **Blue / Cool Areas**: Neutral regions that did not significantly influence the classifier's decision.

## Does Grad-CAM prove that a medical lesion exists?
**NO.** Grad-CAM identifies areas that contributed mathematically to the classifier's prediction. It represents *model attention patterns*, NOT direct etiological or histopathological proof of a medical lesion.

## Why does the highlighted area differ from U-Net?
Grad-CAM generates a coarse, smoothed attention envelope from deep convolutional feature maps (typically 16x16 or 32x32 pixels upsampled to 512x512). In contrast, U-Net produces fine, pixel-by-pixel morphological segmentations. A broad Grad-CAM highlight often covers a cluster of tiny microaneurysms detected by U-Net.
