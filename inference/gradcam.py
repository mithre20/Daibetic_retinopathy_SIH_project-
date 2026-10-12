"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Engine for Explainable AI
Designed for Diabetic Retinopathy Screening on Retinal Fundus Images.

Scientific Principles:
- Computes gradients of the target class score with respect to feature activation maps.
- Pools gradients via Global Average Pooling (GAP) to obtain channel importance weights (alpha).
- Combines feature maps with weights, applies ReLU, and normalizes to [0, 1].
- Produces visually interpretable heatmaps and alpha-blended overlays on retinal fundus scans.
- Distinguishes model attribution (Grad-CAM) from anatomical lesion segmentation (U-Net).

Medical Disclaimer:
Grad-CAM provides an interpretability visualization of the model's prediction.
Highlighted regions indicate image areas that contributed strongly to the model's
prediction and should not be interpreted as definitive clinical evidence or diagnosis.
"""

import os
import io
import base64
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

CLASS_KEYS = [
    "no_dr",
    "mild",
    "moderate",
    "severe",
    "proliferative"
]

MEDICAL_DISCLAIMER = (
    "Grad-CAM provides an interpretability visualization of the model's prediction. "
    "Highlighted regions indicate image areas that contributed strongly to the model's prediction "
    "and should not be interpreted as definitive clinical evidence or a diagnosis. "
    "All AI findings must be verified by a qualified eye-care professional."
)


class GradCAM:
    """
    Standard PyTorch implementation of Grad-CAM.
    Works with any PyTorch convolutional architecture, specifically tuned
    for EfficientNet-B0 and ResNet-50 backbones.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch classification model (in eval mode).
            target_layer: The target convolutional layer to attach hooks to.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward and backward hooks on the target convolutional layer."""
        def forward_hook(module, input, output):
            # Capture feature activations [B, C, H, W]
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Capture output gradients [B, C, H, W]
            self.gradients = grad_output[0].detach()

        # Register forward hook
        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        self.handles.append(h_fwd)

        # Register backward hook (using register_full_backward_hook for PyTorch >= 1.8)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        else:
            h_bwd = self.target_layer.register_backward_hook(backward_hook)
        self.handles.append(h_bwd)

    def remove_hooks(self):
        """Removes registered hooks to prevent memory leaks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates Grad-CAM heatmap for the given input tensor and target class.

        Args:
            input_tensor: 4D tensor (1, 3, H, W), normalized and on model device.
            target_class: Integer class index (0..4). If None, uses model's predicted class.

        Returns:
            Dictionary containing:
                - heatmap: 2D numpy array [H_feat, W_feat] normalized in [0, 1]
                - prediction: Predicted class index
                - target_class: Target class index used for attribution
                - confidence: Probability of predicted class
                - probabilities: 5-class probability dictionary
                - logits: Raw output logits tensor
                - sanity_stats: Dict with min, max, mean, and shape
        """
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        # Ensure model is in eval mode but gradients are tracked
        was_training = self.model.training
        self.model.eval()

        # Forward pass (enable grad for backward backprop)
        with torch.enable_grad():
            tensor_req_grad = input_tensor.clone().requires_grad_(True)
            logits = self.model(tensor_req_grad)

            probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

            # Select target class
            if target_class is None:
                selected_class = predicted_class
            else:
                if not (0 <= target_class < logits.shape[1]):
                    raise ValueError(f"Invalid target_class {target_class}. Must be 0 <= c < {logits.shape[1]}")
                selected_class = int(target_class)

            # Score to backpropagate
            score = logits[0, selected_class]
            score.backward(retain_graph=False)

        # Restore original training mode if needed
        self.model.train(was_training)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GRAD-CAM GENERATION FAILED: Activations or gradients were not captured by hooks.")

        # Activations and gradients: [1, C, H, W]
        act = self.activations[0]  # [C, H, W]
        grad = self.gradients[0]  # [C, H, W]

        # Global Average Pooling (GAP) of gradients across spatial dimensions (H, W) -> [C]
        weights = torch.mean(grad, dim=(1, 2))  # [C]

        # Weighted combination of feature maps: sum_k (alpha_k * A^k)
        cam = torch.zeros(act.shape[1:], dtype=torch.float32, device=act.device)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * act[i]

        # Apply ReLU to retain only features with positive influence on target class
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()

        # Sanity check before normalization
        c_min = float(np.min(cam_np))
        c_max = float(np.max(cam_np))
        c_mean = float(np.mean(cam_np))
        c_shape = list(cam_np.shape)

        # Min-max normalize to [0, 1]
        denom = c_max - c_min
        if denom > 1e-8:
            cam_norm = (cam_np - c_min) / denom
        else:
            # All zero/uniform activation
            cam_norm = np.zeros_like(cam_np)

        prob_dict = {
            CLASS_KEYS[i]: round(float(probs[i]), 4)
            for i in range(len(CLASS_KEYS))
        }

        return {
            "heatmap": cam_norm,
            "prediction": predicted_class,
            "predicted_severity": CLASS_NAMES[predicted_class],
            "target_class": selected_class,
            "target_severity": CLASS_NAMES[selected_class],
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "sanity_stats": {
                "min": round(c_min, 6),
                "max": round(c_max, 6),
                "mean": round(c_mean, 6),
                "dimensions": c_shape,
                "is_finite": bool(np.all(np.isfinite(cam_norm)))
            }
        }


def apply_colormap_and_overlay(
    original_img: Union[str, Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap_name: str = "jet"
) -> Tuple[np.ndarray, np.ndarray, Image.Image, Image.Image]:
    """
    Resizes heatmap to original image dimensions, applies colormap, and creates alpha overlay.

    Args:
        original_img: Path to image, PIL Image, or uint8 RGB numpy array.
        heatmap: 2D float numpy array in [0, 1].
        alpha: Transparency factor in [0, 1]. 0.0 = only fundus, 1.0 = only heatmap.
        colormap_name: Matplotlib colormap name ('jet', 'turbo', 'inferno').

    Returns:
        (orig_np, heatmap_rgb, heatmap_pil, overlay_pil)
    """
    # 1. Load original image as PIL RGB
    if isinstance(original_img, str):
        if not os.path.isfile(original_img):
            raise FileNotFoundError(f"INPUT IMAGE NOT FOUND: {original_img}")
        orig_pil = Image.open(original_img).convert("RGB")
    elif isinstance(original_img, Image.Image):
        orig_pil = original_img.convert("RGB")
    elif isinstance(original_img, np.ndarray):
        orig_pil = Image.fromarray(original_img).convert("RGB")
    else:
        raise ValueError(f"Unsupported original_img type: {type(original_img)}")

    orig_w, orig_h = orig_pil.size
    orig_np = np.array(orig_pil, dtype=np.uint8)

    # 2. Resize heatmap to exact original dimensions using Bicubic interpolation
    heatmap_pil_raw = Image.fromarray(np.uint8(np.clip(heatmap * 255.0, 0, 255)))
    heatmap_resized = heatmap_pil_raw.resize((orig_w, orig_h), Image.BICUBIC)
    heatmap_resized_norm = np.array(heatmap_resized, dtype=np.float32) / 255.0

    # 3. Apply Colormap
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        cmap = plt.get_cmap(colormap_name)
    # Colormap produces RGBA in [0, 1]
    colored_rgba = cmap(heatmap_resized_norm)
    heatmap_rgb = (colored_rgba[:, :, :3] * 255.0).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # 4. Alpha Blending: overlay = alpha * heatmap + (1 - alpha) * original
    alpha = max(0.0, min(1.0, float(alpha)))
    overlay_np = (alpha * heatmap_rgb.astype(np.float32) + (1.0 - alpha) * orig_np.astype(np.float32))
    overlay_np = np.clip(overlay_np, 0, 255).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_np)

    return orig_np, heatmap_rgb, heatmap_pil, overlay_pil


def calculate_gradcam_unet_overlap(
    heatmap_resized: np.ndarray,
    unet_mask: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Quantitative research metric analyzing overlap between high-attribution
    Grad-CAM regions and U-Net lesion segmentation masks.

    Args:
        heatmap_resized: 2D float array in [0, 1] resized to match U-Net mask dimensions.
        unet_mask: 2D integer array where 0=background, 1=microaneurysm, 2=hemorrhage,
                   3=hard_exudate, 4=soft_exudate.
        threshold: Heatmap threshold to define 'high importance' region (default: 0.5).

    Returns:
        Dictionary with per-lesion and overall intersection, union, IoU, and overlap percentages.
    """
    if heatmap_resized.shape != unet_mask.shape:
        # Resize heatmap if shape mismatch
        h, w = unet_mask.shape
        h_pil = Image.fromarray((heatmap_resized * 255.0).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        heatmap_resized = np.array(h_pil, dtype=np.float32) / 255.0

    # Binarize Grad-CAM region
    gradcam_bin = (heatmap_resized >= threshold).astype(bool)
    gradcam_pixels = int(np.sum(gradcam_bin))

    lesion_keys = {
        1: "microaneurysm",
        2: "hemorrhage",
        3: "hard_exudate",
        4: "soft_exudate"
    }

    per_lesion_stats = {}
    total_lesion_mask = (unet_mask > 0).astype(bool)
    total_lesion_pixels = int(np.sum(total_lesion_mask))

    for class_id, name in lesion_keys.items():
        lesion_bin = (unet_mask == class_id)
        l_count = int(np.sum(lesion_bin))

        if l_count > 0:
            intersection = int(np.sum(gradcam_bin & lesion_bin))
            union = int(np.sum(gradcam_bin | lesion_bin))
            iou = round(float(intersection / union), 4) if union > 0 else 0.0
            lesion_covered_pct = round(float((intersection / l_count) * 100.0), 2)
            per_lesion_stats[name] = {
                "detected": True,
                "lesion_pixels": l_count,
                "overlap_pixels": intersection,
                "iou": iou,
                "lesion_coverage_pct": lesion_covered_pct
            }
        else:
            per_lesion_stats[name] = {
                "detected": False,
                "lesion_pixels": 0,
                "overlap_pixels": 0,
                "iou": 0.0,
                "lesion_coverage_pct": 0.0
            }

    overall_intersection = int(np.sum(gradcam_bin & total_lesion_mask))
    overall_union = int(np.sum(gradcam_bin | total_lesion_mask))
    overall_iou = round(float(overall_intersection / overall_union), 4) if overall_union > 0 else 0.0
    overall_coverage = round(float((overall_intersection / total_lesion_pixels) * 100.0), 2) if total_lesion_pixels > 0 else 0.0

    return {
        "gradcam_threshold": threshold,
        "gradcam_attention_pixels": gradcam_pixels,
        "total_lesion_pixels": total_lesion_pixels,
        "overall_intersection": overall_intersection,
        "overall_iou": overall_iou,
        "overall_lesion_coverage_pct": overall_coverage,
        "per_lesion": per_lesion_stats,
        "scientific_note": (
            "Overlap metrics evaluate spatial concordance between classifier attention and "
            "segmentation findings for research inspection only. High overlap does not constitute "
            "proof of etiology or clinical validation."
        )
    }


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL Image to a base64 encoded data URI."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"
