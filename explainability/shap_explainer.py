"""
SHAP Feature-Level Explainer for EfficientNet-B0 Diabetic Retinopathy Classifier
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India
"""

import os
import sys
import time
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.model.efficientnet_dr import (
    EfficientNetDRClassifier,
    build_model,
    get_device,
    CLASS_NAMES,
    CLASS_KEYS
)
from backend.model.inference import get_inference_transforms

import shap


def find_model_checkpoint(custom_path: Optional[str] = None) -> Optional[str]:
    """Finds existing EfficientNet-B0 weights, checking canonical paths."""
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    candidates = [
        os.path.join(ROOT_DIR, "models", "efficientnet_b0_dr.pth"),
        os.path.join(ROOT_DIR, "backend", "model", "model_weights.pth"),
        os.path.join(ROOT_DIR, "training", "checkpoints", "best_model.pth")
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None


class DRModelWrapper(nn.Module):
    """
    Clean wrapper around the 5-class EfficientNetDRClassifier for SHAP compatibility.
    Guarantees eval() mode, preserves gradient flow during attribution, and outputs 5-class logits.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs forward pass returning 5-class logits."""
        return self.model(x)

    def predict(self, images: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Accepts batch of preprocessed images and returns class probabilities.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        if isinstance(images, np.ndarray):
            tensor = torch.from_numpy(images).float().to(device)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-1] == 3: # HWC -> BCHW
                tensor = tensor.permute(0, 3, 1, 2)
        else:
            tensor = images.to(device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs


def compute_shap_maps(
    shap_attr: np.ndarray,
    target_class: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Processes multidimensional SHAP attribution into positive, negative, and combined 2D maps.

    shap_attr can be:
      - (1, 3, 224, 224, 5) or (3, 224, 224, 5) from GradientExplainer
      - or (1, 3, 224, 224) if single class sliced.
    """
    # 1. Extract attribution for the target class
    if shap_attr.ndim == 5:
        # (B, C, H, W, num_classes)
        class_attr = shap_attr[0, :, :, :, target_class] # (C, H, W)
    elif shap_attr.ndim == 4 and shap_attr.shape[-1] == 5:
        # (C, H, W, num_classes)
        class_attr = shap_attr[:, :, :, target_class]
    elif shap_attr.ndim == 4 and shap_attr.shape[0] == 1:
        # (1, C, H, W)
        class_attr = shap_attr[0]
    elif shap_attr.ndim == 3:
        # (C, H, W)
        class_attr = shap_attr
    else:
        raise ValueError(f"Unexpected SHAP array dimensions: {shap_attr.shape}")

    # Check for NaN / Inf
    if np.isnan(class_attr).any() or np.isinf(class_attr).any():
        class_attr = np.nan_to_num(class_attr, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Aggregate across color channels (summing preserves total net directional attribution)
    attr_2d = np.sum(class_attr, axis=0) # (224, 224)

    # 3. Separate positive and negative attributions
    # Positive: Features that push the model TOWARD the target class
    # Negative: Features that push the model AWAY FROM the target class
    pos_attr = np.maximum(attr_2d, 0.0)
    neg_attr = np.maximum(-attr_2d, 0.0)

    # 4. Normalize positive and negative maps independently to [0, 1] for visualization
    pos_max = np.max(pos_attr)
    pos_map = (pos_attr / (pos_max + 1e-8)) if pos_max > 0 else np.zeros_like(pos_attr)

    neg_max = np.max(neg_attr)
    neg_map = (neg_attr / (neg_max + 1e-8)) if neg_max > 0 else np.zeros_like(neg_attr)

    # 5. Combined diverging map centered at 0 in range [-1, 1]
    abs_max = max(pos_max, neg_max, 1e-8)
    combined_map = attr_2d / abs_max

    # 6. Compute quantitative summary statistics
    sparsity = float(np.mean(np.abs(attr_2d) < (0.05 * abs_max)))
    stats = {
        "min": float(np.min(attr_2d)),
        "max": float(np.max(attr_2d)),
        "mean": float(np.mean(attr_2d)),
        "std": float(np.std(attr_2d)),
        "pos_sum": float(np.sum(pos_attr)),
        "neg_sum": float(np.sum(neg_attr)),
        "sparsity": float(sparsity),
        "abs_max": float(abs_max)
    }

    return pos_map, neg_map, combined_map, stats


def generate_shap_explanation(
    image: Union[str, Image.Image, torch.Tensor],
    model: Optional[nn.Module] = None,
    target_class: Optional[int] = None,
    nsamples: int = 30,
    background: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates SHAP feature-level explanation for an input fundus image using EfficientNet-B0.

    Parameters:
        image: Filepath, PIL Image, or preprocessed 4D PyTorch tensor.
        model: Pre-loaded EfficientNetDRClassifier (optional; will load from weights if None).
        target_class: Integer 0..4. If None, defaults to the model's top predicted class.
        nsamples: Number of Monte-Carlo path samples for GradientExplainer (default 30).
        background: Baseline reference tensor (default: zero background).
        device: torch.device (CUDA or CPU).
        model_path: Custom checkpoint path if model is not pre-loaded.

    Returns:
        Structured result dict containing prediction, attributions, maps, and statistics.
    """
    t_start = time.time()

    if device is None:
        device = get_device()

    # 1. Load or validate model
    if model is None:
        ckpt = find_model_checkpoint(model_path)
        if not ckpt:
            raise FileNotFoundError(
                "Unable to generate SHAP explanation: Trained EfficientNet-B0 checkpoint not found."
            )
        model = build_model(num_classes=5, pretrained=False, weights_path=ckpt, device=device)

    model.eval()
    model.to(device)
    wrapped_model = DRModelWrapper(model)

    # 2. Image Preprocessing matching inference pipeline
    transform = get_inference_transforms()
    if isinstance(image, str):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"Retinal image file not found: {image}")
        pil_img = Image.open(image).convert("RGB")
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        orig_size = pil_img.size # (W, H)
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        orig_size = pil_img.size
    elif isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        input_tensor = image.to(device)
        orig_size = (input_tensor.shape[3], input_tensor.shape[2])
    else:
        raise ValueError(f"Unsupported image input type: {type(image)}")

    # 3. Run EfficientNet-B0 forward pass for prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_class = int(np.argmax(probabilities))
    pred_prob = float(probabilities[pred_class])
    pred_label = CLASS_NAMES[pred_class]

    # 4. Resolve target class
    if target_class is None:
        tgt_class = pred_class
    else:
        tgt_class = int(target_class)
        if tgt_class < 0 or tgt_class >= 5:
            raise ValueError(f"Target class must be in range 0..4 (inclusive), got: {tgt_class}")

    tgt_label = CLASS_NAMES[tgt_class]
    tgt_prob = float(probabilities[tgt_class])

    # 5. Baseline reference tensor for SHAP GradientExplainer
    if background is None:
        # Zero baseline is canonical for dark-background retinal fundus imaging
        background = torch.zeros((1, 3, 224, 224), device=device)
    else:
        background = background.to(device)

    # 6. Compute SHAP values with GradientExplainer (Expected Gradients)
    try:
        explainer = shap.GradientExplainer(wrapped_model, background)
        # GradientExplainer returns attribution for all 5 classes
        raw_shap = explainer.shap_values(input_tensor, ranked_outputs=None, nsamples=nsamples)
    except Exception as e:
        raise RuntimeError(
            f"Unable to generate SHAP explanation: Computation failed on {device}. Details: {str(e)}"
        )

    # Normalize raw_shap to numpy array
    if isinstance(raw_shap, list):
        # List of 5 arrays each (1, 3, 224, 224)
        shap_values_stacked = np.stack(raw_shap, axis=-1) # (1, 3, 224, 224, 5)
    elif isinstance(raw_shap, np.ndarray):
        shap_values_stacked = raw_shap
    else:
        shap_values_stacked = np.array(raw_shap)

    # 7. Compute positive, negative, and combined attribution maps
    pos_map, neg_map, combined_map, stats = compute_shap_maps(
        shap_attr=shap_values_stacked,
        target_class=tgt_class
    )

    t_elapsed = time.time() - t_start
    stats["computation_time_seconds"] = round(t_elapsed, 3)
    stats["nsamples"] = nsamples

    prob_dict = {
        CLASS_KEYS[i]: round(float(probabilities[i]), 4)
        for i in range(5)
    }

    return {
        "prediction_class": pred_class,
        "prediction_label": pred_label,
        "prediction_probability": round(pred_prob, 4),
        "target_class": tgt_class,
        "target_label": tgt_label,
        "target_probability": round(tgt_prob, 4),
        "probabilities": prob_dict,
        "shap_values": shap_values_stacked,
        "positive_map": pos_map,
        "negative_map": neg_map,
        "combined_map": combined_map,
        "shap_stats": stats,
        "original_size": orig_size,
        "input_tensor": input_tensor,
        "device": str(device)
    }
