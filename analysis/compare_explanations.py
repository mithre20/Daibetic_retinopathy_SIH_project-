"""
Multi-Modal Explainability Comparison Script
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Compares:
  1. Original Retinal Fundus Photograph
  2. Grad-CAM (CNN Feature Attention Heatmap)
  3. SHAP (Positive/Negative Feature Attribution Overlay)
  4. U-Net (Retinal Lesion Multi-Class Segmentation)

Usage:
    python analysis/compare_explanations.py --image fundus_scan.jpg [--output results/explanations_comparison.png]
"""

import os
import sys
import argparse
from typing import Optional
from PIL import Image
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.model.efficientnet_dr import build_model, get_device, CLASS_NAMES
from backend.model.inference import get_inference_transforms
from explainability.shap_explainer import generate_shap_explanation, find_model_checkpoint
from explainability.shap_visualizer import create_overlay as create_shap_overlay
from inference.gradcam import GradCAM, apply_colormap_and_overlay
from inference.predict_unet import predict_retinal_lesions


def generate_comparison_figure(
    image_path: str,
    output_path: Optional[str] = None,
    target_class: Optional[int] = None
) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if output_path is None:
        output_dir = os.path.join(ROOT_DIR, "results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "explanations_comparison.png")

    device = get_device()
    ckpt = find_model_checkpoint()
    if not ckpt:
        raise FileNotFoundError("EfficientNet-B0 checkpoint not found.")

    model = build_model(num_classes=5, pretrained=False, weights_path=ckpt, device=device)
    model.eval()

    orig_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_pil.size
    transform = get_inference_transforms()
    input_tensor = transform(orig_pil).unsqueeze(0).to(device)

    print("\n1/4: Running Model Inference...")
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    tgt_idx = pred_idx if target_class is None else target_class
    tgt_label = CLASS_NAMES[tgt_idx]

    print("2/4: Computing Grad-CAM (Target Layer: Conv2d Features)...")
    target_layer = model.get_gradcam_target_layer()
    with GradCAM(model, target_layer) as gcam:
        cam_res = gcam.generate_cam(input_tensor, target_class=tgt_idx)
        heatmap_raw = cam_res["heatmap"]
        _, _, _, gradcam_overlay_pil = apply_colormap_and_overlay(
            original_img=orig_pil,
            heatmap=heatmap_raw,
            alpha=0.45,
            colormap_name="jet"
        )

    print("3/4: Computing SHAP Feature Attribution...")
    shap_res = generate_shap_explanation(
        image=orig_pil,
        model=model,
        target_class=tgt_idx,
        nsamples=30,
        device=device
    )
    shap_overlay_pil = create_shap_overlay(
        original_img=orig_pil,
        combined_attr=shap_res["combined_map"],
        alpha=0.45,
        colormap_name="coolwarm",
        show_colorbar=False
    )

    print("4/4: Computing U-Net Lesion Segmentation...")
    try:
        unet_res = predict_retinal_lesions(image_input=image_path, img_size=512)
        unet_overlay_path = unet_res["lesion_overlay_path"]
        unet_overlay_pil = Image.open(unet_overlay_path).convert("RGB")
    except Exception as e:
        print(f"U-Net generation note: {e}")
        # Fallback to black canvas
        unet_overlay_pil = Image.new("RGB", (orig_w, orig_h), color=(20, 20, 30))

    # Assemble Side-by-Side Figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), dpi=200)
    fig.patch.set_facecolor("#0a0f1d")

    panels = [
        ("Original Fundus Scan\n(Color Image)", orig_pil, "#94a3b8"),
        (f"Grad-CAM (Attention)\nTarget: {tgt_label[:14]}...", gradcam_overlay_pil, "#38bdf8"),
        (f"SHAP (Contribution)\nPositive vs Negative", shap_overlay_pil, "#f43f5e"),
        ("U-Net (Lesion Map)\nAnatomical Segmentation", unet_overlay_pil, "#34d399")
    ]

    for ax, (title, img, border_color) in zip(axes, panels):
        ax.set_facecolor("#0a0f1d")
        ax.imshow(img)
        ax.set_title(title, fontsize=11, color="#f8fafc", weight="bold", pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.5)

    plt.suptitle(
        f"AI DR Screen — Multi-Modal Explainability Suite\n"
        f"EfficientNet-B0 Prediction: {pred_label} (Confidence: {confidence * 100:.1f}%)",
        fontsize=13,
        color="#f8fafc",
        weight="bold",
        y=1.03
    )

    plt.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"\nSuccessfully generated multi-modal comparison: {output_path}\n")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Modal Explainability Comparison")
    parser.add_argument("--image", type=str, default="fundus_scan.jpg", help="Path to fundus photograph")
    parser.add_argument("--output", type=str, default=None, help="Output comparison path")
    parser.add_argument("--target-class", type=int, default=None, help="Target class (0-4)")
    args = parser.parse_args()

    generate_comparison_figure(args.image, args.output, args.target_class)
