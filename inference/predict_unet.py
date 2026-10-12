"""
Inference Script for U-Net Retinal Lesion Segmentation

Usage:
    python inference/predict_unet.py --image path/to/fundus.jpg [--model models/unet_retinal_lesion.pth]

Produces:
    - results/inference/predicted_mask.png
    - results/inference/lesion_overlay.png
    - results/inference/microaneurysm_probability.png
    - results/inference/hemorrhage_probability.png
    - results/inference/hard_exudate_probability.png
    - results/inference/soft_exudate_probability.png
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.unet_model import UNet, build_unet, get_device, LESION_CLASSES, LESION_COLORS
from training.dataset_unet import IMAGE_NET_MEAN, IMAGE_NET_STD
from training.visualize_unet import create_color_mask, create_lesion_overlay


def generate_clinical_explanation(lesion_detections: Dict[str, bool], pixel_coverage: Dict[str, float]) -> str:
    """
    Generates honest, grounded clinical explanation text supported strictly by segmentation evidence.
    Does NOT generate unsupported medical claims.
    """
    detected_types = []
    if lesion_detections.get("microaneurysm", False):
        detected_types.append("microaneurysms (early vascular damage)")
    if lesion_detections.get("hemorrhage", False):
        detected_types.append("intraretinal hemorrhages (bleeding)")
    if lesion_detections.get("hard_exudate", False):
        detected_types.append("hard exudates (lipid/protein deposits)")
    if lesion_detections.get("soft_exudate", False):
        detected_types.append("soft exudates / cotton-wool spots (ischemic nerve fiber swelling)")

    if not detected_types:
        return (
            "U-Net Segmentation Analysis: No distinct focal lesion clusters "
            "(microaneurysms, hemorrhages, hard exudates, or soft exudates) "
            "were identified above clinical detection thresholds."
        )

    explanation = (
        f"Explainable AI Finding: Abnormal retinal regions were identified in areas corresponding to "
        f"{', '.join(detected_types)}. "
        f"These segmented regions highlight anatomical zones warranting clinical review by an eye-care professional."
    )
    return explanation


def predict_retinal_lesions(
    image_input,
    model_path: Optional[str] = None,
    img_size: int = 512,
    output_dir: Optional[str] = None,
    min_pixel_threshold: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    if device is None:
        device = get_device()

    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "results", "inference")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Image
    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Input image not found: {image_input}")
        image_name = os.path.basename(image_input)
        img_pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image_name = "uploaded_image.jpg"
        img_pil = image_input.convert("RGB")
    else:
        import io
        image_name = "uploaded_image.jpg"
        img_pil = Image.open(io.BytesIO(image_input)).convert("RGB")

    orig_w, orig_h = img_pil.size

    # Resize matching training resolution (Bilinear)
    img_resized = img_pil.resize((img_size, img_size), Image.BILINEAR)
    img_np = np.array(img_resized)

    # 2. Build or Load Model
    if model_path is None:
        default_model = os.path.join(ROOT_DIR, "models", "unet_retinal_lesion.pth")
        model_path = default_model if os.path.exists(default_model) else None

    model = build_unet(
        in_channels=3,
        num_classes=5,
        weights_path=model_path,
        device=device
    )
    model.eval()

    # 3. Preprocessing
    tensor = TF.to_tensor(img_resized)
    tensor = TF.normalize(tensor, mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        logits = model(tensor)  # [1, 5, H, W]
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # [5, H, W]
        pred_mask = np.argmax(probs, axis=0)  # [H, W]

    # 5. Lesion Statistics & Coverage
    total_pixels = img_size * img_size
    lesion_keys = ["microaneurysm", "hemorrhage", "hard_exudate", "soft_exudate"]
    detections = {}
    coverage_pct = {}
    pixel_counts = {}

    for class_id in range(1, 5):
        key = lesion_keys[class_id - 1]
        count = int(np.sum(pred_mask == class_id))
        pct = round(float((count / total_pixels) * 100.0), 3)

        pixel_counts[key] = count
        coverage_pct[key] = pct
        detections[key] = bool(count >= min_pixel_threshold)

    # 6. Generate Outputs
    # A. Color Mask
    color_mask = create_color_mask(pred_mask)
    mask_pil = Image.fromarray(color_mask)
    mask_path = os.path.join(output_dir, "predicted_mask.png")
    mask_pil.save(mask_path)

    # B. Lesion Overlay
    overlay_arr = create_lesion_overlay(img_np, pred_mask, alpha=0.55)
    overlay_pil = Image.fromarray(overlay_arr)
    overlay_path = os.path.join(output_dir, "lesion_overlay.png")
    overlay_pil.save(overlay_path)

    # C. Individual Probability Maps
    prob_paths = {}
    prob_cmap = plt.get_cmap("inferno")
    for class_id in range(1, 5):
        key = lesion_keys[class_id - 1]
        class_prob = probs[class_id]  # [H, W] in [0, 1]
        colored_prob = (prob_cmap(class_prob)[:, :, :3] * 255).astype(np.uint8)
        prob_img = Image.fromarray(colored_prob)
        p_path = os.path.join(output_dir, f"{key}_probability.png")
        prob_img.save(p_path)
        prob_paths[key] = p_path

    # Clinical explanation
    explanation = generate_clinical_explanation(detections, coverage_pct)

    # Print output matching requirement
    print(f"\nInput image: {image_name}")
    print("Detected lesion regions:")
    print(f"  Microaneurysm : {'detected' if detections['microaneurysm'] else 'not detected'}")
    print(f"  Hemorrhage    : {'detected' if detections['hemorrhage'] else 'not detected'}")
    print(f"  Hard Exudate  : {'detected' if detections['hard_exudate'] else 'not detected'}")
    print(f"  Soft Exudate  : {'detected' if detections['soft_exudate'] else 'not detected'}")
    print("Pixel coverage:")
    print(f"  Microaneurysm : {coverage_pct['microaneurysm']}%")
    print(f"  Hemorrhage    : {coverage_pct['hemorrhage']}%")
    print(f"  Hard Exudate  : {coverage_pct['hard_exudate']}%")
    print(f"  Soft Exudate  : {coverage_pct['soft_exudate']}%")
    print(f"\nClinical Summary:\n  {explanation}\n")
    print(f"Saved predicted mask to   : {mask_path}")
    print(f"Saved lesion overlay to    : {overlay_path}")

    return {
        "image": image_name,
        "input_dimensions": [orig_w, orig_h],
        "processed_dimensions": [img_size, img_size],
        "detections": detections,
        "pixel_coverage_percentage": coverage_pct,
        "pixel_counts": pixel_counts,
        "explanation": explanation,
        "predicted_mask_path": mask_path,
        "lesion_overlay_path": overlay_path,
        "probability_map_paths": prob_paths
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run U-Net Retinal Lesion Segmentation on a Fundus Photograph")
    parser.add_argument("--image", type=str, required=True, help="Path to input retinal fundus photograph")
    parser.add_argument("--model", type=str, default=None, help="Path to trained U-Net checkpoint (.pth)")
    parser.add_argument("--img-size", type=int, default=512, help="Segmentation image dimension (default: 512)")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output masks and overlays")
    args = parser.parse_args()

    predict_retinal_lesions(
        image_input=args.image,
        model_path=args.model,
        img_size=args.img_size,
        output_dir=args.output
    )
