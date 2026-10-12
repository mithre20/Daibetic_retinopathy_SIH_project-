"""
Visualization of U-Net Retinal Lesion Segmentation Outputs (IDRiD)
Generates 4-panel figures:
[Original Fundus Photo] | [Ground Truth Lesion Mask] | [U-Net Predicted Mask] | [Explainable Lesion Overlay]
"""

import os
import sys
import json
import argparse
from typing import Optional, List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torchvision.transforms.functional as TF

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.unet_model import UNet, build_unet, get_device, LESION_CLASSES, LESION_COLORS
from training.dataset_unet import (
    DEFAULT_IDRID_PATH,
    find_lesion_masks_for_image,
    combine_lesion_masks,
    IMAGE_NET_MEAN,
    IMAGE_NET_STD
)


def create_color_mask(mask_2d: np.ndarray) -> np.ndarray:
    """Converts a 2D class-ID mask into an RGB image using clinical lesion colors."""
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in LESION_COLORS.items():
        if class_id == 0:
            continue
        rgb[mask_2d == class_id] = color
    return rgb


def create_lesion_overlay(orig_img_rgb: np.ndarray, mask_2d: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blends the original fundus photo with the colored lesion mask."""
    overlay = orig_img_rgb.copy().astype(np.float32)
    for class_id, color in LESION_COLORS.items():
        if class_id == 0:
            continue
        idx = (mask_2d == class_id)
        if np.any(idx):
            color_arr = np.array(color, dtype=np.float32)
            overlay[idx] = (1.0 - alpha) * overlay[idx] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def visualize_test_predictions(
    model_path: str,
    dataset_path: str,
    split_info_path: str,
    num_samples: int = 5,
    img_size: int = 512,
    output_dir: Optional[str] = None
) -> List[str]:
    device = get_device()

    if not os.path.exists(model_path):
        print(f"[!] MODEL NOT FOUND: {model_path}")
        return []

    if not os.path.exists(split_info_path):
        print(f"[!] SPLIT INFO NOT FOUND: {split_info_path}")
        return []

    with open(split_info_path, "r") as f:
        split_data = json.load(f)

    test_images = split_data.get("test_images", [])
    if not test_images:
        print("[!] No test images in split file.")
        return []

    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "results", "segmentation_examples")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = build_unet(in_channels=3, num_classes=5, weights_path=model_path, device=device)
    model.eval()

    sample_images = test_images[:num_samples]
    saved_files = []

    print(f"\nGenerating {len(sample_images)} test sample visualizations...")

    # Color legend handles
    legend_patches = [
        mpatches.Patch(color=np.array(LESION_COLORS[1]) / 255.0, label="Microaneurysm (MA)"),
        mpatches.Patch(color=np.array(LESION_COLORS[2]) / 255.0, label="Hemorrhage (HE)"),
        mpatches.Patch(color=np.array(LESION_COLORS[3]) / 255.0, label="Hard Exudate (EX)"),
        mpatches.Patch(color=np.array(LESION_COLORS[4]) / 255.0, label="Soft Exudate (SE)")
    ]

    for idx, img_name in enumerate(sample_images):
        # Locate image file
        img_path = None
        for root, _, files in os.walk(dataset_path):
            if img_name in files:
                img_path = os.path.join(root, img_name)
                break
        if not img_path:
            continue

        raw_img = Image.open(img_path).convert("RGB")
        resized_img = raw_img.resize((img_size, img_size), Image.BILINEAR)
        img_np = np.array(resized_img)

        # Ground truth
        mask_paths = find_lesion_masks_for_image(img_path, dataset_path)
        gt_mask = combine_lesion_masks(mask_paths, (img_size, img_size))
        gt_colored = create_color_mask(gt_mask)

        # Predict
        tensor = TF.to_tensor(resized_img)
        tensor = TF.normalize(tensor, mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_colored = create_color_mask(pred_mask)
        overlay_img = create_lesion_overlay(img_np, pred_mask, alpha=0.55)

        # Render 4-panel figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0e1526")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        axes[0].imshow(img_np)
        axes[0].set_title("1. Original Fundus Scan", color="white", fontsize=12, pad=10)

        axes[1].imshow(gt_colored)
        axes[1].set_title("2. Ground-Truth Lesions", color="white", fontsize=12, pad=10)

        axes[2].imshow(pred_colored)
        axes[2].set_title("3. U-Net Prediction", color="white", fontsize=12, pad=10)

        axes[3].imshow(overlay_img)
        axes[3].set_title("4. Explainable Overlay", color="white", fontsize=12, pad=10)

        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=4,
            facecolor="#151d32",
            edgecolor="#2a3b5c",
            labelcolor="white",
            fontsize=10,
            bbox_to_anchor=(0.5, -0.05)
        )

        out_name = f"lesion_seg_sample_{idx + 1}_{os.path.splitext(img_name)[0]}.png"
        out_path = os.path.join(output_dir, out_name)
        plt.savefig(out_path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

        saved_files.append(out_path)
        print(f"  [{idx + 1}/{len(sample_images)}] Saved: {out_path}")

    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize U-Net Retinal Lesion Segmentation Results")
    parser.add_argument("--model", type=str, default=os.path.join(ROOT_DIR, "models", "unet_retinal_lesion.pth"))
    parser.add_argument("--dataset", type=str, default=DEFAULT_IDRID_PATH)
    parser.add_argument("--splits", type=str, default=os.path.join(ROOT_DIR, "results", "unet_split_info.json"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    visualize_test_predictions(
        model_path=args.model,
        dataset_path=args.dataset,
        split_info_path=args.splits,
        num_samples=args.samples,
        img_size=args.img_size,
        output_dir=args.output
    )
