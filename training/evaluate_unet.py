"""
Evaluation Script for U-Net Retinal Lesion Segmentation on Unseen Test Split

Calculates:
- Overall: Mean Dice, Mean IoU, Pixel Accuracy
- Per-class: Dice, IoU, Precision, Recall for Microaneurysms, Hemorrhages, Hard Exudates, Soft Exudates
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.unet_model import UNet, build_unet, get_device, LESION_CLASSES
from training.dataset_unet import (
    DEFAULT_IDRID_PATH,
    locate_idrid_dataset,
    IDRiDLesionDataset
)


def evaluate_unet_test_set(
    model_path: str,
    dataset_path: str,
    split_info_path: str,
    batch_size: int = 4,
    img_size: int = 512,
    output_metrics_path: Optional[str] = None
) -> Dict[str, Any]:
    device = get_device()

    # Verify model file
    if not os.path.exists(model_path):
        msg = f"[!] MODEL NOT FOUND: Trained checkpoint '{model_path}' does not exist."
        print(msg)
        return {"status": "MODEL_NOT_FOUND", "error": msg}

    # Verify split info
    if not os.path.exists(split_info_path):
        msg = f"[!] SPLIT INFO NOT FOUND: Split metadata '{split_info_path}' does not exist."
        print(msg)
        return {"status": "SPLIT_NOT_FOUND", "error": msg}

    with open(split_info_path, "r") as f:
        split_data = json.load(f)

    test_image_names = split_data.get("test_images", [])
    if len(test_image_names) == 0:
        msg = "[!] No test images recorded in split info."
        print(msg)
        return {"status": "NO_TEST_IMAGES", "error": msg}

    # Locate test image absolute paths
    test_paths = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f in test_image_names:
                test_paths.append(os.path.join(root, f))
    test_paths = sorted(list(set(test_paths)))

    if len(test_paths) == 0:
        msg = f"[!] Could not find test image files inside '{dataset_path}'."
        print(msg)
        return {"status": "FILES_NOT_FOUND", "error": msg}

    print(f"Loaded {len(test_paths)} unseen test images from split metadata.")

    # Build dataset and loader
    test_dataset = IDRiDLesionDataset(test_paths, dataset_path, img_size=img_size, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = build_unet(in_channels=3, num_classes=5, weights_path=model_path, device=device)
    model.eval()

    # Per-class counters: TP, FP, FN, TN
    num_classes = 5
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)
    total_pixels = 0
    correct_pixels = 0

    print("\nRunning inference across unseen test split...")
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            total_pixels += masks.numel()
            correct_pixels += (preds == masks).sum().item()

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for c in range(num_classes):
                p_c = (preds_np == c)
                m_c = (masks_np == c)

                tp[c] += np.logical_and(p_c, m_c).sum()
                fp[c] += np.logical_and(p_c, np.logical_not(m_c)).sum()
                fn[c] += np.logical_and(np.logical_not(p_c), m_c).sum()

    pixel_accuracy = float(correct_pixels / max(total_pixels, 1))

    # Calculate per-class metrics
    class_metrics = {}
    class_keys = ["background", "microaneurysm", "hemorrhage", "hard_exudate", "soft_exudate"]
    dices = []
    ious = []

    for c in range(num_classes):
        intersection = tp[c]
        union = tp[c] + fp[c] + fn[c]
        cardinality = 2.0 * tp[c] + fp[c] + fn[c]

        dice = float((2.0 * intersection + 1e-5) / (cardinality + 1e-5))
        iou = float((intersection + 1e-5) / (union + 1e-5))
        prec = float((intersection + 1e-5) / (intersection + fp[c] + 1e-5))
        rec = float((intersection + 1e-5) / (intersection + fn[c] + 1e-5))

        dices.append(dice)
        ious.append(iou)

        class_metrics[class_keys[c]] = {
            "name": LESION_CLASSES[c],
            "dice": round(dice, 4),
            "iou": round(iou, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "true_positive_pixels": int(tp[c]),
            "false_positive_pixels": int(fp[c]),
            "false_negative_pixels": int(fn[c])
        }

    mean_dice = float(np.mean(dices))
    mean_iou = float(np.mean(ious))

    # Print formatted clinical report matching requirement
    print("\n========================================")
    print("U-NET RETINAL LESION SEGMENTATION")
    print(f"Test Images: {len(test_paths)}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")

    for key, label in [
        ("microaneurysm", "Microaneurysm"),
        ("hemorrhage", "Hemorrhage"),
        ("hard_exudate", "Hard Exudate"),
        ("soft_exudate", "Soft Exudate")
    ]:
        cm = class_metrics[key]
        print(f"\n{label}")
        print(f"Dice: {cm['dice']:.4f}")
        print(f"IoU: {cm['iou']:.4f}")
        print(f"Precision: {cm['precision']:.4f}")
        print(f"Recall: {cm['recall']:.4f}")
    print("========================================\n")

    results_data = {
        "model": "U-Net",
        "dataset": "IDRiD Retinal Lesion Segmentation",
        "test_images_count": len(test_paths),
        "mean_dice": round(mean_dice, 4),
        "mean_iou": round(mean_iou, 4),
        "pixel_accuracy": round(pixel_accuracy, 4),
        "per_class_metrics": class_metrics
    }

    if output_metrics_path is None:
        output_metrics_path = os.path.join(ROOT_DIR, "results", "unet_metrics.json")
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    with open(output_metrics_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved evaluation metrics to: {output_metrics_path}")

    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net on Unseen IDRiD Test Set")
    parser.add_argument("--model", type=str, default=os.path.join(ROOT_DIR, "models", "unet_retinal_lesion.pth"), help="Path to trained weights")
    parser.add_argument("--dataset", type=str, default=DEFAULT_IDRID_PATH, help="Path to IDRiD dataset")
    parser.add_argument("--splits", type=str, default=os.path.join(ROOT_DIR, "results", "unet_split_info.json"), help="Path to splits JSON")
    parser.add_argument("--img-size", type=int, default=512, help="Image size")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output", type=str, default=None, help="Output metrics JSON path")
    args = parser.parse_args()

    evaluate_unet_test_set(
        model_path=args.model,
        dataset_path=args.dataset,
        split_info_path=args.splits,
        batch_size=args.batch_size,
        img_size=args.img_size,
        output_metrics_path=args.output
    )
