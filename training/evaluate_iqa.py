"""
Evaluation Script for Retinal Image Quality Assessment (IQA) Model
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Evaluates:
  - Accuracy, Precision, Recall, Macro F1, Weighted F1
  - 3x3 Confusion Matrix across [Good Quality, Poor Quality, Unusable]
  - Safety-Critical False-Good Rate:
      Poor/Unusable incorrectly classified as Good (Unsafe Pass-Through to DR Classifier)
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.iqa_model import (
    RetinalQualityModel,
    build_iqa_model,
    find_iqa_checkpoint,
    get_device,
    get_iqa_transforms,
    QUALITY_CLASSES
)
from training.train_iqa import find_labeled_samples, RetinalQualityDataset


def evaluate_iqa_model(
    model_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    batch_size: int = 16,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluates RetinalQualityModel on test dataset.
    """
    if device is None:
        device = get_device()

    ckpt = find_iqa_checkpoint(model_path)
    if not ckpt:
        print("Error: IQA model checkpoint not found.")
        return {"error": "Checkpoint not found."}

    model = build_iqa_model(num_classes=3, pretrained=False, weights_path=ckpt, device=device)
    model.eval()

    if dataset_dir is None:
        dataset_dir = os.path.join(ROOT_DIR, "DR_Screening_Pipeline", "DR_Screening_Pipeline", "data")

    samples, has_labels = find_labeled_samples(dataset_dir)
    if not has_labels or len(samples) == 0:
        print("\n" + "=" * 60)
        print(" AI DR Screen — Retinal Image Quality (IQA) Evaluation")
        print("=" * 60)
        print("[!] EVALUATION NOTICE:")
        print("    No legitimate quality-labeled test dataset found.")
        print("    The existing project dataset contains DR severity annotations, not IQA labels.")
        print("    Real test evaluation requires an independent quality dataset (e.g., EyeQ Test Set).")
        print("=" * 60 + "\n")
        return {
            "status": "NO_LABELED_TESTSET",
            "message": "Evaluation requires a genuine quality-labeled test dataset."
        }

    # Evaluate on test split (last 15% with seed 42)
    np.random.seed(42)
    indices = np.arange(len(samples))
    np.random.shuffle(indices)
    n_test = max(1, int(0.15 * len(samples)))
    test_samples = [samples[i] for i in indices[-n_test:]]

    ds = RetinalQualityDataset(test_samples, transform=get_iqa_transforms(is_training=False))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metrics
    acc = float(accuracy_score(all_targets, all_preds))
    prec, rec, f1, support = precision_recall_fscore_support(all_targets, all_preds, labels=[0, 1, 2], zero_division=0)
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])

    # Safety-critical metric: Poor (class 1) or Unusable (class 2) predicted as Good (class 0)
    poor_as_good = int(cm[1, 0]) if cm.shape[0] > 1 else 0
    unusable_as_good = int(cm[2, 0]) if cm.shape[0] > 2 else 0
    total_unsafe_pass = poor_as_good + unusable_as_good

    print("\n" + "=" * 60)
    print(" AI DR Screen — Retinal Image Quality (IQA) Test Evaluation")
    print("=" * 60)
    print(f"Total Test Images:       {len(all_targets)}")
    print(f"Overall Accuracy:        {acc * 100:.2f}%")
    print(f"Macro F1-Score:          {macro_f1:.4f}")
    print("\nPer-Class Breakdown:")
    for i, cname in enumerate(QUALITY_CLASSES):
        print(f"  • {cname:<14}: Precision={prec[i]:.4f} | Recall={rec[i]:.4f} | F1={f1[i]:.4f} (N={support[i]})")

    print("\nConfusion Matrix [Rows: True, Cols: Predicted]:")
    print(f"                Good   Poor  Unusable")
    for i, cname in enumerate(["True Good", "True Poor", "True Unusable"]):
        row_str = " ".join([f"{cm[i, j]:6d}" for j in range(3)])
        print(f"  {cname:<14}: {row_str}")

    print("\nSafety-Critical Analysis (False Good Pass-Through):")
    print(f"  • Poor Quality  -> Predicted as Good: {poor_as_good} images")
    print(f"  • Unusable      -> Predicted as Good: {unusable_as_good} images")
    print(f"  • Total Unsafe Pass-Through:         {total_unsafe_pass} images")
    print("=" * 60 + "\n")

    return {
        "status": "SUCCESS",
        "total_test_samples": len(all_targets),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            QUALITY_CLASSES[i]: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i])
            } for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
        "safety_analysis": {
            "poor_to_good_count": poor_as_good,
            "unusable_to_good_count": unusable_as_good,
            "total_unsafe_passthrough": total_unsafe_pass
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Retinal Image Quality Assessment (IQA) Model")
    parser.add_argument("--model", type=str, default=None, help="Path to IQA model checkpoint")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to quality-labeled test dataset")
    args = parser.parse_args()

    evaluate_iqa_model(model_path=args.model, dataset_dir=args.data_dir)
