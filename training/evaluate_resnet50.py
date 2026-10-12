import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg') # Headless rendering
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Ensure root directory in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.resnet50_model import ResNet50DRClassifier, build_resnet50, get_device, CLASS_NAMES
from training.dataset import create_data_loaders, DEFAULT_DATASET_PATH

def plot_confusion_matrix(cm, class_names, output_path):
    """Generates and saves a confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=int(cm[i, j]), va='center', ha='center',
                    color='white' if cm[i, j] > (cm.max() / 2) else 'black',
                    fontweight='bold', fontsize=11)

    fig.colorbar(cax)
    short_labels = ["No DR (0)", "Mild (1)", "Mod (2)", "Sev (3)", "PDR (4)"]
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=30, ha='left')
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels)
    ax.set_xlabel('Predicted ICDR Grade', fontweight='bold', labelpad=10)
    ax.set_ylabel('Ground Truth Grade', fontweight='bold', labelpad=10)
    ax.set_title('ResNet-50 DR Classification — Test Confusion Matrix', pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix visualization to: {output_path}")

def evaluate_resnet50(
    data_dir: str = DEFAULT_DATASET_PATH,
    weights_path: str = None,
    output_dir: str = None,
    batch_size: int = 16
):
    if not os.path.exists(data_dir):
        print("DATASET NOT FOUND")
        sys.exit(1)

    if weights_path is None:
        default_p = os.path.join(ROOT_DIR, "models", "resnet50_dr.pth")
        if os.path.exists(default_p):
            weights_path = default_p

    if weights_path is None or not os.path.exists(weights_path):
        print("RESNET50 WEIGHTS NOT FOUND")
        print("RESNET50 MODEL NOT TRAINED")
        # Proceed with untrained pretrained backbone for benchmark baseline reporting
        is_trained = False
    else:
        is_trained = True

    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print("=" * 70)
    print(" RESNET-50 TEST SET EVALUATION")
    print(f" Device: {device} | Weights: {weights_path if is_trained else 'None (Untrained Baseline)'}")
    print("=" * 70)

    # 1. Load exact test set
    _, _, test_loader, _ = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )

    # 2. Build model
    model = build_resnet50(
        num_classes=5,
        pretrained=not is_trained,
        weights_path=weights_path if is_trained else None,
        device=device
    )
    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Metrics
    acc = float(accuracy_score(all_targets, all_preds))
    macro_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_prec = float(precision_score(all_targets, all_preds, average="weighted", zero_division=0))
    macro_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_rec = float(recall_score(all_targets, all_preds, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

    # Referable DR (Grade >= 2) Sensitivity / Specificity
    binary_targets = (all_targets >= 2).astype(int)
    binary_preds = (all_preds >= 2).astype(int)
    ref_sensitivity = float(recall_score(binary_targets, binary_preds, pos_label=1, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(binary_targets, binary_preds, labels=[0, 1]).ravel()
    ref_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Per-class metrics
    per_class_prec = precision_score(all_targets, all_preds, average=None, labels=[0, 1, 2, 3, 4], zero_division=0).tolist()
    per_class_rec = recall_score(all_targets, all_preds, average=None, labels=[0, 1, 2, 3, 4], zero_division=0).tolist()
    per_class_f1 = f1_score(all_targets, all_preds, average=None, labels=[0, 1, 2, 3, 4], zero_division=0).tolist()

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4])

    # ROC-AUC (One-vs-Rest)
    try:
        auc = float(roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro"))
    except Exception:
        auc = None

    # Print summary
    print(f"Total Test Samples:        {len(all_targets)}")
    print(f"Accuracy:                  {acc * 100:.2f}%")
    print(f"Macro Precision:           {macro_prec:.4f}")
    print(f"Macro Recall (Sensitivity):{macro_rec:.4f}")
    print(f"Macro F1-score:            {macro_f1:.4f}")
    print(f"Weighted F1-score:         {weighted_f1:.4f}")
    if auc is not None:
        print(f"ROC-AUC (OvR Macro):       {auc:.4f}")
    print("-" * 50)
    print("Referable DR (Grade >= 2):")
    print(f"  Sensitivity (Recall):    {ref_sensitivity * 100:.2f}%")
    print(f"  Specificity:             {ref_specificity * 100:.2f}%")
    print(f"  TP: {tp} | FN: {fn} | TN: {tn} | FP: {fp}")
    print("-" * 50)
    print("Confusion Matrix:")
    print(cm)
    print("\nPer-Class Breakdown:")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, zero_division=0))

    # Save metrics JSON
    metrics_data = {
        "model": "ResNet50",
        "is_trained": is_trained,
        "weights_path": weights_path,
        "test_samples": len(all_targets),
        "accuracy": round(acc, 4),
        "precision": round(macro_prec, 4),
        "recall": round(macro_rec, 4),
        "f1_score": round(macro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "weighted_precision": round(weighted_prec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "referable_sensitivity": round(ref_sensitivity, 4),
        "referable_specificity": round(ref_specificity, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(per_class_prec[i], 4),
                "recall": round(per_class_rec[i], 4),
                "f1_score": round(per_class_f1[i], 4)
            }
            for i in range(5)
        }
    }

    metrics_out = os.path.join(output_dir, "resnet50_metrics.json")
    with open(metrics_out, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nSaved ResNet-50 metrics to: {metrics_out}")

    # Save confusion matrix image
    cm_img_out = os.path.join(output_dir, "resnet50_confusion_matrix.png")
    plot_confusion_matrix(cm, CLASS_NAMES, cm_img_out)

    return metrics_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet-50 on Unseen DR Test Set")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    evaluate_resnet50(
        data_dir=args.data_dir,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
