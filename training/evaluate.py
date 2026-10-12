import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Ensure root in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.model.efficientnet_dr import EfficientNetDRClassifier, build_model, get_device, CLASS_NAMES
from training.dataset import create_data_loaders, DEFAULT_DATASET_PATH

def evaluate_test_set(model, test_loader, device=None):
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []

    print("\nRunning evaluation on unseen test split...")
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

    # 1. Overall metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    # 2. Referable DR Sensitivity & Specificity (Clinical primary target: Grade >= 2)
    binary_targets = (all_targets >= 2).astype(int)
    binary_preds = (all_preds >= 2).astype(int)
    ref_sensitivity = recall_score(binary_targets, binary_preds, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(binary_targets, binary_preds, labels=[0, 1]).ravel()
    ref_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 3. Multiclass ROC-AUC
    try:
        # Check if all classes present
        present_classes = np.unique(all_targets)
        if len(present_classes) == 5:
            auc_ovr = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
        else:
            auc_ovr = roc_auc_score(all_targets, all_probs[:, present_classes], multi_class="ovr", average="macro", labels=present_classes)
    except Exception as e:
        auc_ovr = float("nan")

    # Display results
    print("=" * 70)
    print(" CLINICAL PERFORMANCE EVALUATION REPORT (Held-Out Test Set)")
    print("=" * 70)
    print(f" Total Unseen Test Samples: {len(all_targets)}")
    print(f" Overall Accuracy:         {acc * 100:.2f}%")
    print(f" Macro Precision:          {macro_prec:.4f}")
    print(f" Macro Recall (Sensitivity):{macro_rec:.4f}")
    print(f" Macro F1-Score:           {macro_f1:.4f}")
    print(f" Weighted F1-Score:        {weighted_f1:.4f}")
    if not np.isnan(auc_ovr):
        print(f" Multiclass ROC-AUC (OvR): {auc_ovr:.4f}")
    print("-" * 70)
    print(" [CLINICAL SCREENING PRIORITY: REFERABLE DR (Grade >= 2)]")
    print(f" Referable Sensitivity (Recall): {ref_sensitivity * 100:.2f}%  (Target: >= 90%)")
    print(f" Referable Specificity:          {ref_specificity * 100:.2f}%  (Target: >= 85%)")
    print(f" True Positives (TP): {tp} | False Negatives (FN): {fn}")
    print(f" True Negatives (TN): {tn} | False Positives (FP): {fp}")
    print("-" * 70)
    print(" Confusion Matrix (Rows = Actual, Columns = Predicted):")
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4])
    header = "       " + "".join([f"G{i:>6}" for i in range(5)])
    print(header)
    for i, row in enumerate(cm):
        print(f"  G{i}   " + "".join([f"{val:>6d}" for val in row]))

    print("-" * 70)
    print(" Per-Class Clinical Classification Breakdown:")
    report = classification_report(
        all_targets,
        all_preds,
        labels=[0, 1, 2, 3, 4],
        target_names=CLASS_NAMES,
        zero_division=0
    )
    print(report)
    print("=" * 70)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "referable_sensitivity": ref_sensitivity,
        "referable_specificity": ref_specificity,
        "roc_auc": auc_ovr,
        "confusion_matrix": cm.tolist()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 on DR Test Set")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATASET_PATH, help="Path to dataset")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to trained .pth weights")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    args = parser.parse_args()

    # Load data
    _, _, test_loader, _ = create_data_loaders(data_dir=args.data_dir, batch_size=args.batch_size)

    # Load model
    device = get_device()
    model = build_model(
        num_classes=5,
        pretrained=args.weights_path is None,
        weights_path=args.weights_path,
        device=device
    )

    evaluate_test_set(model, test_loader, device=device)
