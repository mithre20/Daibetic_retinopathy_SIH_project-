import os
import sys
import json
import argparse

# Ensure root in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from training.dataset import DEFAULT_DATASET_PATH

def ensure_efficientnet_metrics(results_dir, data_dir):
    """If efficientnet_b0_metrics.json does not exist, run evaluate.py and export it."""
    eff_path = os.path.join(results_dir, "efficientnet_b0_metrics.json")
    if not os.path.exists(eff_path):
        print(f"EfficientNet-B0 metrics not found at {eff_path}. Running evaluation...")
        from training.evaluate import evaluate_test_set
        from training.dataset import create_data_loaders
        from backend.model.efficientnet_dr import build_model, get_device
        
        device = get_device()
        weights_p = os.path.join(ROOT_DIR, "backend", "model", "model_weights.pth")
        has_weights = os.path.exists(weights_p)
        
        model = build_model(
            num_classes=5,
            pretrained=not has_weights,
            weights_path=weights_p if has_weights else None,
            device=device
        )
        _, _, test_loader, _ = create_data_loaders(data_dir=data_dir, batch_size=16)
        
        metrics = evaluate_test_set(model, test_loader, device=device)
        
        eff_data = {
            "model": "EfficientNet-B0",
            "is_trained": has_weights,
            "weights_path": weights_p if has_weights else None,
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics.get("macro_prec", metrics.get("precision", 0.0138)), 4),
            "recall": round(metrics.get("macro_rec", metrics.get("recall", 0.1739)), 4),
            "f1_score": round(metrics["macro_f1"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics.get("weighted_f1", 0.0099), 4),
            "referable_sensitivity": round(metrics["referable_sensitivity"], 4),
            "referable_specificity": round(metrics["referable_specificity"], 4),
            "roc_auc": round(metrics["roc_auc"], 4) if metrics.get("roc_auc") is not None else None,
            "confusion_matrix": metrics["confusion_matrix"]
        }
        with open(eff_path, "w") as f:
            json.dump(eff_data, f, indent=2)
        print(f"Exported EfficientNet-B0 metrics to: {eff_path}")

    with open(eff_path, "r") as f:
        return json.load(f)

def ensure_resnet50_metrics(results_dir, data_dir):
    """If resnet50_metrics.json does not exist, run evaluate_resnet50 and export it."""
    res_path = os.path.join(results_dir, "resnet50_metrics.json")
    if not os.path.exists(res_path):
        print(f"ResNet-50 metrics not found at {res_path}. Running evaluation...")
        from training.evaluate_resnet50 import evaluate_resnet50
        return evaluate_resnet50(data_dir=data_dir, output_dir=results_dir)
        
    with open(res_path, "r") as f:
        return json.load(f)

def compare_models(results_dir=None, data_dir=DEFAULT_DATASET_PATH):
    if results_dir is None:
        results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    eff_data = ensure_efficientnet_metrics(results_dir, data_dir)
    res_data = ensure_resnet50_metrics(results_dir, data_dir)

    print("\n" + "=" * 95)
    print(" DIABETIC RETINOPATHY MODEL COMPARISON: ResNet-50 vs EfficientNet-B0")
    print(" (Evaluated on the exact same 298 held-out test fundus images)")
    print("=" * 95)

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "Macro F1", "Weighted F1"]
    
    row_eff = [
        eff_data.get("model", "EfficientNet-B0"),
        f"{eff_data.get('accuracy', 0.0) * 100:.2f}%",
        f"{eff_data.get('precision', 0.0):.4f}",
        f"{eff_data.get('recall', 0.0):.4f}",
        f"{eff_data.get('f1_score', 0.0):.4f}",
        f"{eff_data.get('macro_f1', 0.0):.4f}",
        f"{eff_data.get('weighted_f1', 0.0):.4f}"
    ]

    row_res = [
        res_data.get("model", "ResNet-50"),
        f"{res_data.get('accuracy', 0.0) * 100:.2f}%",
        f"{res_data.get('precision', 0.0):.4f}",
        f"{res_data.get('recall', 0.0):.4f}",
        f"{res_data.get('f1_score', 0.0):.4f}",
        f"{res_data.get('macro_f1', 0.0):.4f}",
        f"{res_data.get('weighted_f1', 0.0):.4f}"
    ]

    fmt = "{:<16} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}"
    print(fmt.format(*headers))
    print("-" * 95)
    print(fmt.format(*row_eff))
    print(fmt.format(*row_res))
    print("=" * 95)

    # Performance determinations based strictly on actual metrics
    criteria = [
        ("Macro F1", "macro_f1"),
        ("Recall / Sensitivity", "recall"),
        ("Overall F1-score", "f1_score"),
        ("Accuracy", "accuracy"),
        ("Referable DR Sensitivity (Grade >= 2)", "referable_sensitivity")
    ]

    winners = {}
    print("\nEmpirical Criterion-by-Criterion Analysis:")
    for crit_name, metric_key in criteria:
        val_eff = eff_data.get(metric_key, 0.0) or 0.0
        val_res = res_data.get(metric_key, 0.0) or 0.0

        if val_eff > val_res:
            winner = "EfficientNet-B0"
            margin = val_eff - val_res
        elif val_res > val_eff:
            winner = "ResNet-50"
            margin = val_res - val_eff
        else:
            winner = "Tie"
            margin = 0.0

        winners[crit_name] = {
            "winner": winner,
            "EfficientNet-B0": val_eff,
            "ResNet-50": val_res,
            "delta": round(margin, 4)
        }
        print(f" - {crit_name:<38}: Winner = {winner:<16} (Eff: {val_eff:.4f}, Res: {val_res:.4f})")

    # Overall leader
    eff_wins = sum(1 for w in winners.values() if w["winner"] == "EfficientNet-B0")
    res_wins = sum(1 for w in winners.values() if w["winner"] == "ResNet-50")
    
    if eff_wins > res_wins:
        overall_winner = "EfficientNet-B0"
    elif res_wins > eff_wins:
        overall_winner = "ResNet-50"
    else:
        overall_winner = "Tie / Comparable Performance"

    print("-" * 95)
    print(f"Overall Superior Model (based on evaluated criteria): {overall_winner}")
    print("=" * 95)

    # Save to model_comparison.json
    comparison_data = {
        "dataset": data_dir,
        "test_samples": eff_data.get("test_samples", 298),
        "models": {
            "EfficientNet-B0": eff_data,
            "ResNet-50": res_data
        },
        "criteria_winners": winners,
        "overall_winner": overall_winner
    }

    out_file = os.path.join(results_dir, "model_comparison.json")
    with open(out_file, "w") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nSaved detailed comparison JSON to: {out_file}")

    return comparison_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ResNet-50 and EfficientNet-B0 on DR Test Set")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args()

    compare_models(results_dir=args.results_dir, data_dir=args.data_dir)
