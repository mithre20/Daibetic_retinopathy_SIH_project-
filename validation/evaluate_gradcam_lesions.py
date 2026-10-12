"""
Part 9 — Clinically Meaningful Grad-CAM & Lesion Overlap Evaluation
AI DR Screen / Retina_Rovers: Explainable AI for Diabetic Retinopathy Screening in Rural India

Evaluates the spatial concordance and overlap between EfficientNet-B0 Grad-CAM attention
regions and anatomical retinal lesions (Microaneurysms, Hemorrhages, Exudates).

Scientific Principle:
Grad-CAM highlights image regions that contributed to the classifier prediction.
It does NOT prove that a specific lesion caused the diagnosis.

Outputs:
results/retinal_analysis/gradcam_lesion_overlap.json
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.model.efficientnet_dr import build_model as build_efficientnet, get_device
from backend.model.inference import get_inference_transforms
from inference.gradcam import GradCAM, calculate_gradcam_unet_overlap
from retinal_analysis.pipeline import analyze_retinal_structures


def evaluate_gradcam_lesion_overlap():
    print("=" * 80)
    print("PART 9: CLINICALLY MEANINGFUL GRAD-CAM & LESION OVERLAP EVALUATION")
    print("=" * 80)

    device = get_device()
    weights_path = os.path.join(ROOT_DIR, "models", "efficientnet_b0_dr.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    model = build_efficientnet(num_classes=5, pretrained=False, weights_path=weights_path, device=device)
    model.eval()

    # Locate convolutional feature extractor target layer for EfficientNet-B0
    target_layer = None
    if hasattr(model, "features"):
        target_layer = model.features[-1]
    elif hasattr(model, "backbone") and hasattr(model.backbone, "features"):
        target_layer = model.backbone.features[-1]

    if target_layer is None:
        raise AttributeError("Could not identify target convolutional layer for Grad-CAM.")

    gradcam_engine = GradCAM(model, target_layer)
    transform = get_inference_transforms()

    # Test images containing demonstrable lesions
    candidate_images = []
    test_dir = os.path.join(ROOT_DIR, "DR_Screening_Pipeline", "DR_Screening_Pipeline", "data", "test")
    if os.path.exists(test_dir):
        import pandas as pd
        classes_csv = os.path.join(test_dir, "_classes.csv")
        if os.path.exists(classes_csv):
            df = pd.read_csv(classes_csv)
            df.columns = [c.strip() for c in df.columns]
            # Select Moderate, Severe, Proliferative images
            dr_df = df[(df.get("Moderate", 0) == 1) | (df.get("Severe", 0) == 1) | (df.get("Proliferate_DR", 0) == 1)].head(10)
            for _, r in dr_df.iterrows():
                fpath = os.path.join(test_dir, r["filename"].strip())
                if os.path.exists(fpath):
                    candidate_images.append(fpath)

    # Fallback to local sample fundus images if test set subset is small
    sample_img = os.path.join(ROOT_DIR, "data", "sample_fundus.jpg")
    if os.path.exists(sample_img) and sample_img not in candidate_images:
        candidate_images.insert(0, sample_img)

    print(f"Evaluating Grad-CAM lesion concordance across {len(candidate_images)} sample images...")

    results_list = []
    pointing_hits = 0
    total_evaluated = 0

    from retinal_analysis.lesions.microaneurysm import MicroaneurysmDetector
    from retinal_analysis.lesions.hemorrhage import HemorrhageAnalyzer
    from retinal_analysis.lesions.exudate import ExudateSegmenter

    ma_detector = MicroaneurysmDetector()
    he_detector = HemorrhageAnalyzer()
    ex_segmenter = ExudateSegmenter()

    for idx, img_path in enumerate(candidate_images):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            tensor = transform(pil_img).unsqueeze(0).to(device)

            # 1. Forward model prediction
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_class = int(np.argmax(probs))

            # 2. Compute Grad-CAM heatmap for predicted class
            cam_output = gradcam_engine.generate_cam(tensor, target_class=pred_class)
            heatmap = cam_output["heatmap"]
            heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_img.size, resample=Image.Resampling.BILINEAR)) / 255.0

            # 3. Dedicated lesion detection masks
            ma_res = ma_detector.detect(img_path)
            he_res = he_detector.analyze(img_path)
            ex_res = ex_segmenter.segment(img_path)

            w, h = pil_img.size
            lesion_mask = np.zeros((h, w), dtype=np.uint8)

            # Class 1: Microaneurysms
            if "binary_mask_original" in ma_res and ma_res["binary_mask_original"] is not None:
                m = cv2.resize(ma_res["binary_mask_original"], (w, h), interpolation=cv2.INTER_NEAREST)
                lesion_mask[m > 0] = 1

            # Class 2: Hemorrhages
            if "binary_mask_original" in he_res and he_res["binary_mask_original"] is not None:
                h_m = cv2.resize(he_res["binary_mask_original"], (w, h), interpolation=cv2.INTER_NEAREST)
                lesion_mask[h_m > 0] = 2

            # Class 3: Hard Exudates
            if "hard_exudates_mask_orig" in ex_res and ex_res["hard_exudates_mask_orig"] is not None:
                ex_m = cv2.resize(ex_res["hard_exudates_mask_orig"], (w, h), interpolation=cv2.INTER_NEAREST)
                lesion_mask[ex_m > 0] = 3

            # Class 4: Soft Exudates
            if "soft_exudates_mask_orig" in ex_res and ex_res["soft_exudates_mask_orig"] is not None:
                se_m = cv2.resize(ex_res["soft_exudates_mask_orig"], (w, h), interpolation=cv2.INTER_NEAREST)
                lesion_mask[se_m > 0] = 4

            # 4. Calculate overlap metrics using GradCAM engine
            overlap_data = calculate_gradcam_unet_overlap(heatmap_resized, lesion_mask, threshold=0.5)

            # Pointing game: check if max Grad-CAM activation is inside or near a lesion
            max_y, max_x = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
            pointing_hit = bool(lesion_mask[max_y, max_x] > 0)
            if not pointing_hit:
                # Tolerance window of 25 pixels
                sub = lesion_mask[max(0, max_y-25):min(h, max_y+26), max(0, max_x-25):min(w, max_x+26)]
                pointing_hit = bool(np.any(sub > 0))

            if pointing_hit:
                pointing_hits += 1
            total_evaluated += 1

            overlap_data["image"] = os.path.basename(img_path)
            overlap_data["predicted_dr_grade"] = pred_class
            overlap_data["predicted_probability"] = round(float(probs[pred_class]), 4)
            overlap_data["pointing_game_hit"] = pointing_hit
            results_list.append(overlap_data)

        except Exception as e:
            print(f"Error evaluating {img_path}: {e}")
            continue

    pointing_accuracy = round(float((pointing_hits / max(1, total_evaluated)) * 100.0), 2)
    avg_iou = round(float(np.mean([r["overall_iou"] for r in results_list])) if results_list else 0.0, 4)
    avg_coverage = round(float(np.mean([r["overall_lesion_coverage_pct"] for r in results_list])) if results_list else 0.0, 2)

    gradcam_engine.remove_hooks()

    summary = {
        "evaluation_summary": {
            "total_images_evaluated": total_evaluated,
            "pointing_game_accuracy_pct": pointing_accuracy,
            "average_iou": avg_iou,
            "average_lesion_coverage_pct": avg_coverage,
            "pointing_hits": pointing_hits
        },
        "scientific_statement": "Grad-CAM highlights image regions that contributed to the classifier prediction.",
        "prohibited_claim": "Grad-CAM does NOT prove that lesions caused the diagnosis.",
        "per_image_results": results_list
    }

    out_dir = os.path.join(ROOT_DIR, "results", "retinal_analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "gradcam_lesion_overlap.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- GRAD-CAM & LESION OVERLAP EVALUATION COMPLETED ---")
    print(f"Images Evaluated:       {total_evaluated}")
    print(f"Pointing Game Accuracy: {pointing_accuracy}%")
    print(f"Mean Lesion Coverage:   {avg_coverage}%")
    print(f"Mean IoU:               {avg_iou}")
    print(f"Saved results to:       {out_file}")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    evaluate_gradcam_lesion_overlap()
