"""
CLI Inference Script for SHAP Feature-Level Explainability
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Usage:
    python inference/predict_shap.py --image path/to/fundus.jpg
    python inference/predict_shap.py --image path/to/fundus.jpg --target-class 2
    python inference/predict_shap.py --image path/to/fundus.jpg --output results/shap/
    python inference/predict_shap.py --image path/to/fundus.jpg --with-unet
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from explainability.shap_explainer import (
    generate_shap_explanation,
    find_model_checkpoint
)
from explainability.shap_visualizer import (
    create_positive_map,
    create_negative_map,
    create_combined_map,
    create_overlay
)
from explainability.shap_summary import (
    generate_human_readable_summary,
    compute_lesion_aware_shap_stats,
    compute_shap_unet_overlap,
    MEDICAL_SAFETY_DISCLAIMER
)


def run_shap_inference(
    image_path: str,
    target_class: Optional[int] = None,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    nsamples: int = 30,
    with_unet: bool = False
) -> Dict[str, Any]:
    """
    Executes end-to-end SHAP explanation on a single retinal fundus image.
    """
    if not os.path.isfile(image_path):
        print(f"Error: Retinal image not found at: {image_path}")
        sys.exit(1)

    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "results", "shap")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 55)
    print(" AI DR Screen — Generating SHAP Feature-Level Explanation")
    print("=" * 55)
    print(f"Input image:        {os.path.basename(image_path)}")
    if target_class is not None:
        print(f"Requested target:   Class {target_class}")
    else:
        print("Requested target:   Auto (Model Top Prediction)")
    print(f"Monte-Carlo samples: {nsamples}")

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image file: {e}")
        sys.exit(1)

    orig_w, orig_h = pil_img.size

    # 1. Run SHAP Explanation
    try:
        shap_res = generate_shap_explanation(
            image=pil_img,
            target_class=target_class,
            nsamples=nsamples,
            model_path=model_path
        )
    except Exception as e:
        print(f"\nUnable to generate SHAP explanation.\nDetails: {str(e)}")
        sys.exit(1)

    pred_class = shap_res["prediction_class"]
    pred_label = shap_res["prediction_label"]
    pred_prob = shap_res["prediction_probability"]
    tgt_class = shap_res["target_class"]
    tgt_label = shap_res["target_label"]
    pos_map = shap_res["positive_map"]
    neg_map = shap_res["negative_map"]
    comb_map = shap_res["combined_map"]
    stats = shap_res["shap_stats"]
    raw_shap = shap_res["shap_values"]

    # 2. Optional U-Net Lesion Integration
    lesion_stats = None
    overlap_info = None
    if with_unet:
        try:
            from inference.predict_unet import predict_retinal_lesions
            print("\nRunning U-Net Lesion Segmentation for lesion-aware attribution...")
            unet_res = predict_retinal_lesions(image_input=image_path, img_size=512)
            # Check if masks exist
            if "class_masks" in unet_res:
                lesion_stats = compute_lesion_aware_shap_stats(comb_map, unet_res["class_masks"])
            if "combined_mask" in unet_res:
                overlap_info = compute_shap_unet_overlap(comb_map, unet_res["combined_mask"])
        except Exception as ue:
            print(f"Warning: U-Net analysis skipped: {ue}")

    # 3. Generate Visualizations matching original dimensions
    pos_img = create_positive_map(pos_map, orig_size=(orig_w, orig_h))
    neg_img = create_negative_map(neg_map, orig_size=(orig_w, orig_h))
    comb_img = create_combined_map(comb_map, orig_size=(orig_w, orig_h))
    overlay_img = create_overlay(pil_img, comb_map, alpha=0.45, show_colorbar=True)

    # 4. Save Image Artifacts
    orig_save_path = os.path.join(output_dir, "original.png")
    pos_save_path = os.path.join(output_dir, "shap_positive.png")
    neg_save_path = os.path.join(output_dir, "shap_negative.png")
    overlay_save_path = os.path.join(output_dir, "shap_overlay.png")
    values_save_path = os.path.join(output_dir, "shap_values.npy")
    json_save_path = os.path.join(output_dir, "shap_result.json")

    pil_img.save(orig_save_path)
    pos_img.save(pos_save_path)
    neg_img.save(neg_save_path)
    overlay_img.save(overlay_save_path)

    # Save numerical attribution array
    np.save(values_save_path, raw_shap)

    # 5. Generate Clinical Narrative Summary
    summary = generate_human_readable_summary(
        prediction_class=pred_class,
        prediction_label=pred_label,
        confidence=pred_prob,
        target_class=tgt_class,
        target_label=tgt_label,
        shap_stats=stats,
        lesion_stats=lesion_stats,
        overlap_info=overlap_info
    )

    # 6. Save JSON Result
    result_data = {
        "prediction_class": pred_class,
        "prediction_label": pred_label,
        "prediction_probability": pred_prob,
        "target_class": tgt_class,
        "target_label": tgt_label,
        "explanation_method": "SHAP",
        "explainer": "shap.GradientExplainer (Expected Gradients)",
        "model": "EfficientNet-B0",
        "input_dimensions": [orig_w, orig_h],
        "probabilities": shap_res["probabilities"],
        "shap_statistics": stats,
        "saved_artifacts": {
            "original": os.path.abspath(orig_save_path),
            "shap_positive": os.path.abspath(pos_save_path),
            "shap_negative": os.path.abspath(neg_save_path),
            "shap_overlay": os.path.abspath(overlay_save_path),
            "shap_values_npy": os.path.abspath(values_save_path)
        },
        "attribution_dimensions": list(raw_shap.shape),
        "narrative_summary": summary,
        "disclaimer": MEDICAL_SAFETY_DISCLAIMER
    }

    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    # Print summary output
    print(summary["formatted_text"])
    print(f"\nSaved Artifacts:")
    print(f"  • {orig_save_path}")
    print(f"  • {pos_save_path}")
    print(f"  • {neg_save_path}")
    print(f"  • {overlay_save_path}")
    print(f"  • {values_save_path} (Shape: {raw_shap.shape})")
    print(f"  • {json_save_path}")
    print("=" * 55 + "\n")

    return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP Feature-Level Explanation for Retinal Fundus Image")
    parser.add_argument("--image", type=str, required=True, help="Path to retinal fundus image")
    parser.add_argument("--target-class", type=int, default=None, choices=[0, 1, 2, 3, 4], help="Target DR class to explain (0-4)")
    parser.add_argument("--output", type=str, default=None, help="Directory to save explanation artifacts")
    parser.add_argument("--model", type=str, default=None, help="Path to custom EfficientNet-B0 checkpoint")
    parser.add_argument("--samples", type=int, default=30, help="Number of path samples for GradientExplainer")
    parser.add_argument("--with-unet", action="store_true", help="Include complementary U-Net lesion segmentation overlap")
    args = parser.parse_args()

    run_shap_inference(
        image_path=args.image,
        target_class=args.target_class,
        output_dir=args.output,
        model_path=args.model,
        nsamples=args.samples,
        with_unet=args.with_unet
    )
