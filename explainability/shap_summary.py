"""
Clinical and Research Summary Generator for SHAP Feature-Level Attributions
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India
"""

from typing import Dict, Any, Optional
import numpy as np

MEDICAL_SAFETY_DISCLAIMER = (
    "SHAP visualizations explain which image regions influenced the AI classifier's prediction. "
    "They are not a medical diagnosis and do not prove the presence or absence of a retinal lesion. "
    "Results should be reviewed by a qualified healthcare professional."
)


def compute_lesion_aware_shap_stats(
    shap_2d_attr: np.ndarray,
    unet_masks: Optional[Dict[str, np.ndarray]]
) -> Dict[str, Any]:
    """
    Calculates empirical SHAP attribution statistics strictly within predicted U-Net lesion masks.
    Only real pixel values are evaluated; no fake numbers are generated.
    """
    if not unet_masks:
        return {}

    lesion_stats = {}
    h, w = shap_2d_attr.shape

    for lesion_name, mask in unet_masks.items():
        if mask is None:
            continue

        # Ensure mask is boolean and matches attribution shape
        if mask.shape != (h, w):
            # Resize binary mask to match SHAP dimensions
            from PIL import Image
            mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
            mask_resized = np.array(mask_pil.resize((w, h), Image.NEAREST)) > 127
        else:
            mask_resized = mask > 0

        pixel_count = int(np.sum(mask_resized))
        if pixel_count > 0:
            vals_in_lesion = shap_2d_attr[mask_resized]
            mean_contrib = float(np.mean(vals_in_lesion))
            max_contrib = float(np.max(vals_in_lesion))
            min_contrib = float(np.min(vals_in_lesion))
            pos_fraction = float(np.mean(vals_in_lesion > 0))

            lesion_stats[lesion_name] = {
                "available": True,
                "pixel_count": pixel_count,
                "mean_contribution": round(mean_contrib, 6),
                "max_contribution": round(max_contrib, 6),
                "min_contribution": round(min_contrib, 6),
                "positive_attribution_ratio": round(pos_fraction, 4),
                "status": "Positive contribution" if mean_contrib > 0 else "Negative / neutral contribution"
            }
        else:
            lesion_stats[lesion_name] = {
                "available": False,
                "pixel_count": 0,
                "message": "No lesion region segmented by U-Net"
            }

    return lesion_stats


def compute_shap_unet_overlap(
    shap_attr: np.ndarray,
    binary_lesion_mask: np.ndarray,
    shap_threshold_percentile: float = 85.0
) -> Dict[str, Any]:
    """
    Computes empirical spatial overlap (Intersection over Union / Dice) between
    high-attribution SHAP regions and U-Net lesion segmentations.
    """
    if binary_lesion_mask is None or np.sum(binary_lesion_mask) == 0:
        return {
            "overlap_available": False,
            "message": "No U-Net lesion segments available for spatial overlap calculation."
        }

    from PIL import Image
    h, w = shap_attr.shape
    if binary_lesion_mask.shape != (h, w):
        mask_pil = Image.fromarray((binary_lesion_mask > 0).astype(np.uint8) * 255)
        unet_bin = np.array(mask_pil.resize((w, h), Image.NEAREST)) > 127
    else:
        unet_bin = binary_lesion_mask > 0

    # High positive attribution mask
    pos_attr = np.maximum(shap_attr, 0.0)
    cutoff = np.percentile(pos_attr, shap_threshold_percentile)
    shap_bin = pos_attr >= cutoff

    intersection = np.logical_and(shap_bin, unet_bin).sum()
    union = np.logical_or(shap_bin, unet_bin).sum()
    iou = float(intersection / (union + 1e-8))
    dice = float((2.0 * intersection) / (shap_bin.sum() + unet_bin.sum() + 1e-8))

    return {
        "overlap_available": True,
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "overlap_percentage": round(dice * 100, 2),
        "shap_high_attr_pixels": int(shap_bin.sum()),
        "unet_lesion_pixels": int(unet_bin.sum()),
        "intersecting_pixels": int(intersection),
        "note": "Overlap metric represents empirical spatial intersection between model attribution and U-Net segmentation, not a clinical correctness claim."
    }


def generate_human_readable_summary(
    prediction_class: int,
    prediction_label: str,
    confidence: float,
    target_class: int,
    target_label: str,
    shap_stats: Dict[str, Any],
    lesion_stats: Optional[Dict[str, Any]] = None,
    overlap_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Synthesizes numerical SHAP attribution and optional U-Net evidence into a concise,
    clinically safe, and honest human-readable explanation.
    """
    is_target_same_as_pred = (prediction_class == target_class)

    # Narrative explanation
    if is_target_same_as_pred:
        overview = (
            f"Model Prediction: {prediction_label} (Confidence: {confidence * 100:.1f}%).\n"
            f"The SHAP explainer decomposed pixel-level evidence supporting and opposing this classification."
        )
    else:
        overview = (
            f"Model Prediction: {prediction_label} (Confidence: {confidence * 100:.1f}%).\n"
            f"Target Explained Class: {target_label} (Class {target_class}).\n"
            f"The SHAP explainer analyzed why the model did or did not classify this image as {target_label}."
        )

    # Key contributing region text
    pos_summary = (
        "Highlighted warm regions indicate retinal areas that contributed positively, "
        f"pushing the classifier toward predicting {target_label}."
    )
    neg_summary = (
        "Contrasting cool regions contributed negatively, opposing this classification "
        "or pulling the decision toward alternate DR grades."
    )

    # Structure breakdown
    breakdown = {
        "prediction": prediction_label,
        "prediction_confidence": f"{confidence * 100:.1f}%",
        "target_class_explained": target_label,
        "positive_contribution_narrative": pos_summary,
        "negative_contribution_narrative": neg_summary,
        "sparsity": f"{shap_stats.get('sparsity', 0.0) * 100:.1f}% neutral retinal background",
        "computation_time": f"{shap_stats.get('computation_time_seconds', 0.0):.2f}s",
        "disclaimer": MEDICAL_SAFETY_DISCLAIMER
    }

    if lesion_stats:
        breakdown["lesion_aware_contributions"] = lesion_stats

    if overlap_info and overlap_info.get("overlap_available"):
        breakdown["shap_unet_overlap"] = overlap_info

    # Formatted plain-text block
    text_lines = [
        "=" * 50,
        " AI DR SCREEN — SHAP FEATURE-LEVEL EXPLANATION",
        "=" * 50,
        f"Prediction:            {prediction_label} (Class {prediction_class})",
        f"Confidence:            {confidence * 100:.1f}%",
        f"Explained Target:      {target_label} (Class {target_class})",
        "",
        "SHAP Feature Attribution Breakdown:",
        f"  • Positive Evidence:  {pos_summary}",
        f"  • Negative Evidence:  {neg_summary}",
        f"  • Background Ratio:   {breakdown['sparsity']}",
        f"  • Compute Duration:   {breakdown['computation_time']}",
        ""
    ]

    if lesion_stats:
        text_lines.append("Lesion-Aware Regional Attribution (via U-Net Segmentation):")
        for lname, linfo in lesion_stats.items():
            disp_name = lname.replace("_", " ").title()
            if linfo.get("available"):
                text_lines.append(
                    f"  • {disp_name:<18}: Mean SHAP = {linfo['mean_contribution']:+.6f} ({linfo['status']})"
                )
            else:
                text_lines.append(f"  • {disp_name:<18}: Not available (no segmented lesions)")
        text_lines.append("")

    if overlap_info and overlap_info.get("overlap_available"):
        text_lines.append(f"SHAP / U-Net Lesion Spatial Overlap (Dice): {overlap_info['overlap_percentage']}%")
        text_lines.append("")

    text_lines.extend([
        "Medical Safety Disclaimer:",
        f"  {MEDICAL_SAFETY_DISCLAIMER}",
        "=" * 50
    ])

    breakdown["formatted_text"] = "\n".join(text_lines)
    return breakdown
