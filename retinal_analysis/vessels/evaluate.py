"""
Retinal Vessel Evaluation Module
Computes Dice, IoU, Sensitivity, Specificity, Precision, F1, and Pixel Accuracy.
"""

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np

from retinal_analysis.vessels.model import RetinalVesselSegmenter
from retinal_analysis.preprocessing import load_fundus_image, extract_retinal_fov_mask


def evaluate_vessel_segmentation(
    image_paths: List[str],
    ground_truth_mask_paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates retinal vessel segmentation on a cohort.
    If ground truth masks are provided, computes pixel-level segmentation metrics.
    """
    segmenter = RetinalVesselSegmenter()
    has_gt = ground_truth_mask_paths is not None and len(ground_truth_mask_paths) == len(image_paths)
    
    dices, ious = [], []
    sensitivities, specificities = [], []
    precisions, f1_scores = [], []
    accuracies = []
    densities = []
    
    for idx, path in enumerate(image_paths):
        img_rgb = load_fundus_image(path)
        res = segmenter.segment(img_rgb)
        densities.append(res["vessel_density_pct"])
        
        if has_gt:
            gt_mask = load_fundus_image(ground_truth_mask_paths[idx])
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[:, :, 0]
            gt_bin = (gt_mask > 127).astype(np.uint8)
            pred_bin = (res["binary_mask_original"] > 127).astype(np.uint8)
            
            # Restrict to retinal FOV
            fov = extract_retinal_fov_mask(img_rgb) > 0
            pred_f = pred_bin[fov]
            gt_f = gt_bin[fov]
            
            tp = float(np.logical_and(pred_f == 1, gt_f == 1).sum())
            fp = float(np.logical_and(pred_f == 1, gt_f == 0).sum())
            fn = float(np.logical_and(pred_f == 0, gt_f == 1).sum())
            tn = float(np.logical_and(pred_f == 0, gt_f == 0).sum())
            
            sens = tp / (tp + fn + 1e-6)
            spec = tn / (tn + fp + 1e-6)
            prec = tp / (tp + fp + 1e-6)
            f1 = (2 * prec * sens) / (prec + sens + 1e-6)
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
            iou = tp / (tp + fp + fn + 1e-6)
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
            
            sensitivities.append(sens)
            specificities.append(spec)
            precisions.append(prec)
            f1_scores.append(f1)
            accuracies.append(acc)
            ious.append(iou)
            dices.append(dice)
            
    summary = {
        "total_images": len(image_paths),
        "mean_vessel_density_pct": round(float(np.mean(densities)), 2) if densities else 0.0,
        "ground_truth_available": has_gt
    }
    
    if has_gt and len(dices) > 0:
        summary.update({
            "mean_dice": round(float(np.mean(dices)), 4),
            "mean_iou": round(float(np.mean(ious)), 4),
            "mean_sensitivity": round(float(np.mean(sensitivities)), 4),
            "mean_specificity": round(float(np.mean(specificities)), 4),
            "mean_precision": round(float(np.mean(precisions)), 4),
            "mean_f1": round(float(np.mean(f1_scores)), 4),
            "mean_pixel_accuracy": round(float(np.mean(accuracies)), 4),
            "status": "EVALUATED_AGAINST_GROUND_TRUTH"
        })
    else:
        summary.update({
            "status": "EVALUATION_PENDING_VESSEL_MASKS",
            "message": "Directional vessel extraction functional; awaiting benchmark manual vessel annotations (e.g. DRIVE/STARE)."
        })
        
    return summary
