"""
Optic Disc Evaluation Module
Computes Center Localization Error, Tolerance Accuracy (<= 1.0 R_OD), and Dice/IoU.
"""

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

from retinal_analysis.optic_disc.model import OpticDiscDetector
from retinal_analysis.preprocessing import load_fundus_image


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes Dice similarity coefficient between two binary masks."""
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    total = (mask1 > 0).sum() + (mask2 > 0).sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)


def evaluate_optic_disc_dataset(
    image_paths: List[str],
    ground_truth_annotations: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Evaluates Optic Disc localization on a set of images.
    If ground truth is provided, calculates localization errors and IoU.
    If ground truth is unavailable, reports execution statistics and flags evaluation status.
    """
    detector = OpticDiscDetector()
    results = []
    
    has_gt = ground_truth_annotations is not None and len(ground_truth_annotations) == len(image_paths)
    center_errors = []
    normalized_errors = []
    ious = []
    dices = []
    detected_count = 0
    
    for idx, path in enumerate(image_paths):
        img_rgb = load_fundus_image(path)
        pred = detector.detect(img_rgb)
        if pred["detected"]:
            detected_count += 1
            
        if has_gt:
            gt = ground_truth_annotations[idx]
            gt_cx = gt["center_x"]
            gt_cy = gt["center_y"]
            gt_r = gt.get("radius", pred["radius"])
            
            # Euclidean distance
            dist = float(np.sqrt((pred["center"][0] - gt_cx)**2 + (pred["center"][1] - gt_cy)**2))
            center_errors.append(dist)
            
            # Error normalized by ground truth radius
            norm_err = dist / max(gt_r, 1e-3)
            normalized_errors.append(norm_err)
            
            # Binary masks for IoU/Dice
            h, w = img_rgb.shape[:2]
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(pred_mask, (int(pred["center"][0]), int(pred["center"][1])), int(pred["radius"]), 255, -1)
            
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(gt_mask, (int(gt_cx), int(gt_cy)), int(gt_r), 255, -1)
            
            ious.append(compute_iou(pred_mask, gt_mask))
            dices.append(compute_dice(pred_mask, gt_mask))
            
    summary = {
        "total_images": len(image_paths),
        "detected_count": detected_count,
        "detection_rate": round(detected_count / max(len(image_paths), 1), 4),
        "ground_truth_available": has_gt
    }
    
    if has_gt and len(center_errors) > 0:
        tolerance_rate = float(np.mean([err <= 1.0 for err in normalized_errors]))
        summary.update({
            "mean_center_error_px": round(float(np.mean(center_errors)), 2),
            "median_center_error_px": round(float(np.median(center_errors)), 2),
            "mean_normalized_error": round(float(np.mean(normalized_errors)), 4),
            "tolerance_success_rate": round(tolerance_rate, 4),
            "mean_iou": round(float(np.mean(ious)), 4),
            "mean_dice": round(float(np.mean(dices)), 4),
            "status": "EVALUATED_AGAINST_GROUND_TRUTH"
        })
    else:
        summary.update({
            "status": "EVALUATION_PENDING_GROUND_TRUTH_MASKS",
            "message": "Algorithmic localization operational; awaiting pixel-level bounding-box ground truth dataset."
        })
        
    return summary
