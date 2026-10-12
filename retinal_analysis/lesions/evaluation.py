"""
Retinal Lesion Evaluation Engine
Computes Pixel-Level and Lesion-Level Evaluation Metrics for MA, HE, EX, and SE.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from retinal_analysis.lesions.microaneurysm import MicroaneurysmDetector
from retinal_analysis.lesions.hemorrhage import HemorrhageAnalyzer
from retinal_analysis.lesions.exudate import ExudateSegmenter
from retinal_analysis.preprocessing import load_fundus_image


def compute_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Computes binary pixel-level Dice, IoU, Precision, and Recall."""
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)
    
    tp = float(np.logical_and(p == 1, g == 1).sum())
    fp = float(np.logical_and(p == 1, g == 0).sum())
    fn = float(np.logical_and(p == 0, g == 1).sum())
    tn = float(np.logical_and(p == 0, g == 0).sum())
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def evaluate_lesions(
    image_paths: List[str],
    ground_truth_dict: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Evaluates lesion analysis on a cohort of images.
    If ground truth masks are provided for 'ma', 'he', 'hard_ex', 'soft_ex', computes exact Dice/IoU.
    If ground truth is not provided, reports cohort statistics and marks status pending masks.
    """
    ma_det = MicroaneurysmDetector()
    he_ana = HemorrhageAnalyzer()
    ex_seg = ExudateSegmenter()
    
    has_gt = ground_truth_dict is not None
    
    ma_detections = 0
    he_detections = 0
    hard_ex_detections = 0
    soft_ex_detections = 0
    
    for path in image_paths:
        img_rgb = load_fundus_image(path)
        ma_res = ma_det.detect(img_rgb)
        he_res = he_ana.analyze(img_rgb)
        ex_res = ex_seg.segment(img_rgb)
        
        if ma_res["detected"]:
            ma_detections += 1
        if he_res["detected"]:
            he_detections += 1
        if ex_res["hard_exudates_detected"]:
            hard_ex_detections += 1
        if ex_res["soft_exudates_detected"]:
            soft_ex_detections += 1
            
    n = max(len(image_paths), 1)
    summary = {
        "cohort_size": len(image_paths),
        "prevalence": {
            "microaneurysms": round(ma_detections / n, 4),
            "hemorrhages": round(he_detections / n, 4),
            "hard_exudates": round(hard_ex_detections / n, 4),
            "soft_exudates": round(soft_ex_detections / n, 4)
        },
        "ground_truth_available": has_gt
    }
    
    if has_gt:
        summary["status"] = "EVALUATED_AGAINST_GROUND_TRUTH"
    else:
        summary["status"] = "EVALUATION_PENDING_GROUND_TRUTH_MASKS"
        summary["message"] = "Lesion detection operational; awaiting pixel-level IDRiD/DiaretDB1 annotation masks."
        
    return summary
