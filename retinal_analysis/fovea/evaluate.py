"""
Fovea Evaluation Module
Computes Mean Localization Error, Median Error, and Tolerance Percentage (<= 1.0 R_OD).
"""

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np

from retinal_analysis.fovea.model import FoveaLocalizer
from retinal_analysis.optic_disc.model import OpticDiscDetector
from retinal_analysis.preprocessing import load_fundus_image


def evaluate_fovea_dataset(
    image_paths: List[str],
    ground_truth_annotations: Optional[List[Dict[str, Any]]] = None,
    tolerance_radius_px: float = 40.0
) -> Dict[str, Any]:
    """
    Evaluates Fovea localization on a set of images.
    If ground truth is provided, calculates mean/median distance errors and tolerance accuracy.
    If ground truth is unavailable, reports execution statistics and flags evaluation status.
    """
    localizer = FoveaLocalizer()
    od_detector = OpticDiscDetector()
    
    has_gt = ground_truth_annotations is not None and len(ground_truth_annotations) == len(image_paths)
    errors = []
    normalized_errors = []
    within_tolerance_count = 0
    detected_count = 0
    
    for idx, path in enumerate(image_paths):
        img_rgb = load_fundus_image(path)
        od_res = od_detector.detect(img_rgb)
        fovea_res = localizer.locate(img_rgb, optic_disc_result=od_res)
        
        if fovea_res["detected"]:
            detected_count += 1
            
        if has_gt:
            gt = ground_truth_annotations[idx]
            gt_fx = gt["fovea_x"]
            gt_fy = gt["fovea_y"]
            
            # Euclidean distance error
            dist = float(np.sqrt((fovea_res["center"][0] - gt_fx)**2 + (fovea_res["center"][1] - gt_fy)**2))
            errors.append(dist)
            
            # Clinical tolerance based on optic disc radius (approx 1 OD radius)
            od_r = max(od_res["radius"], tolerance_radius_px)
            norm_dist = dist / od_r
            normalized_errors.append(norm_dist)
            
            if dist <= od_r:
                within_tolerance_count += 1
                
    summary = {
        "total_images": len(image_paths),
        "detected_count": detected_count,
        "detection_rate": round(detected_count / max(len(image_paths), 1), 4),
        "ground_truth_available": has_gt
    }
    
    if has_gt and len(errors) > 0:
        summary.update({
            "mean_localization_error_px": round(float(np.mean(errors)), 2),
            "median_localization_error_px": round(float(np.median(errors)), 2),
            "mean_normalized_error_od_radii": round(float(np.mean(normalized_errors)), 4),
            "tolerance_percentage": round(within_tolerance_count / len(errors), 4),
            "tolerance_definition": "<= 1.0 Optic Disc Radius (~1.5 mm on retina)",
            "status": "EVALUATED_AGAINST_GROUND_TRUTH"
        })
    else:
        summary.update({
            "status": "EVALUATION_PENDING_GROUND_TRUTH_COORDINATES",
            "tolerance_definition": "<= 1.0 Optic Disc Radius (~1.5 mm on retina)",
            "message": "Fovea localization operational; awaiting expert manual fovea landmark annotations."
        })
        
    return summary
