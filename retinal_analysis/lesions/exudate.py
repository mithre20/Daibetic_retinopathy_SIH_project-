"""
Dedicated Exudate Segmentation Architecture
Multi-threshold luminance segmentation for Hard Exudates (lipid deposits) and Soft Exudates (cotton wool spots).
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

from retinal_analysis.preprocessing import (
    preprocess_fundus,
    load_fundus_image,
    apply_clahe,
    extract_retinal_fov_mask
)
from retinal_analysis.optic_disc.model import OpticDiscDetector


class ExudateSegmenter:
    """
    Exudate segmentation engine differentiating Hard Exudates from Soft Exudates.
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.od_detector = OpticDiscDetector(target_size=target_size)

    def segment(
        self,
        image_input: Any,
        optic_disc_mask: Optional[np.ndarray] = None,
        unet_hard_ex_prob: Optional[np.ndarray] = None,
        unet_soft_ex_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Segments Hard and Soft Exudates.

        Returns:
            Dict containing:
                - hard_exudates_detected: bool
                - soft_exudates_detected: bool
                - hard_exudates_mask: uint8 array [H, W] at target_size
                - soft_exudates_mask: uint8 array [H, W] at target_size
                - hard_exudates_mask_orig: uint8 array at original size
                - soft_exudates_mask_orig: uint8 array at original size
                - hard_exudates_area_pct: float
                - soft_exudates_area_pct: float
                - confidence: float
        """
        prep = preprocess_fundus(image_input, target_size=self.target_size)
        resized_rgb = prep["resized_rgb"]
        green = prep["green_channel"]
        fov_mask = prep["fov_mask"]
        orig_shape = prep["orig_shape"]
        h, w = self.target_size
        
        # 1. Suppress Optic Disc (as OD is bright yellow/orange and would otherwise look like giant exudate)
        if optic_disc_mask is None:
            od_res = self.od_detector.detect(prep["original_rgb"])
            optic_disc_mask = od_res["mask"]
        elif optic_disc_mask.shape != (h, w):
            optic_disc_mask = cv2.resize(optic_disc_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        # Dilate optic disc generously to mask the entire physiological cup and peripapillary halo
        od_suppress = cv2.dilate(optic_disc_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        safe_fov = cv2.erode(fov_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        
        # 2. Hard Exudates Segmentation
        # Hard Exudates are bright yellow: high red, high green, sharp local edges
        red = resized_rgb[:, :, 0]
        clahe_g = apply_clahe(green, clip_limit=2.5)
        
        # White Top-Hat on green channel highlights bright structures
        k_bright = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        white_tophat = cv2.morphologyEx(clahe_g, cv2.MORPH_TOPHAT, k_bright)
        
        # Candidate bright field outside optic disc
        bright_field = cv2.bitwise_and(white_tophat, white_tophat, mask=safe_fov)
        bright_field[od_suppress > 0] = 0
        
        # High threshold for hard exudates (distinct, high contrast)
        safe_pixels = bright_field[safe_fov > 0]
        if len(safe_pixels) > 0 and np.max(safe_pixels) > 0:
            he_thresh_val = np.percentile(safe_pixels[safe_pixels > 0], 92.0) if np.any(safe_pixels > 0) else 35
            he_thresh_val = max(he_thresh_val, 28)
        else:
            he_thresh_val = 40
            
        _, hard_mask = cv2.threshold(bright_field, he_thresh_val, 255, cv2.THRESH_BINARY)
        
        # Filter tiny 1-pixel noise and oversize components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hard_mask)
        final_hard_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 4 <= area <= 1500:
                final_hard_mask[labels == i] = 255
                
        # 3. Soft Exudates (Cotton Wool Spots)
        # Larger, pale-white, fuzzy boundaries, lower edge gradient
        # Medium-scale morphological opening on green channel
        k_soft = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        soft_tophat = cv2.morphologyEx(clahe_g, cv2.MORPH_TOPHAT, k_soft)
        soft_field = cv2.bitwise_and(soft_tophat, soft_tophat, mask=safe_fov)
        soft_field[od_suppress > 0] = 0
        soft_field[final_hard_mask > 0] = 0  # mutually exclusive
        
        if len(safe_pixels) > 0 and np.max(safe_pixels) > 0:
            se_thresh_val = np.percentile(safe_pixels[safe_pixels > 0], 85.0) if np.any(safe_pixels > 0) else 25
            se_thresh_val = max(se_thresh_val, 22)
        else:
            se_thresh_val = 30
            
        _, raw_soft_mask = cv2.threshold(soft_field, se_thresh_val, 255, cv2.THRESH_BINARY)
        
        # Soft exudates must be larger in area than punctate hard exudates (area >= 40 px)
        num_labels_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(raw_soft_mask)
        final_soft_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, num_labels_s):
            area = stats_s[i, cv2.CC_STAT_AREA]
            if 40 <= area <= 3000:
                final_soft_mask[labels_s == i] = 255
                
        # Optional fusion with U-Net outputs if available
        if unet_hard_ex_prob is not None:
            if unet_hard_ex_prob.shape != (h, w):
                unet_hard_ex_prob = cv2.resize(unet_hard_ex_prob, (w, h))
            u_hard = (unet_hard_ex_prob >= 0.35).astype(np.uint8) * 255
            final_hard_mask = cv2.bitwise_or(final_hard_mask, u_hard)
            final_hard_mask[od_suppress > 0] = 0
            
        if unet_soft_ex_prob is not None:
            if unet_soft_ex_prob.shape != (h, w):
                unet_soft_ex_prob = cv2.resize(unet_soft_ex_prob, (w, h))
            u_soft = (unet_soft_ex_prob >= 0.35).astype(np.uint8) * 255
            final_soft_mask = cv2.bitwise_or(final_soft_mask, u_soft)
            final_soft_mask[od_suppress > 0] = 0
            
        # Calculate coverage percentages
        retina_area = float((safe_fov > 0).sum())
        hard_area_pct = float((final_hard_mask > 0).sum()) / retina_area * 100.0 if retina_area > 0 else 0.0
        soft_area_pct = float((final_soft_mask > 0).sum()) / retina_area * 100.0 if retina_area > 0 else 0.0
        
        detected_hard = (final_hard_mask > 0).sum() > 15
        detected_soft = (final_soft_mask > 0).sum() > 40
        
        # Scale to original resolution
        hard_orig = cv2.resize(final_hard_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        soft_orig = cv2.resize(final_soft_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return {
            "hard_exudates_detected": bool(detected_hard),
            "soft_exudates_detected": bool(detected_soft),
            "hard_exudates_mask": final_hard_mask,
            "soft_exudates_mask": final_soft_mask,
            "hard_exudates_mask_original": hard_orig,
            "soft_exudates_mask_original": soft_orig,
            "hard_exudates_area_pct": round(hard_area_pct, 3),
            "soft_exudates_area_pct": round(soft_area_pct, 3),
            "confidence": 0.88 if (detected_hard or detected_soft) else 0.94,
            "clinical_significance": {
                "hard_exudates": "Lipoprotein extravasation indicating blood-retinal barrier breakdown (macular edema risk)",
                "soft_exudates": "Cotton-wool spots indicating retinal precapillary arteriolar ischemia / infarction"
            }
        }


def segment_exudates(
    image_input: Any,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Production wrapper saving exudate outputs to disk."""
    if output_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = os.path.join(root_dir, "results", "retinal_analysis", "exudates")
    os.makedirs(output_dir, exist_ok=True)
    
    from retinal_analysis.visualization import draw_lesion_overlay, save_image
    
    orig_rgb = load_fundus_image(image_input)
    segmenter = ExudateSegmenter()
    result = segmenter.segment(orig_rgb)
    
    orig_path = os.path.join(output_dir, "original.png")
    save_image(orig_rgb, orig_path)
    
    hard_mask_path = os.path.join(output_dir, "hard_exudates_mask.png")
    save_image(result["hard_exudates_mask_original"], hard_mask_path)
    
    soft_mask_path = os.path.join(output_dir, "soft_exudates_mask.png")
    save_image(result["soft_exudates_mask_original"], soft_mask_path)
    
    overlay_rgb = draw_lesion_overlay(orig_rgb, {
        "hard_exudate": result["hard_exudates_mask_original"],
        "soft_exudate": result["soft_exudates_mask_original"]
    })
    overlay_path = os.path.join(output_dir, "overlay.png")
    save_image(overlay_rgb, overlay_path)
    
    result["saved_paths"] = {
        "original": orig_path,
        "hard_exudates_mask": hard_mask_path,
        "soft_exudates_mask": soft_mask_path,
        "overlay": overlay_path
    }
    result["overlay_rgb"] = overlay_rgb
    return result
