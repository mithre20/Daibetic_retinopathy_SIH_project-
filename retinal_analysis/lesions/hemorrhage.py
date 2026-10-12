"""
Dedicated Hemorrhage Detection and Classification Architecture
Segments dot/blot and flame-shaped hemorrhages, extracts lesion-level metrics, and classifies presence/absence.
"""

import os
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import cv2

from retinal_analysis.preprocessing import (
    preprocess_fundus,
    load_fundus_image,
    apply_clahe,
    extract_retinal_fov_mask
)
from retinal_analysis.vessels.model import RetinalVesselSegmenter
from retinal_analysis.optic_disc.model import OpticDiscDetector


class HemorrhageAnalyzer:
    """
    Intraretinal hemorrhage detector and classifier.
    Extracts red lesions, suppresses vessels and optic disc, and performs quadrant analysis.
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.vessel_segmenter = RetinalVesselSegmenter(target_size=target_size)
        self.od_detector = OpticDiscDetector(target_size=target_size)

    def analyze(
        self,
        image_input: Any,
        vessel_mask: Optional[np.ndarray] = None,
        optic_disc_mask: Optional[np.ndarray] = None,
        unet_he_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyzes retinal hemorrhages.

        Returns:
            Dict containing:
                - classification: "Hemorrhages Present" or "Hemorrhages Absent"
                - detected: bool
                - hemorrhage_count: int
                - total_area_px: int
                - area_pct: float
                - quadrant_distribution: Dict with count/area per quadrant
                - binary_mask: uint8 array [H, W] at target_size
                - binary_mask_original: uint8 array at original dimensions
                - confidence: float
        """
        prep = preprocess_fundus(image_input, target_size=self.target_size)
        resized_rgb = prep["resized_rgb"]
        green = prep["green_channel"]
        fov_mask = prep["fov_mask"]
        orig_shape = prep["orig_shape"]
        h, w = self.target_size
        
        # 1. Obtain Vessel Mask and Optic Disc Mask for suppression
        if vessel_mask is None:
            v_res = self.vessel_segmenter.segment(prep["original_rgb"])
            vessel_mask = v_res["binary_mask"]
        elif vessel_mask.shape != (h, w):
            vessel_mask = cv2.resize(vessel_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        if optic_disc_mask is None:
            od_res = self.od_detector.detect(prep["original_rgb"])
            optic_disc_mask = od_res["mask"]
        elif optic_disc_mask.shape != (h, w):
            optic_disc_mask = cv2.resize(optic_disc_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        vessel_dilated = cv2.dilate(vessel_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        od_dilated = cv2.dilate(optic_disc_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        safe_fov = cv2.erode(fov_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        
        suppression = cv2.bitwise_or(vessel_dilated, od_dilated)
        
        # 2. Extract dark red/brown lesions
        # Hemorrhages absorb green heavily (dark in green channel) and reflect red moderately
        # Contrast: Red channel minus Green channel is elevated for blood
        red = resized_rgb[:, :, 0].astype(np.float32)
        green_f = green.astype(np.float32)
        
        red_diff = np.clip(red - green_f, 0, 255).astype(np.uint8)
        clahe_diff = apply_clahe(red_diff, clip_limit=2.5)
        
        # Bottom-hat on green channel extracts dark lesions larger than microaneurysms
        k_he = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bottom_hat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, k_he)
        
        # Combine spectral red difference and morphological darkness
        he_field = cv2.bitwise_and(bottom_hat, clahe_diff)
        he_field = cv2.bitwise_and(he_field, he_field, mask=safe_fov)
        he_field[suppression > 0] = 0
        
        # 3. Adaptive Thresholding
        safe_pixels = he_field[safe_fov > 0]
        if len(safe_pixels) > 0 and np.max(safe_pixels) > 0:
            thresh_val = np.percentile(safe_pixels[safe_pixels > 0], 91.0) if np.any(safe_pixels > 0) else 35
            thresh_val = max(thresh_val, 22)
        else:
            thresh_val = 35
            
        _, raw_mask = cv2.threshold(he_field, thresh_val, 255, cv2.THRESH_BINARY)
        
        # 4. Connected Component Analysis
        # Hemorrhages are larger than isolated 1-2 pixel noise (> 25 pixels)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask)
        final_mask = np.zeros((h, w), dtype=np.uint8)
        regions = []
        
        # Center of image for quadrant assignment
        mid_x, mid_y = w // 2, h // 2
        quadrants = {
            "superior_temporal": 0,
            "inferior_temporal": 0,
            "superior_nasal": 0,
            "inferior_nasal": 0
        }
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 25 <= area <= 8000:
                final_mask[labels == i] = 255
                cx, cy = centroids[i]
                regions.append({"centroid": [float(cx), float(cy)], "area": int(area)})
                
                # Determine quadrant
                is_sup = cy < mid_y
                is_left = cx < mid_x
                if is_sup and is_left:
                    quadrants["superior_temporal"] += 1
                elif (not is_sup) and is_left:
                    quadrants["inferior_temporal"] += 1
                elif is_sup and (not is_left):
                    quadrants["superior_nasal"] += 1
                else:
                    quadrants["inferior_nasal"] += 1
                    
        # Optional fusion with U-Net Class 2 (Hemorrhages)
        if unet_he_prob is not None:
            if unet_he_prob.shape != (h, w):
                unet_he_prob = cv2.resize(unet_he_prob, (w, h))
            u_he = (unet_he_prob >= 0.35).astype(np.uint8) * 255
            final_mask = cv2.bitwise_or(final_mask, u_he)
            final_mask[suppression > 0] = 0
            
        total_he_pixels = int((final_mask > 0).sum())
        retina_pixels = float((safe_fov > 0).sum())
        area_pct = (total_he_pixels / retina_pixels * 100.0) if retina_pixels > 0 else 0.0
        
        detected = len(regions) > 0 or total_he_pixels > 35
        classification = "Hemorrhages Present" if detected else "Hemorrhages Absent"
        confidence = 0.89 if detected else 0.93
        
        # Scale to original dimensions
        mask_original = cv2.resize(final_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return {
            "classification": classification,
            "detected": bool(detected),
            "hemorrhage_count": len(regions),
            "total_area_px": total_he_pixels,
            "area_percentage": round(area_pct, 3),
            "quadrant_distribution": quadrants,
            "binary_mask": final_mask,
            "binary_mask_original": mask_original,
            "confidence": round(confidence, 4),
            "clinical_significance": (
                "Intraretinal vascular rupture (dot/blot in INL/OPL; flame-shaped in RNFL) "
                "indicating moderate to severe retinal ischemia and vascular fragility."
            )
        }


def analyze_hemorrhages(
    image_input: Any,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Production wrapper saving hemorrhage outputs to disk."""
    if output_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = os.path.join(root_dir, "results", "retinal_analysis", "hemorrhages")
    os.makedirs(output_dir, exist_ok=True)
    
    from retinal_analysis.visualization import draw_lesion_overlay, save_image
    
    orig_rgb = load_fundus_image(image_input)
    analyzer = HemorrhageAnalyzer()
    result = analyzer.analyze(orig_rgb)
    
    orig_path = os.path.join(output_dir, "original.png")
    save_image(orig_rgb, orig_path)
    
    mask_path = os.path.join(output_dir, "predicted_mask.png")
    save_image(result["binary_mask_original"], mask_path)
    
    overlay_rgb = draw_lesion_overlay(orig_rgb, {"hemorrhage": result["binary_mask_original"]})
    overlay_path = os.path.join(output_dir, "overlay.png")
    save_image(overlay_rgb, overlay_path)
    
    result["saved_paths"] = {
        "original": orig_path,
        "predicted_mask": mask_path,
        "overlay": overlay_path
    }
    result["overlay_rgb"] = overlay_rgb
    return result
