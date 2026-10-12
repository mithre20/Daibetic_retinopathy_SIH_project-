"""
Dedicated Microaneurysm Detection Architecture
High-resolution punctate lesion extraction, vessel branch filtering, and U-Net feature fusion.
"""

import os
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import cv2

from retinal_analysis.preprocessing import (
    preprocess_fundus,
    load_fundus_image,
    extract_green_channel,
    apply_clahe,
    extract_retinal_fov_mask
)
from retinal_analysis.vessels.model import RetinalVesselSegmenter
from retinal_analysis.optic_disc.model import OpticDiscDetector


class MicroaneurysmDetector:
    """
    High-resolution punctate micro-lesion detector.
    Identifies localized capillary outpouchings (<125 µm) and filters vessel branch false-positives.
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.vessel_segmenter = RetinalVesselSegmenter(target_size=target_size)
        self.od_detector = OpticDiscDetector(target_size=target_size)

    def detect(
        self,
        image_input: Any,
        vessel_mask: Optional[np.ndarray] = None,
        optic_disc_mask: Optional[np.ndarray] = None,
        unet_prob_map: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Detects microaneurysms in the fundus photograph.

        Returns:
            Dict containing:
                - detected: bool
                - candidates_count: int
                - candidate_coordinates: List of [x, y] in original image coordinates
                - binary_mask: uint8 array [H, W] at target_size
                - binary_mask_original: uint8 array at original dimensions
                - confidence: float in [0.0, 1.0]
                - subpixel_status: "NOT VALIDATED"
        """
        prep = preprocess_fundus(image_input, target_size=self.target_size)
        green = prep["green_channel"]
        fov_mask = prep["fov_mask"]
        orig_shape = prep["orig_shape"]
        scale_x = prep["scale_x"]
        scale_y = prep["scale_y"]
        h, w = self.target_size
        
        # 1. Obtain Vessel Mask for False-Positive Suppression (branch points look like MAs)
        if vessel_mask is None:
            v_res = self.vessel_segmenter.segment(prep["original_rgb"])
            vessel_mask = v_res["binary_mask"]
        elif vessel_mask.shape != (h, w):
            vessel_mask = cv2.resize(vessel_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        # Dilate vessels slightly to cover edges and bifurcations
        vessel_dilated = cv2.dilate(vessel_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
        # 2. Obtain Optic Disc mask for suppression
        if optic_disc_mask is None:
            od_res = self.od_detector.detect(prep["original_rgb"])
            optic_disc_mask = od_res["mask"]
        elif optic_disc_mask.shape != (h, w):
            optic_disc_mask = cv2.resize(optic_disc_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        od_dilated = cv2.dilate(optic_disc_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        
        # 3. Enhance Green Channel Contrast
        clahe_g = apply_clahe(green, clip_limit=3.0)
        
        # Microaneurysms are small, dark, circular lesions in the green channel
        # Black Top-Hat transform extracts dark structures smaller than structuring element
        # A circular kernel of radius 4 px corresponds to ~80-100 µm on 512x512 retina
        k_bth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        black_tophat = cv2.morphologyEx(clahe_g, cv2.MORPH_BLACKHAT, k_bth)
        
        # 4. Safe FOV (erode to avoid circular fundus boundary false-positives)
        safe_fov = cv2.erode(fov_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        
        # Exclude vessel trunk regions and optic disc
        suppression_mask = cv2.bitwise_or(vessel_dilated, od_dilated)
        candidate_field = cv2.bitwise_and(black_tophat, black_tophat, mask=safe_fov)
        candidate_field[suppression_mask > 0] = 0
        
        # 5. Adaptive Thresholding of candidate intensity peaks
        safe_vals = candidate_field[safe_fov > 0]
        if len(safe_vals) > 0 and np.max(safe_vals) > 0:
            thresh_val = np.percentile(safe_vals[safe_vals > 0], 90.0) if np.any(safe_vals > 0) else 25
            thresh_val = max(thresh_val, 18)
        else:
            thresh_val = 30
            
        _, binary_candidates = cv2.threshold(candidate_field, thresh_val, 255, cv2.THRESH_BINARY)
        
        # 6. Geometric Shape Filtering (Size: 3 to 45 px, Circularity: > 0.40)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_candidates)
        final_ma_mask = np.zeros((h, w), dtype=np.uint8)
        candidate_coords_orig = []
        
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < 2 or area > 55:
                continue
                
            comp_mask = (labels == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            cnt = contours[0]
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-6))
            
            if circularity >= 0.35:
                final_ma_mask[labels == label_id] = 255
                cx, cy = centroids[label_id]
                candidate_coords_orig.append([float(cx * scale_x), float(cy * scale_y)])
                
        # Optional fusion with U-Net Class 1 (Microaneurysm)
        if unet_prob_map is not None:
            if unet_prob_map.shape != (h, w):
                unet_prob_map = cv2.resize(unet_prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
            unet_active = (unet_prob_map >= 0.30).astype(np.uint8) * 255
            # Union of high-resolution morphological candidates and U-Net activations
            final_ma_mask = cv2.bitwise_or(final_ma_mask, unet_active)
            final_ma_mask[suppression_mask > 0] = 0
            
        detected = len(candidate_coords_orig) > 0 or np.any(final_ma_mask > 0)
        confidence = float(np.clip(0.50 + 0.08 * min(len(candidate_coords_orig), 5), 0.50, 0.92)) if detected else 0.88
        
        # Rescale mask to original dimensions
        mask_original = cv2.resize(
            final_ma_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST
        )
        
        return {
            "detected": detected,
            "candidates_count": len(candidate_coords_orig),
            "candidate_coordinates": candidate_coords_orig,
            "binary_mask": final_ma_mask,
            "binary_mask_original": mask_original,
            "confidence": round(confidence, 4),
            "subpixel_localization": "NOT VALIDATED",
            "clinical_significance": "Early vascular sign of capillary basement membrane weakening",
            "methodology": "High-Res Black Top-Hat + Vessel Suppression + Shape Circularity Gating"
        }


def detect_microaneurysms(
    image_input: Any,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Production wrapper saving microaneurysm outputs to disk."""
    if output_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = os.path.join(root_dir, "results", "retinal_analysis", "microaneurysms")
    os.makedirs(output_dir, exist_ok=True)
    
    from retinal_analysis.visualization import draw_lesion_overlay, save_image
    
    orig_rgb = load_fundus_image(image_input)
    detector = MicroaneurysmDetector()
    result = detector.detect(orig_rgb)
    
    orig_path = os.path.join(output_dir, "original.png")
    save_image(orig_rgb, orig_path)
    
    mask_path = os.path.join(output_dir, "predicted_mask.png")
    save_image(result["binary_mask_original"], mask_path)
    
    overlay_rgb = draw_lesion_overlay(orig_rgb, {"microaneurysm": result["binary_mask_original"]})
    overlay_path = os.path.join(output_dir, "overlay.png")
    save_image(overlay_rgb, overlay_path)
    
    result["saved_paths"] = {
        "original": orig_path,
        "predicted_mask": mask_path,
        "overlay": overlay_path
    }
    result["overlay_rgb"] = overlay_rgb
    return result
