"""
Retinal Vessel Segmentation Architecture
Multi-scale directional matched filtering, CLAHE, and adaptive hysteresis thresholding.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

from retinal_analysis.preprocessing import (
    preprocess_fundus,
    extract_green_channel,
    apply_clahe,
    illumination_equalization,
    extract_retinal_fov_mask
)


class RetinalVesselSegmenter:
    """
    Directional morphological vessel segmentation engine.
    Detects major vascular trunks, second-order branches, and fine precapillary arterioles.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        line_lengths: Tuple[int, ...] = (9, 13, 17),
        num_angles: int = 12
    ):
        self.target_size = target_size
        self.line_lengths = line_lengths
        self.num_angles = num_angles
        self._kernels = self._generate_directional_kernels()

    def _generate_directional_kernels(self):
        """Builds oriented linear structuring elements at discrete angular intervals."""
        kernels = {}
        for length in self.line_lengths:
            kernels[length] = []
            for i in range(self.num_angles):
                theta = i * (180.0 / self.num_angles)
                rad = np.deg2rad(theta)
                
                # Canvas size
                ksize = length if length % 2 != 0 else length + 1
                kernel = np.zeros((ksize, ksize), dtype=np.uint8)
                center = ksize // 2
                
                # Draw line through center at angle theta
                dx = int(round((center) * np.cos(rad)))
                dy = int(round((center) * np.sin(rad)))
                cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)
                kernels[length].append(kernel)
        return kernels

    def segment(self, image_input: Any) -> Dict[str, Any]:
        """
        Segments retinal vasculature from fundus photograph.

        Returns:
            Dict containing:
                - binary_mask: uint8 array [H, W] (255 = vessel, 0 = background)
                - vessel_probability: float32 array [H, W] in [0.0, 1.0]
                - vessel_density: float percentage of retina covered by vessels
                - binary_mask_original: uint8 array scaled to original image dimensions
        """
        prep = preprocess_fundus(image_input, target_size=self.target_size)
        green = prep["green_channel"]
        fov_mask = prep["fov_mask"]
        orig_shape = prep["orig_shape"]
        
        # 1. Invert green channel: vessels are darker than retinal fundus, so inverting makes them bright
        inverted = 255 - green
        
        # 2. Equalize background illumination & enhance contrast
        equalized = illumination_equalization(inverted, fov_mask=fov_mask)
        clahe_vessels = apply_clahe(equalized, clip_limit=2.0)
        
        # 3. Apply Multi-Scale Directional Line Morphology (Top-Hat)
        max_response = np.zeros(self.target_size, dtype=np.float32)
        
        for length in self.line_lengths:
            scale_max = np.zeros(self.target_size, dtype=np.uint8)
            for kernel in self._kernels[length]:
                opened = cv2.morphologyEx(clahe_vessels, cv2.MORPH_OPEN, kernel)
                scale_max = np.maximum(scale_max, opened)
            max_response += scale_max.astype(np.float32)
            
        max_response /= len(self.line_lengths)
        
        # 4. Normalize response inside FOV
        safe_fov = cv2.erode(fov_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        vessel_prob = np.zeros(self.target_size, dtype=np.float32)
        
        fov_pixels = max_response[safe_fov > 0]
        if len(fov_pixels) > 0:
            p_min = np.percentile(fov_pixels, 5)
            p_max = np.percentile(fov_pixels, 99.5)
            norm_resp = np.clip((max_response - p_min) / (p_max - p_min + 1e-6), 0.0, 1.0)
            vessel_prob = norm_resp * (safe_fov > 0).astype(np.float32)
            
        # 5. Threshold to binary mask
        # Adaptive thresholding based on distribution
        mean_prob = np.mean(vessel_prob[safe_fov > 0]) if np.any(safe_fov > 0) else 0.2
        std_prob = np.std(vessel_prob[safe_fov > 0]) if np.any(safe_fov > 0) else 0.1
        threshold_val = float(np.clip(mean_prob + 0.6 * std_prob, 0.30, 0.55))
        
        binary_mask = (vessel_prob >= threshold_val).astype(np.uint8) * 255
        
        # 6. Post-processing: remove isolated noise dots (< 8 connected pixels)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] < 12:
                binary_mask[labels == label_id] = 0
                
        # Mask with safe FOV to eliminate boundary artifacts
        binary_mask = cv2.bitwise_and(binary_mask, safe_fov)
        
        # Calculate vessel density (vessel pixels / total retina pixels)
        retina_area = float((safe_fov > 0).sum())
        vessel_pixels = float((binary_mask > 0).sum())
        vessel_density = (vessel_pixels / retina_area * 100.0) if retina_area > 0 else 0.0
        
        # Scale to original dimensions
        binary_mask_original = cv2.resize(
            binary_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST
        )
        
        return {
            "binary_mask": binary_mask,
            "binary_mask_original": binary_mask_original,
            "vessel_probability": vessel_prob,
            "vessel_density_pct": round(vessel_density, 2),
            "target_size": self.target_size,
            "threshold_used": round(threshold_val, 4),
            "available": True
        }
