"""
Fovea Localization Architecture
Combines anatomical OD-temporal spatial prior with macular avascular intensity depression.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

from retinal_analysis.preprocessing import preprocess_fundus
from retinal_analysis.optic_disc.model import OpticDiscDetector


class FoveaLocalizer:
    """
    Anatomical Fovea Localizer.
    Locates the foveal avascular zone (FAZ) using optic disc geometry and local minimum reflectance.
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.od_detector = OpticDiscDetector(target_size=target_size)

    def locate(
        self,
        image_input: Any,
        optic_disc_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Locates the fovea in the fundus image.

        Args:
            image_input: Fundus image (file, PIL, numpy, or bytes)
            optic_disc_result: Optional precomputed Optic Disc detection result

        Returns:
            Dict containing:
                - detected: bool
                - center: [x, y] in original image coordinates
                - confidence: float in [0.0, 1.0]
                - roi: [x1, y1, x2, y2] search region in original coordinates
                - center_target: [x, y] at target_size
                - eye_laterality: "OD" (Right Eye) or "OS" (Left Eye)
        """
        prep = preprocess_fundus(image_input, target_size=self.target_size)
        resized_rgb = prep["resized_rgb"]
        green = prep["green_channel"]
        fov_mask = prep["fov_mask"]
        scale_x = prep["scale_x"]
        scale_y = prep["scale_y"]
        h, w = self.target_size
        
        # 1. Obtain Optic Disc landmark
        if optic_disc_result is None or not optic_disc_result.get("detected"):
            od_res = self.od_detector.detect(prep["original_rgb"])
        else:
            od_res = optic_disc_result
            
        if not od_res.get("detected"):
            # If OD not found, fallback to central macular avascular assumption
            fx_t, fy_t = w * 0.5, h * 0.5
            return self._format_result(
                fx_t, fy_t, scale_x, scale_y, (0, 0, w, h), 0.35, "UNKNOWN"
            )
            
        od_cx = od_res["center_target"][0]
        od_cy = od_res["center_target"][1]
        od_r = max(od_res["radius_target"], 20.0)
        od_diam = 2.0 * od_r
        
        # 2. Determine Eye Laterality (OD = Right Eye, OS = Left Eye)
        # In a standard posterior pole view:
        # If disc is on the RIGHT side of the image (x > w/2), it's a RIGHT eye (OD) -> Macula is to the LEFT.
        # If disc is on the LEFT side of the image (x < w/2), it's a LEFT eye (OS) -> Macula is to the RIGHT.
        if od_cx > (w * 0.5):
            eye_laterality = "OD"   # Right Eye
            temporal_direction = -1 # Macula is located leftwards
        else:
            eye_laterality = "OS"   # Left Eye
            temporal_direction = 1  # Macula is located rightwards
            
        # 3. Predict expected fovea position:
        # Standard distance is approx 2.5 disc diameters temporal, with slight inferior offset (~0.2 OD diam)
        expected_fx = od_cx + (temporal_direction * 2.5 * od_diam)
        expected_fy = od_cy + (0.2 * od_diam)
        
        # Clamp to image boundaries
        expected_fx = float(np.clip(expected_fx, w * 0.15, w * 0.85))
        expected_fy = float(np.clip(expected_fy, h * 0.20, h * 0.80))
        
        # 4. Refine position by finding the darkest avascular zone inside macular search ROI
        roi_half_w = int(od_diam * 0.75)
        roi_half_h = int(od_diam * 0.60)
        
        rx1 = max(0, int(expected_fx - roi_half_w))
        ry1 = max(0, int(expected_fy - roi_half_h))
        rx2 = min(w - 1, int(expected_fx + roi_half_w))
        ry2 = min(h - 1, int(expected_fy + roi_half_h))
        
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[ry1:ry2, rx1:rx2] = 255
        roi_mask = cv2.bitwise_and(roi_mask, fov_mask)
        
        # Heavy blur to remove small capillary noise and find the broad foveal depression
        blurred_green = cv2.GaussianBlur(green, (25, 25), 0)
        
        # Find minimum intensity in the ROI
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred_green, mask=roi_mask)
        
        # Weighted combination: 60% intensity minimum + 40% anatomical prior
        refined_fx = 0.60 * min_loc[0] + 0.40 * expected_fx
        refined_fy = 0.60 * min_loc[1] + 0.40 * expected_fy
        
        confidence = float(np.clip(od_res["confidence"] * 0.92, 0.50, 0.95))
        
        return self._format_result(
            refined_fx, refined_fy, scale_x, scale_y,
            (rx1, ry1, rx2, ry2), confidence, eye_laterality
        )

    def _format_result(
        self,
        fx_t: float,
        fy_t: float,
        scale_x: float,
        scale_y: float,
        roi_t: Tuple[int, int, int, int],
        confidence: float,
        laterality: str
    ) -> Dict[str, Any]:
        orig_fx = fx_t * scale_x
        orig_fy = fy_t * scale_y
        orig_roi = [
            int(roi_t[0] * scale_x), int(roi_t[1] * scale_y),
            int(roi_t[2] * scale_x), int(roi_t[3] * scale_y)
        ]
        return {
            "detected": True,
            "center": [float(orig_fx), float(orig_fy)],
            "confidence": round(confidence, 4),
            "roi": orig_roi,
            "center_target": [float(fx_t), float(fy_t)],
            "roi_target": list(roi_t),
            "eye_laterality": laterality
        }
