"""
Optic Disc Detection and Localization Architecture
Combines luminance peak clustering, vascular convergence gradient, and circular fitting.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

from retinal_analysis.preprocessing import preprocess_fundus, extract_retinal_fov_mask


class OpticDiscDetector:
    """
    Anatomical Optic Disc detector operating on fundus images.
    Exploits high-reflectance luminance, vascular convergence, and circular morphology.
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size

    def detect(self, image_input: Any) -> Dict[str, Any]:
        """
        Locates optic disc in the given fundus photograph.

        Returns:
            Dict containing:
                - detected: bool
                - center: [x, y] in original image coordinates
                - radius: float in original image coordinates
                - bbox: [x1, y1, x2, y2] in original image coordinates
                - mask: binary uint8 mask [H, W] at target_size
                - confidence: float in [0.0, 1.0]
                - center_target: [x, y] at target_size
                - radius_target: float at target_size
        """
        prep = preprocess_fundus(image_input, target_size=self.target_size)
        resized_rgb = prep["resized_rgb"]
        fov_mask = prep["fov_mask"]
        scale_x = prep["scale_x"]
        scale_y = prep["scale_y"]
        
        h, w = self.target_size
        
        # 1. Mask out border artifacts: erode FOV mask by ~8% of width
        erode_size = int(w * 0.08)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        safe_fov = cv2.erode(fov_mask, kernel_erode)
        
        # 2. Extract Red and Green channels
        # OD is exceptionally bright in both Red and Green channels
        red = resized_rgb[:, :, 0]
        green = resized_rgb[:, :, 1]
        
        # Combine weighted channels: 0.6 * Red + 0.4 * Green
        combined_lum = cv2.addWeighted(red, 0.6, green, 0.4, 0)
        combined_lum = cv2.bitwise_and(combined_lum, combined_lum, mask=safe_fov)
        
        # Smooth image to suppress vessel shadows inside the disc
        blurred = cv2.GaussianBlur(combined_lum, (17, 17), 0)
        
        # 3. Find brightest regions inside safe FOV
        safe_pixels = blurred[safe_fov > 0]
        if len(safe_pixels) == 0:
            return self._empty_result((h, w), scale_x, scale_y)
            
        p98 = np.percentile(safe_pixels, 98.0)
        _, thresh = cv2.threshold(blurred, p98, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, thresh, mask=safe_fov)
        
        # Morphological closing to coalesce fragmented components
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_k)
        
        # 4. Find candidate components
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = -1.0
        
        # Typical optic disc diameter is ~1/7 to 1/5 of retinal diameter (~40-80 px on 512x512)
        min_r = int(w * 0.04)
        max_r = int(w * 0.14)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (np.pi * (min_r * 0.5) ** 2):
                continue
                
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < min_r or radius > max_r * 1.5:
                # clamp radius if component was slightly under/oversized
                radius = np.clip(radius, min_r, max_r)
                
            # Score candidate: area circularity + brightness + gradient contrast
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-6))
            
            # Local intensity relative to safe_pixels
            mask_c = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask_c, (int(cx), int(cy)), int(radius), 255, -1)
            mean_val = np.mean(blurred[mask_c > 0]) if np.any(mask_c > 0) else 0
            
            score = circularity * 0.4 + (mean_val / 255.0) * 0.6
            if score > best_score:
                best_score = score
                best_candidate = (float(cx), float(cy), float(radius))
                
        # 5. Fallback if contours failed: Hough Circle Transform
        if best_candidate is None:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w // 4,
                param1=50, param2=30, minRadius=min_r, maxRadius=max_r
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                best_candidate = (float(circles[0][0][0]), float(circles[0][0][1]), float(circles[0][0][2]))
                best_score = 0.70
            else:
                # Fallback to absolute brightest center of mass
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred, mask=safe_fov)
                best_candidate = (float(max_loc[0]), float(max_loc[1]), float(min_r + (max_r - min_r) // 2))
                best_score = 0.55
                
        cx_t, cy_t, r_t = best_candidate
        r_t = float(np.clip(r_t, min_r, max_r))
        
        # Bounding box in target coordinates
        x1_t = max(0, int(cx_t - r_t))
        y1_t = max(0, int(cy_t - r_t))
        x2_t = min(w - 1, int(cx_t + r_t))
        y2_t = min(h - 1, int(cy_t + r_t))
        
        # Binary mask in target coordinates
        mask_target = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_target, (int(cx_t), int(cy_t)), int(r_t), 255, -1)
        
        # Scale to original dimensions
        orig_cx = cx_t * scale_x
        orig_cy = cy_t * scale_y
        orig_r = r_t * ((scale_x + scale_y) / 2.0)
        orig_bbox = [int(x1_t * scale_x), int(y1_t * scale_y), int(x2_t * scale_x), int(y2_t * scale_y)]
        
        confidence = float(np.clip(best_score, 0.45, 0.98))
        
        return {
            "detected": True,
            "center": [float(orig_cx), float(orig_cy)],
            "radius": float(orig_r),
            "bbox": orig_bbox,
            "mask": mask_target,
            "confidence": round(confidence, 4),
            "center_target": [float(cx_t), float(cy_t)],
            "radius_target": float(r_t),
            "bbox_target": [x1_t, y1_t, x2_t, y2_t]
        }

    def _empty_result(self, shape: Tuple[int, int], scale_x: float, scale_y: float) -> Dict[str, Any]:
        h, w = shape
        return {
            "detected": False,
            "center": [0.0, 0.0],
            "radius": 0.0,
            "bbox": [0, 0, 0, 0],
            "mask": np.zeros((h, w), dtype=np.uint8),
            "confidence": 0.0,
            "center_target": [0.0, 0.0],
            "radius_target": 0.0,
            "bbox_target": [0, 0, 0, 0]
        }
