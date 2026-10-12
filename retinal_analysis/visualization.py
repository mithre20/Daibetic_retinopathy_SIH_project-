"""
Standardized Clinical Visualization & Overlay Utilities
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Provides publication-quality overlays adhering to established medical color conventions:
- Optic Disc: Amber/Orange contour & cross-center
- Fovea: Cyan crosshair with avascular zone target circle
- Retinal Vessels: Semi-transparent emerald green
- Microaneurysms: Vivid Red (255, 50, 50)
- Hemorrhages: Deep Crimson (180, 0, 0)
- Hard Exudates: Vivid Yellow (255, 230, 0)
- Soft Exudates: Sky Blue / Cyan (0, 220, 255)
"""

import os
import io
import base64
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from PIL import Image
import cv2

# Medical color palette (RGB)
COLOR_OPTIC_DISC = (255, 140, 0)    # Amber / Dark Orange
COLOR_FOVEA = (0, 230, 230)         # Bright Cyan
COLOR_VESSELS = (16, 185, 129)       # Emerald Green
COLOR_MICROANEURYSM = (255, 50, 50)  # Vivid Red
COLOR_HEMORRHAGE = (180, 0, 0)       # Crimson
COLOR_HARD_EXUDATE = (255, 230, 0)   # Vivid Yellow
COLOR_SOFT_EXUDATE = (0, 200, 255)   # Light Blue / Cyan


def draw_optic_disc_overlay(
    image_rgb: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    confidence: float = 1.0,
    alpha: float = 0.8
) -> np.ndarray:
    """Draws localized optic disc circle, center cross, and bounding box."""
    overlay = image_rgb.copy()
    cx, cy = int(center[0]), int(center[1])
    r = max(int(radius), 5)
    
    # Outer circle contour
    cv2.circle(overlay, (cx, cy), r, COLOR_OPTIC_DISC, 2, cv2.LINE_AA)
    # Center dot & small crosshair
    cv2.circle(overlay, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.drawMarker(overlay, (cx, cy), COLOR_OPTIC_DISC, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    
    # Optional bounding box
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_OPTIC_DISC, 1, cv2.LINE_AA)
        
    # Text badge
    label = f"OD ({confidence*100:.1f}%)"
    cv2.putText(overlay, label, (max(5, cx - r), max(20, cy - r - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_OPTIC_DISC, 1, cv2.LINE_AA)
                
    return cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)


def draw_fovea_overlay(
    image_rgb: np.ndarray,
    fovea_center: Tuple[int, int],
    od_center: Optional[Tuple[int, int]] = None,
    confidence: float = 1.0,
    alpha: float = 0.85
) -> np.ndarray:
    """Draws localized fovea crosshair and optional anatomical vector connecting to Optic Disc."""
    overlay = image_rgb.copy()
    fx, fy = int(fovea_center[0]), int(fovea_center[1])
    
    # Fovea concentric target rings
    cv2.circle(overlay, (fx, fy), 12, COLOR_FOVEA, 1, cv2.LINE_AA)
    cv2.circle(overlay, (fx, fy), 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.drawMarker(overlay, (fx, fy), COLOR_FOVEA, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    
    # Optional anatomical axis connecting OD to Fovea
    if od_center is not None:
        ox, oy = int(od_center[0]), int(od_center[1])
        cv2.line(overlay, (ox, oy), (fx, fy), (200, 200, 200), 1, cv2.LINE_AA)
        
    # Badge
    label = f"Fovea ({confidence*100:.1f}%)"
    cv2.putText(overlay, label, (max(5, fx - 25), max(20, fy - 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_FOVEA, 1, cv2.LINE_AA)
                
    return cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)


def draw_vessel_overlay(
    image_rgb: np.ndarray,
    vessel_mask: np.ndarray,
    alpha: float = 0.45,
    color: Tuple[int, int, int] = COLOR_VESSELS
) -> np.ndarray:
    """Blends segmented retinal vasculature onto original fundus photograph."""
    overlay = image_rgb.copy()
    is_vessel = vessel_mask > 0
    overlay[is_vessel] = color
    return cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)


def draw_lesion_overlay(
    image_rgb: np.ndarray,
    masks_dict: Dict[str, np.ndarray],
    alpha: float = 0.55
) -> np.ndarray:
    """
    Blends lesion segmentation masks onto fundus image using established palette.
    keys: 'microaneurysm', 'hemorrhage', 'hard_exudate', 'soft_exudate'
    """
    overlay = image_rgb.copy()
    
    palette = {
        "microaneurysm": COLOR_MICROANEURYSM,
        "hemorrhage": COLOR_HEMORRHAGE,
        "hard_exudate": COLOR_HARD_EXUDATE,
        "soft_exudate": COLOR_SOFT_EXUDATE
    }
    
    # Priority order: soft exudate > hard exudate > hemorrhage > microaneurysm
    for lesion_key in ["microaneurysm", "hemorrhage", "hard_exudate", "soft_exudate"]:
        mask = masks_dict.get(lesion_key)
        if mask is not None and np.any(mask > 0):
            overlay[mask > 0] = palette[lesion_key]
            
    return cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)


def draw_composite_evidence(
    image_rgb: np.ndarray,
    od_result: Optional[dict] = None,
    fovea_result: Optional[dict] = None,
    vessel_mask: Optional[np.ndarray] = None,
    lesion_masks: Optional[Dict[str, np.ndarray]] = None,
    gradcam_heatmap: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generates unified clinical evidence visualization combining:
    Fundus + Vessels + Optic Disc + Fovea + Lesions (+ optional Grad-CAM contours)
    """
    composite = image_rgb.copy()
    
    # 1. Retinal Vasculature layer
    if vessel_mask is not None and np.any(vessel_mask > 0):
        composite = draw_vessel_overlay(composite, vessel_mask, alpha=0.35)
        
    # 2. Lesion layer
    if lesion_masks is not None:
        composite = draw_lesion_overlay(composite, lesion_masks, alpha=0.55)
        
    # 3. Optic Disc landmark
    if od_result and od_result.get("detected", False):
        composite = draw_optic_disc_overlay(
            composite,
            center=od_result["center"],
            radius=od_result["radius"],
            bbox=od_result.get("bbox"),
            confidence=od_result.get("confidence", 1.0)
        )
        
    # 4. Fovea landmark
    if fovea_result and fovea_result.get("detected", False):
        od_c = od_result["center"] if od_result and od_result.get("detected") else None
        composite = draw_fovea_overlay(
            composite,
            fovea_center=fovea_result["center"],
            od_center=od_c,
            confidence=fovea_result.get("confidence", 1.0)
        )
        
    # 5. Optional Grad-CAM salient contour
    if gradcam_heatmap is not None:
        norm_cam = cv2.normalize(gradcam_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(norm_cam, 175, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(composite, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
        
    return composite


def numpy_to_base64_data_uri(image_rgb: np.ndarray, format_type: str = "PNG") -> str:
    """Encodes RGB numpy array into data:image/png;base64,... string."""
    pil_img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=format_type)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format_type.lower()};base64,{encoded}"


def save_image(image_rgb: np.ndarray, save_path: str) -> str:
    """Saves image to disk, creating parent directories if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    pil_img = Image.fromarray(image_rgb)
    pil_img.save(save_path)
    return save_path
