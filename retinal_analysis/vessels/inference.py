"""
Retinal Vessel Segmentation Inference Pipeline
Saves predicted mask, overlay, and metrics JSON to results/retinal_analysis/vessels/
"""

import os
import json
from typing import Union, Dict, Any, Optional
import numpy as np
from PIL import Image

from retinal_analysis.vessels.model import RetinalVesselSegmenter
from retinal_analysis.visualization import draw_vessel_overlay, save_image
from retinal_analysis.preprocessing import load_fundus_image

_DEFAULT_SEGMENTER = None

def get_vessel_segmenter() -> RetinalVesselSegmenter:
    global _DEFAULT_SEGMENTER
    if _DEFAULT_SEGMENTER is None:
        _DEFAULT_SEGMENTER = RetinalVesselSegmenter()
    return _DEFAULT_SEGMENTER


def segment_retinal_vessels(
    image_input: Union[str, bytes, Image.Image, np.ndarray],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs vessel segmentation and generates binary mask, overlay, and density metrics.
    """
    if output_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = os.path.join(root_dir, "results", "retinal_analysis", "vessels")
    os.makedirs(output_dir, exist_ok=True)
    
    orig_rgb = load_fundus_image(image_input)
    segmenter = get_vessel_segmenter()
    result = segmenter.segment(orig_rgb)
    
    # Save original image
    orig_path = os.path.join(output_dir, "original.png")
    save_image(orig_rgb, orig_path)
    
    # Save predicted binary mask (original scale)
    mask_path = os.path.join(output_dir, "predicted_mask.png")
    save_image(result["binary_mask_original"], mask_path)
    
    # Draw and save overlay
    overlay_rgb = draw_vessel_overlay(
        image_rgb=orig_rgb,
        vessel_mask=result["binary_mask_original"],
        alpha=0.45
    )
    overlay_path = os.path.join(output_dir, "overlay.png")
    save_image(overlay_rgb, overlay_path)
    
    # Save metrics JSON
    metrics_payload = {
        "available": True,
        "vessel_density_percentage": result["vessel_density_pct"],
        "threshold_used": result["threshold_used"],
        "methodology": "Multi-Scale Directional Line Matched Filtering + CLAHE",
        "original_image_path": orig_path,
        "predicted_mask_path": mask_path,
        "overlay_path": overlay_path,
        "status": "VALIDATED_ALGORITHMIC"
    }
    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
        
    result["saved_paths"] = {
        "original": orig_path,
        "predicted_mask": mask_path,
        "overlay": overlay_path,
        "metrics_json": metrics_json_path
    }
    result["overlay_rgb"] = overlay_rgb
    return result
