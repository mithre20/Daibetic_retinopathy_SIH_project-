"""
Fovea Localization Inference Pipeline
Saves visual overlays and prediction JSON to results/retinal_analysis/fovea/
"""

import os
import json
from typing import Union, Dict, Any, Optional
import numpy as np
from PIL import Image

from retinal_analysis.fovea.model import FoveaLocalizer
from retinal_analysis.optic_disc.inference import locate_optic_disc
from retinal_analysis.visualization import draw_fovea_overlay, save_image
from retinal_analysis.preprocessing import load_fundus_image

_DEFAULT_LOCALIZER = None

def get_fovea_localizer() -> FoveaLocalizer:
    global _DEFAULT_LOCALIZER
    if _DEFAULT_LOCALIZER is None:
        _DEFAULT_LOCALIZER = FoveaLocalizer()
    return _DEFAULT_LOCALIZER


def locate_fovea(
    image_input: Union[str, bytes, Image.Image, np.ndarray],
    optic_disc_result: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs Fovea localization and generates visual overlays and metadata.
    """
    if output_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = os.path.join(root_dir, "results", "retinal_analysis", "fovea")
    os.makedirs(output_dir, exist_ok=True)
    
    orig_rgb = load_fundus_image(image_input)
    
    # Check if OD was provided or needs to be found
    if optic_disc_result is None:
        optic_disc_result = locate_optic_disc(orig_rgb)
        
    localizer = get_fovea_localizer()
    result = localizer.locate(orig_rgb, optic_disc_result=optic_disc_result)
    
    # Save original image
    orig_path = os.path.join(output_dir, "original.png")
    save_image(orig_rgb, orig_path)
    
    # Draw and save overlay
    od_c = (int(optic_disc_result["center"][0]), int(optic_disc_result["center"][1])) if optic_disc_result.get("detected") else None
    overlay_rgb = draw_fovea_overlay(
        image_rgb=orig_rgb,
        fovea_center=(int(result["center"][0]), int(result["center"][1])),
        od_center=od_c,
        confidence=result["confidence"]
    )
    overlay_path = os.path.join(output_dir, "fovea_overlay.png")
    save_image(overlay_rgb, overlay_path)
    
    # Save prediction JSON
    pred_payload = {
        "detected": result["detected"],
        "center_x": round(result["center"][0], 2),
        "center_y": round(result["center"][1], 2),
        "eye_laterality": result["eye_laterality"],
        "confidence": result["confidence"],
        "search_roi": result["roi"],
        "original_image_path": orig_path,
        "overlay_image_path": overlay_path,
        "methodology": "Anatomical Relational Prior + Macular Intensity Depression",
        "status": "VALIDATED_ALGORITHMIC"
    }
    pred_json_path = os.path.join(output_dir, "prediction.json")
    with open(pred_json_path, "w") as f:
        json.dump(pred_payload, f, indent=2)
        
    result["saved_paths"] = {
        "original": orig_path,
        "overlay": overlay_path,
        "prediction_json": pred_json_path
    }
    result["overlay_rgb"] = overlay_rgb
    return result
