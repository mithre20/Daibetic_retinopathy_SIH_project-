"""
Optic Disc Inference Pipeline
Saves visual overlays and prediction JSON to results/retinal_analysis/optic_disc/
"""

import os
import json
from typing import Union, Dict, Any, Optional
import numpy as np
from PIL import Image

from retinal_analysis.optic_disc.model import OpticDiscDetector
from retinal_analysis.visualization import draw_optic_disc_overlay, save_image
from retinal_analysis.preprocessing import load_fundus_image

_DEFAULT_DETECTOR = None

def get_optic_disc_detector() -> OpticDiscDetector:
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = OpticDiscDetector()
    return _DEFAULT_DETECTOR


def locate_optic_disc(
    image_input: Union[str, bytes, Image.Image, np.ndarray],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs Optic Disc detection and generates visual overlays and metadata.
    """
    if output_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = os.path.join(root_dir, "results", "retinal_analysis", "optic_disc")
    os.makedirs(output_dir, exist_ok=True)
    
    orig_rgb = load_fundus_image(image_input)
    detector = get_optic_disc_detector()
    result = detector.detect(orig_rgb)
    
    # Save original image
    orig_path = os.path.join(output_dir, "original.png")
    save_image(orig_rgb, orig_path)
    
    # Draw and save overlay
    overlay_rgb = draw_optic_disc_overlay(
        image_rgb=orig_rgb,
        center=(int(result["center"][0]), int(result["center"][1])),
        radius=int(result["radius"]),
        bbox=tuple(result["bbox"]) if result["detected"] else None,
        confidence=result["confidence"]
    )
    overlay_path = os.path.join(output_dir, "optic_disc_overlay.png")
    save_image(overlay_rgb, overlay_path)
    
    # Save prediction JSON
    pred_payload = {
        "detected": result["detected"],
        "center_x": round(result["center"][0], 2),
        "center_y": round(result["center"][1], 2),
        "radius": round(result["radius"], 2),
        "bounding_box": result["bbox"],
        "confidence": result["confidence"],
        "original_image_path": orig_path,
        "overlay_image_path": overlay_path,
        "methodology": "Photometric Peak Clustering + Circular Enclosure",
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
