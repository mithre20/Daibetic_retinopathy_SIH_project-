"""
Unified Retinal Analysis Pipeline
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Chains:
    Fundus Image
         ↓
    Preprocessing
         ↓
    Optic Disc Localization
         ↓
    Fovea Localization
         ↓
    Vessel Segmentation
         ↓
    Lesion Analysis (Microaneurysm, Hemorrhage, Hard & Soft Exudates)
         ↓
    Neovascularization Check (Pending Dataset)
         ↓
    Structured Retinal Evidence Payload + Composite Clinical Visualization
"""

import os
import json
import time
from typing import Dict, Any, Union, Optional
import numpy as np
from PIL import Image

from retinal_analysis.preprocessing import preprocess_fundus, load_fundus_image
from retinal_analysis.optic_disc.model import OpticDiscDetector
from retinal_analysis.fovea.model import FoveaLocalizer
from retinal_analysis.vessels.model import RetinalVesselSegmenter
from retinal_analysis.lesions.microaneurysm import MicroaneurysmDetector
from retinal_analysis.lesions.hemorrhage import HemorrhageAnalyzer
from retinal_analysis.lesions.exudate import ExudateSegmenter
from retinal_analysis.neovascularization.model import NeovascularizationDetector
from retinal_analysis.visualization import (
    draw_optic_disc_overlay,
    draw_fovea_overlay,
    draw_vessel_overlay,
    draw_lesion_overlay,
    draw_composite_evidence,
    numpy_to_base64_data_uri,
    save_image
)


class UnifiedRetinalPipeline:
    """
    End-to-end multi-task retinal structure and diabetic lesion analysis pipeline.
    """

    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        self.od_detector = OpticDiscDetector(target_size=target_size)
        self.fovea_localizer = FoveaLocalizer(target_size=target_size)
        self.vessel_segmenter = RetinalVesselSegmenter(target_size=target_size)
        self.ma_detector = MicroaneurysmDetector(target_size=target_size)
        self.he_analyzer = HemorrhageAnalyzer(target_size=target_size)
        self.ex_segmenter = ExudateSegmenter(target_size=target_size)
        self.nv_detector = NeovascularizationDetector(target_size=target_size)

    def analyze(
        self,
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        output_dir: Optional[str] = None,
        include_base64: bool = False
    ) -> Dict[str, Any]:
        """
        Executes complete retinal structural and lesion analysis.
        """
        start_time = time.perf_counter()
        
        if output_dir is None:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            output_dir = os.path.join(root_dir, "results", "retinal_analysis", "pipeline")
        os.makedirs(output_dir, exist_ok=True)
        
        orig_rgb = load_fundus_image(image_input)
        
        # 1. Preprocessing
        t0 = time.perf_counter()
        prep = preprocess_fundus(orig_rgb, target_size=self.target_size)
        t_prep = (time.perf_counter() - t0) * 1000.0
        
        # 2. Optic Disc Localization
        t0 = time.perf_counter()
        od_res = self.od_detector.detect(orig_rgb)
        t_od = (time.perf_counter() - t0) * 1000.0
        
        # 3. Fovea Localization
        t0 = time.perf_counter()
        fovea_res = self.fovea_localizer.locate(orig_rgb, optic_disc_result=od_res)
        t_fovea = (time.perf_counter() - t0) * 1000.0
        
        # 4. Retinal Vessel Segmentation
        t0 = time.perf_counter()
        vessel_res = self.vessel_segmenter.segment(orig_rgb)
        t_vessels = (time.perf_counter() - t0) * 1000.0
        
        # 5. Lesion Analysis
        t0 = time.perf_counter()
        ma_res = self.ma_detector.detect(
            orig_rgb,
            vessel_mask=vessel_res["binary_mask"],
            optic_disc_mask=od_res["mask"]
        )
        he_res = self.he_analyzer.analyze(
            orig_rgb,
            vessel_mask=vessel_res["binary_mask"],
            optic_disc_mask=od_res["mask"]
        )
        ex_res = self.ex_segmenter.segment(
            orig_rgb,
            optic_disc_mask=od_res["mask"]
        )
        t_lesions = (time.perf_counter() - t0) * 1000.0
        
        # 6. Neovascularization
        t0 = time.perf_counter()
        nv_res = self.nv_detector.detect(
            orig_rgb,
            optic_disc_result=od_res,
            vessel_mask=vessel_res["binary_mask"]
        )
        t_nv = (time.perf_counter() - t0) * 1000.0
        
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        # 7. Generate Composite Visualizations
        lesion_masks_dict = {
            "microaneurysm": ma_res["binary_mask_original"],
            "hemorrhage": he_res["binary_mask_original"],
            "hard_exudate": ex_res["hard_exudates_mask_original"],
            "soft_exudate": ex_res["soft_exudates_mask_original"]
        }
        
        composite_rgb = draw_composite_evidence(
            image_rgb=orig_rgb,
            od_result=od_res,
            fovea_result=fovea_res,
            vessel_mask=vessel_res["binary_mask_original"],
            lesion_masks=lesion_masks_dict
        )
        
        # Save images to disk
        orig_path = os.path.join(output_dir, "original.png")
        save_image(orig_rgb, orig_path)
        
        composite_path = os.path.join(output_dir, "composite_evidence.png")
        save_image(composite_rgb, composite_path)
        
        vessel_path = os.path.join(output_dir, "vessel_overlay.png")
        vessel_overlay_rgb = draw_vessel_overlay(orig_rgb, vessel_res["binary_mask_original"])
        save_image(vessel_overlay_rgb, vessel_path)
        
        od_path = os.path.join(output_dir, "optic_disc_overlay.png")
        od_overlay_rgb = draw_optic_disc_overlay(
            orig_rgb,
            center=(int(od_res["center"][0]), int(od_res["center"][1])),
            radius=int(od_res["radius"]),
            bbox=tuple(od_res["bbox"]),
            confidence=od_res["confidence"]
        )
        save_image(od_overlay_rgb, od_path)
        
        fovea_path = os.path.join(output_dir, "fovea_overlay.png")
        fovea_overlay_rgb = draw_fovea_overlay(
            orig_rgb,
            fovea_center=(int(fovea_res["center"][0]), int(fovea_res["center"][1])),
            od_center=(int(od_res["center"][0]), int(od_res["center"][1])),
            confidence=fovea_res["confidence"]
        )
        save_image(fovea_overlay_rgb, fovea_path)
        
        lesion_overlay_path = os.path.join(output_dir, "lesion_overlay.png")
        lesion_overlay_rgb = draw_lesion_overlay(orig_rgb, lesion_masks_dict)
        save_image(lesion_overlay_rgb, lesion_overlay_path)
        
        # 8. Assemble Clean Structured Evidence
        evidence = {
            "retinal_structures": {
                "optic_disc": {
                    "detected": od_res["detected"],
                    "center": [round(c, 1) for c in od_res["center"]],
                    "radius": round(od_res["radius"], 1),
                    "bounding_box": od_res["bbox"],
                    "confidence": od_res["confidence"],
                    "overlay_path": od_path,
                    "status": "VALIDATED_ALGORITHMIC"
                },
                "fovea": {
                    "detected": fovea_res["detected"],
                    "center": [round(c, 1) for c in fovea_res["center"]],
                    "eye_laterality": fovea_res["eye_laterality"],
                    "confidence": fovea_res["confidence"],
                    "overlay_path": fovea_path,
                    "status": "VALIDATED_ALGORITHMIC"
                },
                "vessels": {
                    "available": True,
                    "vessel_density_pct": vessel_res["vessel_density_pct"],
                    "overlay_path": vessel_path,
                    "status": "VALIDATED_ALGORITHMIC"
                }
            },
            "lesions": {
                "microaneurysms": {
                    "detected": ma_res["detected"],
                    "count": ma_res["candidates_count"],
                    "confidence": ma_res["confidence"],
                    "subpixel_localization": "NOT VALIDATED",
                    "status": "VALIDATED_ALGORITHMIC"
                },
                "hemorrhages": {
                    "detected": he_res["detected"],
                    "classification": he_res["classification"],
                    "count": he_res["hemorrhage_count"],
                    "area_percentage": he_res["area_percentage"],
                    "quadrants": he_res["quadrant_distribution"],
                    "confidence": he_res["confidence"],
                    "status": "VALIDATED_ALGORITHMIC"
                },
                "hard_exudates": {
                    "detected": ex_res["hard_exudates_detected"],
                    "area_percentage": ex_res["hard_exudates_area_pct"],
                    "confidence": ex_res["confidence"],
                    "status": "VALIDATED_ALGORITHMIC"
                },
                "soft_exudates": {
                    "detected": ex_res["soft_exudates_detected"],
                    "area_percentage": ex_res["soft_exudates_area_pct"],
                    "confidence": ex_res["confidence"],
                    "status": "VALIDATED_ALGORITHMIC"
                }
            },
            "neovascularization": {
                "status": nv_res["status"],
                "detected": nv_res["detected"],
                "classification": nv_res["classification"],
                "message": nv_res["message"]
            },
            "execution_times_ms": {
                "preprocessing": round(t_prep, 1),
                "optic_disc": round(t_od, 1),
                "fovea": round(t_fovea, 1),
                "vessels": round(t_vessels, 1),
                "lesions": round(t_lesions, 1),
                "neovascularization": round(t_nv, 1),
                "total_pipeline": round(total_time_ms, 1)
            },
            "composite_overlay_path": composite_path
        }
        
        # Save structured evidence JSON
        def _to_serializable(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            return obj

        evidence_serializable = _to_serializable(evidence)
        json_path = os.path.join(output_dir, "retinal_evidence.json")
        with open(json_path, "w") as f:
            json.dump(evidence_serializable, f, indent=2)
            
        evidence["json_path"] = json_path
        
        if include_base64:
            evidence["composite_overlay_base64"] = numpy_to_base64_data_uri(composite_rgb)
            evidence["vessel_overlay_base64"] = numpy_to_base64_data_uri(vessel_overlay_rgb)
            evidence["lesion_overlay_base64"] = numpy_to_base64_data_uri(lesion_overlay_rgb)
            
        return evidence


_GLOBAL_PIPELINE = None

def analyze_retinal_structures(
    image_input: Any,
    output_dir: Optional[str] = None,
    include_base64: bool = False
) -> Dict[str, Any]:
    """Global convenience function for unified retinal structural analysis."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = UnifiedRetinalPipeline()
    return _GLOBAL_PIPELINE.analyze(image_input, output_dir=output_dir, include_base64=include_base64)
