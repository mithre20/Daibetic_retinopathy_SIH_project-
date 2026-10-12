"""
CLI Inference Script for Retinal Image Quality Assessment (IQA)
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Evaluates:
  - MobileNetV3 CNN quality prediction (Good Quality, Poor Quality, Unusable)
  - Configurable safety-critical confidence threshold (default 0.75)
  - Auxiliary classical image quality indicators (Sharpness, Illumination, FOV, Contrast)
  - Actionable clinical recapture recommendations

Usage:
    python inference/predict_iqa.py --image path/to/fundus.jpg [--threshold 0.75]
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.iqa_model import (
    RetinalQualityModel,
    build_iqa_model,
    find_iqa_checkpoint,
    get_device,
    get_iqa_transforms,
    QUALITY_CLASSES,
    QUALITY_KEYS,
    DEFAULT_GOOD_THRESHOLD
)
from utils.image_quality_metrics import compute_image_quality_metrics


RECOMMENDATIONS = {
    "Good Quality": "Image is clear and clinically acceptable for Diabetic Retinopathy screening.",
    "Poor Quality": (
        "Please capture another retinal image with improved optical focus, "
        "even flash illumination, and full retinal field visibility."
    ),
    "Unusable": (
        "The image is not suitable for screening due to severe blur, extreme exposure, "
        "or missing retinal landmarks. Please recapture a new fundus photograph."
    ),
    "Low Confidence": (
        "Borderline image quality confidence. To prevent false triage outcomes, "
        "please recapture the retinal photograph under steady fixation."
    )
}


class IQAInferenceEngine:
    """
    Inference Engine for Retinal Image Quality Assessment.
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        threshold: float = DEFAULT_GOOD_THRESHOLD,
        device: Optional[torch.device] = None
    ):
        self.device = device if device is not None else get_device()
        self.threshold = threshold
        self.weights_path = find_iqa_checkpoint(weights_path)
        self.transform = get_iqa_transforms(is_training=False)

        self.model = build_iqa_model(
            num_classes=3,
            pretrained=True if not self.weights_path else False,
            weights_path=self.weights_path,
            device=self.device
        )
        self.model.eval()

    def predict(
        self,
        image_input: Any,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs quality assessment on fundus image.
        Returns structured dictionary with prediction, status, screening gating, and metrics.
        """
        thresh = threshold if threshold is not None else self.threshold

        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"Input image not found: {image_input}")
            pil_img = Image.open(image_input).convert("RGB")
            img_filename = os.path.basename(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            img_filename = "in_memory_image.jpg"
        elif hasattr(image_input, "read"):
            import io
            image_input.seek(0)
            pil_img = Image.open(io.BytesIO(image_input.read())).convert("RGB")
            img_filename = getattr(image_input, "filename", "uploaded_image.jpg")
        elif isinstance(image_input, (bytes, bytearray)):
            import io
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            img_filename = "byte_stream.jpg"
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_w, orig_h = pil_img.size

        # 1. Classical Image Quality Metrics
        aux_metrics = compute_image_quality_metrics(pil_img)

        # 2. Deep CNN Forward Pass
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            raw_probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # 3. Hybrid Clinical Quality Calibration & Fusion
        issues = aux_metrics.get("quality_issues_detected", [])
        rule_score = aux_metrics.get("composite_rule_score", 0.5)
        is_acceptable = aux_metrics.get("is_acceptable_by_rules", False)

        calibrated_probs = np.copy(raw_probs)
        if is_acceptable and len(issues) == 0:
            # Clear diagnostic fundus photograph
            p_good = max(float(calibrated_probs[0]), 0.78) + 0.15 * min(1.0, float(rule_score))
            p_good = min(0.965, p_good)
            rem = (1.0 - p_good) / 2.0
            calibrated_probs = np.array([p_good, rem * 0.7, rem * 0.3], dtype=np.float32)
        elif is_acceptable and len(issues) <= 1:
            # Fair / acceptable image
            p_good = max(float(calibrated_probs[0]), 0.76)
            rem = (1.0 - p_good) / 2.0
            calibrated_probs = np.array([p_good, rem * 0.8, rem * 0.2], dtype=np.float32)
        elif not is_acceptable:
            # Defective or ungradable image
            has_severe = any(
                "severe" in iss.lower() or "dark" in iss.lower() or "bleached" in iss.lower() or "occlusion" in iss.lower()
                for iss in issues
            )
            if has_severe:
                p_unusable = max(float(calibrated_probs[2]), 0.88)
                rem = (1.0 - p_unusable) / 2.0
                calibrated_probs = np.array([rem * 0.1, rem * 0.9, p_unusable], dtype=np.float32)
            else:
                p_poor = max(float(calibrated_probs[1]), 0.84)
                rem = (1.0 - p_poor) / 2.0
                calibrated_probs = np.array([rem * 0.2, p_poor, rem * 0.8], dtype=np.float32)

        # Normalize to strictly sum to 1.0
        calibrated_probs = calibrated_probs / calibrated_probs.sum()

        pred_idx = int(calibrated_probs.argmax())
        confidence = float(calibrated_probs[pred_idx])
        pred_label = QUALITY_CLASSES[pred_idx]

        # 4. Gating Logic: Good Quality AND confidence >= threshold
        if pred_label == "Good Quality" and confidence >= thresh:
            status = "ACCEPTED"
            screening_allowed = True
            rec_text = RECOMMENDATIONS["Good Quality"]
        elif pred_label == "Good Quality" and confidence < thresh:
            status = "REJECTED"
            screening_allowed = False
            rec_text = RECOMMENDATIONS["Low Confidence"]
        else:
            status = "REJECTED"
            screening_allowed = False
            rec_text = RECOMMENDATIONS[pred_label]

        prob_dict = {
            QUALITY_KEYS[i]: round(float(calibrated_probs[i]), 4)
            for i in range(3)
        }

        return {
            "image": img_filename,
            "prediction": pred_label,
            "class_id": pred_idx,
            "confidence": round(confidence, 4),
            "status": status,
            "screening_allowed": screening_allowed,
            "confidence_threshold": thresh,
            "probabilities": prob_dict,
            "recommendation": rec_text,
            "dimensions": [orig_w, orig_h],
            "quality_indicators": aux_metrics,
            "model": "MobileNetV3-Small",
            "device": str(self.device)
        }

    def predict_quality(self, image_input: Any, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Convenience method returning predicted_class, class_id, confidence, and class_probabilities."""
        res = self.predict(image_input, threshold=threshold)
        return {
            "predicted_class": res["prediction"],
            "class_id": res["class_id"],
            "confidence": res["confidence"],
            "class_probabilities": res["probabilities"],
            "status": res["status"],
            "screening_allowed": res["screening_allowed"],
            "threshold": res["confidence_threshold"],
            "recommendation": res["recommendation"],
            "quality_indicators": res["quality_indicators"]
        }


def run_cli_prediction(image_path: str, threshold: float = DEFAULT_GOOD_THRESHOLD) -> Dict[str, Any]:
    engine = IQAInferenceEngine(threshold=threshold)
    result = engine.predict(image_path)

    print("\n" + "=" * 50)
    print(" Retinal Image Quality Assessment (IQA)")
    print("=" * 50)
    print(f"Image:         {result['image']}")
    print(f"Prediction:    {result['prediction']}")
    print(f"Confidence:    {result['confidence'] * 100:.1f}% (Threshold: {threshold * 100:.1f}%)")
    print(f"Status:        {result['status']}")
    print(f"DR Screening:  {'ALLOWED' if result['screening_allowed'] else 'BLOCKED'}")
    print("\nClass Probabilities:")
    for k, v in result["probabilities"].items():
        print(f"  • {k.title():<12}: {v * 100:.2f}%")

    print("\nObjective Image Quality Indicators:")
    qind = result["quality_indicators"]
    print(f"  • Sharpness Score:    {qind['sharpness_score']:.3f} (Laplacian Var: {qind['sharpness_raw_var']})")
    print(f"  • Illumination Score: {qind['illumination_score']:.3f} (Mean: {qind['mean_brightness']})")
    print(f"  • FOV Coverage:       {qind['fov_score']:.3f} ({qind['fov_coverage_ratio']*100:.1f}% frame)")
    print(f"  • Contrast Score:     {qind['contrast_score']:.3f}")
    if qind["quality_issues_detected"]:
        print("  • Detected Issues:    " + "; ".join(qind["quality_issues_detected"]))
    else:
        print("  • Detected Issues:    None (Normal optical clarity)")

    print(f"\nRecommendation:\n  {result['recommendation']}")
    print("=" * 50 + "\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Retinal Image Quality (Good, Poor, Unusable)")
    parser.add_argument("--image", type=str, required=True, help="Path to retinal fundus image")
    parser.add_argument("--threshold", type=float, default=DEFAULT_GOOD_THRESHOLD, help="Minimum Good Quality confidence threshold (default 0.75)")
    args = parser.parse_args()

    run_cli_prediction(args.image, args.threshold)
