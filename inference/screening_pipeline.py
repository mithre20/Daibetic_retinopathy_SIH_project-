"""
Integrated Screening Pipeline Gatekeeper for AI DR Screen
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Safety Architecture:
                 RETINAL FUNDUS IMAGE
                          │
                          ▼
             ┌──────────────────────────┐
             │ IMAGE QUALITY ASSESSMENT │
             │     MobileNetV3-Small    │
             └────────────┬─────────────┘
                          │
             ┌────────────┴─────────────┐
             │                          │
        GOOD + CONFIDENT           POOR / UNUSABLE /
        (Confidence >= 0.75)        LOW CONFIDENCE
             │                          │
             ▼                          ▼
      EfficientNet-B0             SCREENING BLOCKED
             │                     (dr_prediction = None)
             ▼                          │
        DR SEVERITY                     ▼
             │                  Capture New Image
     ┌───────┼────────┐
     ▼       ▼        ▼
  Grad-CAM  SHAP     U-Net
"""

import os
import sys
import json
from typing import Dict, Any, Optional, Union
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from inference.predict_iqa import IQAInferenceEngine
from backend.model.inference import DRInferenceEngine
from models.iqa_model import DEFAULT_GOOD_THRESHOLD


class RetinalScreeningGatekeeper:
    """
    Comprehensive screening gatekeeper enforcing safe medical triage.
    """
    def __init__(
        self,
        iqa_threshold: float = DEFAULT_GOOD_THRESHOLD,
        good_threshold: Optional[float] = None,
        iqa_weights: Optional[str] = None,
        dr_weights: Optional[str] = None,
        device: Any = None
    ):
        if good_threshold is not None:
            iqa_threshold = good_threshold
        self.iqa_threshold = iqa_threshold
        self.device = device
        self.iqa_engine = IQAInferenceEngine(weights_path=iqa_weights, threshold=iqa_threshold, device=device)
        self.dr_engine = None
        self.dr_weights = dr_weights

    def evaluate_gate(
        self,
        predicted_class: str,
        confidence: float,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Pure gating logic evaluator.
        """
        th = threshold if threshold is not None else self.iqa_threshold
        if predicted_class == "Good Quality":
            if confidence >= th:
                return {
                    "screening_allowed": True,
                    "reason": f"Image quality is Good Quality with confidence {confidence*100:.2f}% (threshold {th*100:.2f}%). Proceeding with DR screening."
                }
            else:
                return {
                    "screening_allowed": False,
                    "reason": f"DR screening blocked: Image quality is Good Quality, but confidence {confidence*100:.2f}% is below safety threshold {th*100:.2f}%. Recapture required."
                }
        else:
            return {
                "screening_allowed": False,
                "reason": f"DR screening blocked: Image quality is {predicted_class} (confidence: {confidence*100:.2f}%, threshold: {th*100:.2f}%). Recapture required."
            }

    def get_dr_engine(self) -> DRInferenceEngine:
        """Lazily initialize the primary EfficientNet-B0 DR inference engine."""
        if self.dr_engine is None:
            self.dr_engine = DRInferenceEngine(weights_path=self.dr_weights)
        return self.dr_engine

    def screen(
        self,
        image_input: Any,
        threshold: Optional[float] = None,
        include_explainability: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the two-stage gatekeeper screening pipeline.

        Returns:
            Structured dictionary containing quality assessment and, strictly if accepted,
            the EfficientNet-B0 Diabetic Retinopathy classification.
        """
        th = threshold if threshold is not None else self.iqa_threshold

        # Step 1 & 2: Run Retinal Image Quality Assessment
        iqa_res = self.iqa_engine.predict(image_input, threshold=th)
        gate_res = self.evaluate_gate(
            predicted_class=iqa_res["prediction"],
            confidence=iqa_res["confidence"],
            threshold=th
        )
        screening_allowed = gate_res["screening_allowed"]

        # Build consistent image_quality payload
        image_quality = {
            "predicted_class": iqa_res["prediction"],
            "class": iqa_res["prediction"],
            "class_id": iqa_res["class_id"],
            "confidence": iqa_res["confidence"],
            "status": "ACCEPTED" if screening_allowed else "REJECTED",
            "quality_status": "ACCEPTED" if screening_allowed else "REJECTED",
            "screening_allowed": screening_allowed,
            "threshold": th,
            "confidence_threshold": th,
            "probabilities": iqa_res["probabilities"],
            "class_probabilities": iqa_res["probabilities"],
            "quality_indicators": iqa_res.get("quality_indicators", {}),
            "classical_metrics": iqa_res.get("quality_indicators", {}),
            "defect_indicators": iqa_res.get("defect_indicators", []),
            "recommendations": iqa_res.get("recommendations", [iqa_res.get("recommendation", "")])
        }

        # Step 3: Check Gate Criteria
        from decision.referral_engine import generate_referral_decision

        if not screening_allowed:
            # STOP immediately: Do NOT run DR classifier
            ref_decision = generate_referral_decision(
                dr_grade=None,
                dr_confidence=None,
                image_quality_status=image_quality["status"],
                image_quality_confidence=image_quality["confidence"]
            )
            return {
                "screening_status": "BLOCKED",
                "screening_allowed": False,
                "screening_gate": gate_res,
                "image_quality": image_quality,
                "quality_assessment": image_quality,
                "dr_prediction": None,
                "risk_level": None,
                "referral_priority": None,
                "referral_decision": ref_decision,
                "message": gate_res["reason"],
                "recommendation": iqa_res.get("recommendation", "Please capture another retinal image.")
            }

        # Step 4: Image is Good & Confident -> Run EfficientNet-B0
        dr_eng = self.get_dr_engine()
        dr_res = dr_eng.predict(image_input)

        # Step 5: Deterministic Referral Engine
        ref_decision = generate_referral_decision(
            dr_grade=dr_res["prediction"],
            dr_confidence=dr_res["confidence"],
            image_quality_status="ACCEPTED",
            image_quality_confidence=image_quality["confidence"]
        )

        response = {
            "screening_status": "ACCEPTED",
            "screening_allowed": True,
            "screening_gate": gate_res,
            "image_quality": image_quality,
            "quality_assessment": image_quality,
            "dr_prediction": {
                "prediction": dr_res["prediction"],
                "class_id": dr_res["prediction"],
                "severity": dr_res["severity"],
                "confidence": dr_res["confidence"],
                "probabilities": dr_res["probabilities"],
                "referable": dr_res["referable"],
                "screening_recommendation": dr_res["screening_recommendation"]
            },
            "risk_level": ref_decision["risk_level"],
            "referral_priority": ref_decision["referral_priority"],
            "referral_decision": ref_decision,
            "message": "Image quality verified. Screening completed successfully.",
            "recommendation": ref_decision["recommendation"],
            "disclaimer": ref_decision["disclaimer"]
        }

        # Optional: Run Grad-CAM if requested
        if include_explainability:
            try:
                from inference.gradcam_efficientnet import generate_gradcam_explanation
                gc_res = generate_gradcam_explanation(
                    image=image_input,
                    target_class=dr_res["prediction"],
                    device=self.device
                )
                response["gradcam"] = gc_res
            except Exception as e:
                response["gradcam_error"] = str(e)

        return response

    def process(
        self,
        image_input: Any,
        threshold: Optional[float] = None,
        include_explainability: bool = False
    ) -> Dict[str, Any]:
        """Alias for screen method."""
        return self.screen(image_input, threshold=threshold, include_explainability=include_explainability)


# Module-level convenience function
_gatekeeper = None

def screen_retinal_image(
    image_input: Any,
    threshold: float = DEFAULT_GOOD_THRESHOLD,
    include_explainability: bool = False
) -> Dict[str, Any]:
    """
    Global entrypoint for IQA-gated DR screening.
    """
    global _gatekeeper
    if _gatekeeper is None or _gatekeeper.iqa_threshold != threshold:
        _gatekeeper = RetinalScreeningGatekeeper(iqa_threshold=threshold)
    return _gatekeeper.screen(image_input, threshold=threshold, include_explainability=include_explainability)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run IQA-Gated Retinal Screening Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to fundus photograph")
    parser.add_argument("--threshold", type=float, default=DEFAULT_GOOD_THRESHOLD, help="IQA Good Confidence Threshold")
    args = parser.parse_args()

    res = screen_retinal_image(args.image, threshold=args.threshold)
    print("\n" + "=" * 55)
    print(" AI DR Screen — Gated Screening Pipeline Result")
    print("=" * 55)
    print(f"Status:       {res['screening_status']}")
    print(f"Quality:      {res['quality_assessment']['class']} ({res['quality_assessment']['confidence']*100:.1f}%)")
    if res["dr_prediction"] is not None:
        print(f"DR Grade:     {res['dr_prediction']['severity']} ({res['dr_prediction']['confidence']*100:.1f}%)")
        print(f"Referable:    {res['dr_prediction']['referable']}")
        if "referral_decision" in res and res["referral_decision"]:
            ref = res["referral_decision"]
            print(f"Risk Level:   {ref.get('risk_level')}")
            print(f"Priority:     {ref.get('referral_priority')}")
            print(f"Action:       {ref.get('recommendation')}")
    else:
        print("DR Grade:     None (DR Classification blocked by Quality Gate)")
        if "referral_decision" in res and res["referral_decision"]:
            print(f"Referral:     {res['referral_decision'].get('recommendation')}")
    print(f"\nMessage:      {res['message']}")
    print("=" * 55 + "\n")
