"""
Diabetic Retinopathy Referral Decision Engine
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Deterministic, rule-based screening referral engine implementing confidence-aware
safety logic and strict image quality assessment (IQA) gating.

IMPORTANT:
This is NOT a machine learning or deep neural network model.
It is a transparent, auditable, rule-based clinical decision layer.
"""

from typing import Dict, Any, Optional, Union
from decision.referral_rules import (
    REFERRAL_RULES,
    DR_GRADE_NAMES,
    DR_GRADE_FULL_NAMES,
    DEFAULT_MIN_DR_CONFIDENCE,
    get_rule_for_grade
)

CLINICAL_DISCLAIMER: str = (
    "AI screening results are not a medical diagnosis. "
    "Professional clinical evaluation by an ophthalmologist or optometrist is recommended."
)


def generate_referral_decision(
    dr_grade: Optional[int],
    dr_confidence: Optional[float],
    image_quality_status: Optional[str] = "ACCEPTED",
    image_quality_confidence: Optional[float] = 1.0,
    minimum_dr_confidence: float = DEFAULT_MIN_DR_CONFIDENCE
) -> Dict[str, Any]:
    """
    Generate a transparent, deterministic referral decision based on DR severity grade,
    prediction confidence, and prior Image Quality Assessment (IQA).

    Parameters:
        dr_grade (int): DR severity grade (0 to 4).
        dr_confidence (float): Model confidence score for the DR grade (0.0 to 1.0).
        image_quality_status (str): Quality gate status ("ACCEPTED", "REJECTED", "Good Quality", "Poor Quality", "Unusable").
        image_quality_confidence (float): Confidence of the quality prediction (0.0 to 1.0).
        minimum_dr_confidence (float): Configurable engineering threshold for flagging low confidence (default: 0.50).

    Returns:
        Dict[str, Any]: Structured decision payload.
    """
    # -------------------------------------------------------------------------
    # STEP 1: Strict Image Quality Assessment (IQA) Pre-Gating
    # -------------------------------------------------------------------------
    # If image quality is poor, unusable, or good with insufficient confidence,
    # screening MUST be blocked immediately: dr_prediction = null, risk = null.
    is_quality_blocked = False
    block_reason = ""

    if image_quality_status is None:
        is_quality_blocked = True
        block_reason = "Missing image quality assessment status."
    else:
        status_upper = str(image_quality_status).strip().upper()
        if status_upper in ["REJECTED", "BLOCKED", "POOR", "POOR QUALITY", "UNUSABLE"]:
            is_quality_blocked = True
            block_reason = f"Image quality assessment rejected input ({image_quality_status})."
        elif status_upper in ["ACCEPTED", "GOOD", "GOOD QUALITY"]:
            # Check confidence threshold (default 0.75 for Good Quality)
            if image_quality_confidence is not None and image_quality_confidence < 0.75:
                is_quality_blocked = True
                block_reason = (
                    f"Good quality confidence ({image_quality_confidence*100:.1f}%) "
                    f"is below the safety threshold (75.0%)."
                )
        else:
            # Unknown quality status treated conservatively as blocked
            is_quality_blocked = True
            block_reason = f"Unrecognized image quality status: '{image_quality_status}'."

    if is_quality_blocked:
        return {
            "screening_allowed": False,
            "decision_status": "BLOCKED",
            "dr_prediction": None,
            "dr_grade": None,
            "dr_label": None,
            "dr_confidence": None,
            "risk_level": None,
            "referral_priority": None,
            "recommendation": "Please capture another retinal image.",
            "why_recommendation": (
                "Screening cannot be completed. "
                "Retinal Image Quality Assessment (IQA) gatekeeper blocked this image "
                "due to substandard optical quality. "
                "Recapture required before DR assessment."
            ),
            "reason": block_reason,
            "disclaimer": CLINICAL_DISCLAIMER
        }

    # -------------------------------------------------------------------------
    # STEP 2: Validate DR Grade and Confidence Inputs
    # -------------------------------------------------------------------------
    if dr_grade is None:
        return {
            "screening_allowed": False,
            "decision_status": "INVALID_INPUT",
            "error": "Unable to generate referral recommendation because a valid DR screening result is not available.",
            "dr_prediction": None,
            "dr_grade": None,
            "dr_label": None,
            "dr_confidence": None,
            "risk_level": None,
            "referral_priority": None,
            "recommendation": "Unable to generate referral recommendation because a valid DR screening result is not available.",
            "why_recommendation": "No DR classification result was provided to the referral engine.",
            "disclaimer": CLINICAL_DISCLAIMER
        }

    if not isinstance(dr_grade, int) or dr_grade not in REFERRAL_RULES:
        return {
            "screening_allowed": False,
            "decision_status": "INVALID_INPUT",
            "error": f"Invalid DR grade '{dr_grade}'. Expected integer between 0 and 4.",
            "dr_prediction": None,
            "dr_grade": dr_grade,
            "dr_label": None,
            "dr_confidence": dr_confidence,
            "risk_level": None,
            "referral_priority": None,
            "recommendation": "Unable to generate referral recommendation because a valid DR screening result is not available.",
            "why_recommendation": f"DR grade {dr_grade} is not a recognized severity grade (0 to 4).",
            "disclaimer": CLINICAL_DISCLAIMER
        }

    if dr_confidence is None or not isinstance(dr_confidence, (int, float)) or not (0.0 <= dr_confidence <= 1.0):
        return {
            "screening_allowed": False,
            "decision_status": "INVALID_INPUT",
            "error": "Unable to generate referral recommendation because DR prediction confidence is missing or invalid.",
            "dr_prediction": None,
            "dr_grade": dr_grade,
            "dr_label": DR_GRADE_NAMES.get(dr_grade, "Unknown"),
            "dr_confidence": dr_confidence,
            "risk_level": None,
            "referral_priority": None,
            "recommendation": "Unable to generate referral recommendation because a valid DR screening result is not available.",
            "why_recommendation": "Prediction confidence value is missing or outside the [0.0, 1.0] range.",
            "disclaimer": CLINICAL_DISCLAIMER
        }

    # -------------------------------------------------------------------------
    # STEP 3: Base Rule Evaluation
    # -------------------------------------------------------------------------
    rule = REFERRAL_RULES[dr_grade]
    dr_label = rule["dr_label"]
    dr_full_label = rule["dr_full_label"]
    risk_level = rule["risk_level"]
    referral_priority = rule["referral_priority"]
    recommendation = rule["recommendation"]
    clinical_action = rule["clinical_action"]

    # -------------------------------------------------------------------------
    # STEP 4: Confidence-Aware Logic & High-Severity Safety Safeguards
    # -------------------------------------------------------------------------
    is_low_confidence = float(dr_confidence) < float(minimum_dr_confidence)
    confidence_pct = dr_confidence * 100.0

    if is_low_confidence:
        decision_status = "LOW_CONFIDENCE"
        if dr_grade in [3, 4]:
            # SAFETY RULE 11: High-severity results (Grade 3 & 4) MUST NEVER be downgraded!
            # Maintain High Risk and Priority referral.
            risk_level = "High Risk"
            referral_priority = "Priority"
            recommendation = "Priority ophthalmologist referral"
            why_recommendation = (
                f"Why this recommendation?\n\n"
                f"Detected DR grade: {dr_label} (Grade {dr_grade})\n"
                f"Decision rule: DR Grade 3–4 → High Risk\n"
                f"Recommendation: Priority ophthalmologist referral.\n\n"
                f"Prediction confidence is low ({confidence_pct:.1f}% vs threshold {minimum_dr_confidence*100:.0f}%); "
                f"professional review is especially important. "
                f"High-severity screening signal is preserved and not downgraded."
            )
        else:
            # SAFETY RULE 10: For low-severity grades with low confidence, flag professional review
            decision_status = "LOW_CONFIDENCE"
            recommendation = "Professional review recommended"
            why_recommendation = (
                f"Why this recommendation?\n\n"
                f"Detected DR grade: {dr_label} (Grade {dr_grade})\n"
                f"Decision rule: Low-confidence prediction ({confidence_pct:.1f}% < {minimum_dr_confidence*100:.0f}% threshold).\n"
                f"Recommendation: High-priority professional review recommended because the model prediction has low confidence."
            )
    else:
        decision_status = "VALID"
        if dr_grade == 0:
            rule_str = "DR Grade 0 → Low Risk"
        elif dr_grade in [1, 2]:
            rule_str = "DR Grade 1–2 → Moderate Risk"
        else:
            rule_str = "DR Grade 3–4 → High Risk"

        why_recommendation = (
            f"Why this recommendation?\n\n"
            f"Detected DR grade: {dr_label} (Grade {dr_grade})\n"
            f"Decision rule: {rule_str}\n"
            f"Recommendation: {recommendation}."
        )

    # -------------------------------------------------------------------------
    # STEP 5: Assemble Structured Response (Conforming to Sections 15-20)
    # -------------------------------------------------------------------------
    return {
        "dr_grade": dr_grade,
        "dr_label": dr_label,
        "dr_full_label": dr_full_label,
        "dr_confidence": round(float(dr_confidence), 4),
        "risk_level": risk_level,
        "referral_priority": referral_priority,
        "recommendation": recommendation,
        "clinical_action": clinical_action,
        "decision_status": decision_status,
        "screening_allowed": True,
        "why_recommendation": why_recommendation,
        "minimum_dr_confidence": minimum_dr_confidence,
        "disclaimer": CLINICAL_DISCLAIMER
    }
