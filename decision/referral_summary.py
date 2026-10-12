"""
Diabetic Retinopathy Clinical Screening Summary Formatter
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Assembles a unified, transparent clinical screening report combining:
1. Retinal Image Quality Assessment (IQA)
2. Primary DR Severity Grade (EfficientNet-B0)
3. Deterministic Risk Level and Referral Recommendation
4. Explicit "Why this recommendation?" Explanation
5. Visual and Feature Explainability References (Grad-CAM, SHAP, U-Net)
6. Optional Clinical Context from Medical Reports (Glucose, HbA1c, Blood Pressure)
   without unvalidated mathematical scoring.
7. Mandatory Clinical Screening Disclaimer.
"""

from typing import Dict, Any, Optional
from decision.referral_engine import CLINICAL_DISCLAIMER


def format_clinical_screening_summary(
    referral_decision: Dict[str, Any],
    image_quality: Optional[Dict[str, Any]] = None,
    explainability: Optional[Dict[str, Any]] = None,
    medical_report_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format a unified, transparent screening summary dictionary.
    """
    screening_allowed = referral_decision.get("screening_allowed", False)

    # 1. Image Quality section
    iq_summary = {
        "status": image_quality.get("status", "UNKNOWN") if image_quality else "UNKNOWN",
        "predicted_class": image_quality.get("predicted_class", "Unknown") if image_quality else "Unknown",
        "confidence": image_quality.get("confidence", 0.0) if image_quality else 0.0,
        "screening_allowed": screening_allowed
    }

    # 2. DR Classification section
    dr_summary = {
        "dr_grade": referral_decision.get("dr_grade"),
        "dr_label": referral_decision.get("dr_label"),
        "dr_confidence": referral_decision.get("dr_confidence"),
        "confidence_percentage": (
            f"{referral_decision.get('dr_confidence') * 100:.1f}%"
            if referral_decision.get("dr_confidence") is not None
            else "N/A"
        )
    }

    # 3. Risk & Referral section
    risk_summary = {
        "risk_level": referral_decision.get("risk_level"),
        "referral_priority": referral_decision.get("referral_priority"),
        "recommendation": referral_decision.get("recommendation"),
        "clinical_action": referral_decision.get("clinical_action"),
        "decision_status": referral_decision.get("decision_status")
    }

    # 4. Explainability section
    explainability_summary = {
        "gradcam_available": bool(explainability and explainability.get("gradcam")),
        "shap_available": bool(explainability and explainability.get("shap")),
        "unet_available": bool(explainability and explainability.get("unet")),
        "description": "Grad-CAM (visual heatmaps), SHAP (feature attribution), and U-Net (lesion segmentation) provide auxiliary explainability."
    }

    # 5. Optional Medical Report Context (Zero unvalidated score invention)
    clinical_context = None
    if medical_report_data:
        clinical_context = {
            "glucose": medical_report_data.get("glucose"),
            "hba1c": medical_report_data.get("hba1c"),
            "blood_pressure": medical_report_data.get("blood_pressure"),
            "note": (
                "These additional values provide clinical context and should be "
                "reviewed by a qualified healthcare professional. "
                "They do not alter the deterministic DR referral rule."
            )
        }

    return {
        "title": "AI DR SCREEN — CLINICAL SCREENING SUMMARY",
        "screening_allowed": screening_allowed,
        "image_quality": iq_summary,
        "dr_screening": dr_summary,
        "risk_referral": risk_summary,
        "why_recommendation": referral_decision.get("why_recommendation", ""),
        "explainability": explainability_summary,
        "additional_clinical_context": clinical_context,
        "disclaimer": referral_decision.get("disclaimer", CLINICAL_DISCLAIMER)
    }


def format_screening_summary_text(
    referral_decision: Dict[str, Any],
    image_quality: Optional[Dict[str, Any]] = None,
    medical_report_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generates human-readable plain text matching Section 25.
    """
    lines = [
        "=======================================================",
        " AI DR SCREEN — EXPLAINABLE AI DR SCREENING IN RURAL INDIA",
        "======================================================="
    ]

    # Image Quality
    if image_quality:
        iq_class = image_quality.get("predicted_class", image_quality.get("class", "Unknown"))
        iq_conf = image_quality.get("confidence", 0.0)
        lines.append(f"IMAGE QUALITY:\n  ✓ {iq_class} ({iq_conf * 100:.1f}%)")
    else:
        lines.append("IMAGE QUALITY:\n  Not assessed")
    lines.append("-" * 55)

    # If blocked
    if not referral_decision.get("screening_allowed", False):
        lines.append("DR SCREENING:\n  BLOCKED (Quality gate rejection)")
        lines.append(f"RECOMMENDATION:\n  {referral_decision.get('recommendation', 'Capture another image.')}")
        lines.append(f"\nREASON:\n  {referral_decision.get('reason', 'Image quality substandard.')}")
        lines.append("-" * 55)
        lines.append(f"DISCLAIMER:\n  {referral_decision.get('disclaimer', '')}")
        lines.append("=======================================================")
        return "\n".join(lines)

    # DR Screening
    dr_label = referral_decision.get("dr_label", "Unknown")
    dr_conf = referral_decision.get("dr_confidence", 0.0)
    lines.append(f"DR SCREENING:\n  {dr_label} (Confidence: {dr_conf * 100:.1f}%)")
    lines.append("-" * 55)

    # Risk / Referral
    risk = referral_decision.get("risk_level", "Unknown")
    rec = referral_decision.get("recommendation", "Unknown")
    priority = referral_decision.get("referral_priority", "Unknown")
    lines.append(f"RISK / REFERRAL:\n  Risk Category:     {risk}\n  Referral Priority: {priority}\n  Action:            {rec}")
    lines.append("-" * 55)

    # Why recommendation
    why = referral_decision.get("why_recommendation", "")
    lines.append(f"{why}")
    lines.append("-" * 55)

    # Medical report info if provided
    if medical_report_data:
        lines.append("ADDITIONAL REPORT INFORMATION:")
        if "glucose" in medical_report_data:
            lines.append(f"  ├── Blood Glucose: {medical_report_data['glucose']}")
        if "hba1c" in medical_report_data:
            lines.append(f"  ├── HbA1c:         {medical_report_data['hba1c']}")
        if "blood_pressure" in medical_report_data:
            lines.append(f"  └── Blood Pressure: {medical_report_data['blood_pressure']}")
        lines.append("  Note: Contextual only. Does not modify deterministic DR rule.")
        lines.append("-" * 55)

    lines.append(f"DISCLAIMER:\n  {referral_decision.get('disclaimer', '')}")
    lines.append("=======================================================")
    return "\n".join(lines)
