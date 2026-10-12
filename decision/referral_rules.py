"""
Diabetic Retinopathy Referral Rules Definition
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Centralized, deterministic, rule-based screening policy mapping DR severity grades (0-4)
to clinical risk categories and referral priorities.

IMPORTANT:
These are prototype screening rules designed for automated triage in rural outreach camps.
Actual referral intervals and clinical management must follow locally applicable ophthalmology
guidelines (e.g., ICMR, AIOS, WHO) and the judgment of a licensed ophthalmologist.
"""

from typing import Dict, Any, Optional

# Default engineering threshold for flagging low-confidence AI predictions.
# This is an engineering threshold, NOT a clinically validated cutoff.
DEFAULT_MIN_DR_CONFIDENCE: float = 0.50

# Valid DR Severity Class Mapping (matching EfficientNet-B0 and ResNet50)
DR_GRADE_NAMES: Dict[int, str] = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# Full clinical descriptions for documentation and display
DR_GRADE_FULL_NAMES: Dict[int, str] = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-Proliferative Diabetic Retinopathy",
    2: "Moderate Non-Proliferative Diabetic Retinopathy",
    3: "Severe Non-Proliferative Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}

# Centralized Referral Rules Mapping
REFERRAL_RULES: Dict[int, Dict[str, Any]] = {
    0: {
        "grade": 0,
        "dr_label": "No DR",
        "dr_full_label": "No Diabetic Retinopathy",
        "risk_level": "Low Risk",
        "priority": "Routine",
        "referral_priority": "Routine",
        "recommendation": "Routine follow-up",
        "clinical_action": "Annual routine retinal screening recommended as per diabetes standard of care.",
        "rule_summary": "DR Grade 0 → Low Risk (Routine follow-up)"
    },
    1: {
        "grade": 1,
        "dr_label": "Mild DR",
        "dr_full_label": "Mild Non-Proliferative Diabetic Retinopathy",
        "risk_level": "Moderate Risk",
        "priority": "Recommended",
        "referral_priority": "Recommended",
        "recommendation": "Eye examination recommended",
        "clinical_action": "Comprehensive eye examination by an optometrist or ophthalmologist within 6 to 12 months.",
        "rule_summary": "DR Grade 1–2 → Moderate Risk (Eye examination recommended)"
    },
    2: {
        "grade": 2,
        "dr_label": "Moderate DR",
        "dr_full_label": "Moderate Non-Proliferative Diabetic Retinopathy",
        "risk_level": "Moderate Risk",
        "priority": "Recommended",
        "referral_priority": "Recommended",
        "recommendation": "Eye examination recommended",
        "clinical_action": "Comprehensive ophthalmic examination within 3 to 6 months to evaluate progression and macular status.",
        "rule_summary": "DR Grade 1–2 → Moderate Risk (Eye examination recommended)"
    },
    3: {
        "grade": 3,
        "dr_label": "Severe DR",
        "dr_full_label": "Severe Non-Proliferative Diabetic Retinopathy",
        "risk_level": "High Risk",
        "priority": "Priority",
        "referral_priority": "Priority",
        "recommendation": "Priority ophthalmologist referral",
        "clinical_action": "Prompt referral to a retinal specialist/ophthalmologist within 2 to 4 weeks for clinical evaluation and staging.",
        "rule_summary": "DR Grade 3–4 → High Risk (Priority ophthalmologist referral)"
    },
    4: {
        "grade": 4,
        "dr_label": "Proliferative DR",
        "dr_full_label": "Proliferative Diabetic Retinopathy",
        "risk_level": "High Risk",
        "priority": "Priority",
        "referral_priority": "Priority",
        "recommendation": "Priority ophthalmologist referral",
        "clinical_action": "Urgent referral to an ophthalmologist/vitreoretinal specialist for immediate intervention (e.g. panretinal photocoagulation or anti-VEGF therapy).",
        "rule_summary": "DR Grade 3–4 → High Risk (Priority ophthalmologist referral)"
    }
}


def get_rule_for_grade(grade: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve rule dictionary for a given DR severity grade (0-4).
    Returns None if grade is invalid.
    """
    if not isinstance(grade, int) or grade not in REFERRAL_RULES:
        return None
    # Return a shallow copy so caller cannot mutate global rules
    return dict(REFERRAL_RULES[grade])
