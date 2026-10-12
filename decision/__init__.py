"""
Decision Package for AI DR Screen
Explainable AI for Diabetic Retinopathy Screening in Rural India

Deterministic, rule-based screening referral decision engine.
"""

from decision.referral_rules import (
    REFERRAL_RULES,
    DR_GRADE_NAMES,
    DR_GRADE_FULL_NAMES,
    DEFAULT_MIN_DR_CONFIDENCE,
    get_rule_for_grade
)
from decision.referral_engine import (
    generate_referral_decision,
    CLINICAL_DISCLAIMER
)
from decision.referral_summary import (
    format_clinical_screening_summary,
    format_screening_summary_text
)

__all__ = [
    "REFERRAL_RULES",
    "DR_GRADE_NAMES",
    "DR_GRADE_FULL_NAMES",
    "DEFAULT_MIN_DR_CONFIDENCE",
    "get_rule_for_grade",
    "generate_referral_decision",
    "CLINICAL_DISCLAIMER",
    "format_clinical_screening_summary",
    "format_screening_summary_text"
]
