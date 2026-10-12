"""
Unit Tests for Retinal DR Risk and Referral Decision Engine
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Test Suite covering:
- Tests 1-5: Deterministic DR severity grade mapping (Grades 0 to 4)
- Tests 6-7: Quality gate pre-triage (Poor Quality, Unusable blocking screening)
- Test 8: Low-confidence prediction safety rule (recommends professional review)
- Test 9: High-severity safety rule (Grades 3 & 4 never downgraded)
- Test 10: Invalid DR grade validation
- Test 11: Missing DR prediction handling
- Test 12: Structured summary and existing pipeline integrity
"""

import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from decision.referral_rules import (
    REFERRAL_RULES,
    DR_GRADE_NAMES,
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


class TestReferralEngine(unittest.TestCase):
    """
    Comprehensive test suite for the rule-based referral decision engine.
    """

    # -------------------------------------------------------------------------
    # Test 1: Grade 0 -> Low Risk, Routine
    # -------------------------------------------------------------------------
    def test_grade_0_routine_referral(self):
        res = generate_referral_decision(
            dr_grade=0,
            dr_confidence=0.88,
            image_quality_status="ACCEPTED",
            image_quality_confidence=0.95
        )
        self.assertTrue(res["screening_allowed"])
        self.assertEqual(res["dr_grade"], 0)
        self.assertEqual(res["dr_label"], "No DR")
        self.assertEqual(res["risk_level"], "Low Risk")
        self.assertEqual(res["referral_priority"], "Routine")
        self.assertEqual(res["recommendation"], "Routine follow-up")
        self.assertEqual(res["decision_status"], "VALID")
        self.assertIn("DR Grade 0", res["why_recommendation"])
        self.assertIn(CLINICAL_DISCLAIMER, res["disclaimer"])

    # -------------------------------------------------------------------------
    # Test 2: Grade 1 -> Moderate Risk, Recommended
    # -------------------------------------------------------------------------
    def test_grade_1_mild_dr(self):
        res = generate_referral_decision(
            dr_grade=1,
            dr_confidence=0.76,
            image_quality_status="ACCEPTED",
            image_quality_confidence=0.92
        )
        self.assertTrue(res["screening_allowed"])
        self.assertEqual(res["dr_grade"], 1)
        self.assertEqual(res["dr_label"], "Mild DR")
        self.assertEqual(res["risk_level"], "Moderate Risk")
        self.assertEqual(res["referral_priority"], "Recommended")
        self.assertEqual(res["recommendation"], "Eye examination recommended")
        self.assertEqual(res["decision_status"], "VALID")
        self.assertIn("DR Grade 1–2", res["why_recommendation"])

    # -------------------------------------------------------------------------
    # Test 3: Grade 2 -> Moderate Risk, Recommended
    # -------------------------------------------------------------------------
    def test_grade_2_moderate_dr(self):
        res = generate_referral_decision(
            dr_grade=2,
            dr_confidence=0.82,
            image_quality_status="ACCEPTED",
            image_quality_confidence=0.90
        )
        self.assertTrue(res["screening_allowed"])
        self.assertEqual(res["dr_grade"], 2)
        self.assertEqual(res["dr_label"], "Moderate DR")
        self.assertEqual(res["risk_level"], "Moderate Risk")
        self.assertEqual(res["referral_priority"], "Recommended")
        self.assertEqual(res["recommendation"], "Eye examination recommended")
        self.assertEqual(res["decision_status"], "VALID")
        self.assertIn("DR Grade 1–2", res["why_recommendation"])

    # -------------------------------------------------------------------------
    # Test 4: Grade 3 -> High Risk, Priority
    # -------------------------------------------------------------------------
    def test_grade_3_severe_dr(self):
        res = generate_referral_decision(
            dr_grade=3,
            dr_confidence=0.89,
            image_quality_status="ACCEPTED",
            image_quality_confidence=0.94
        )
        self.assertTrue(res["screening_allowed"])
        self.assertEqual(res["dr_grade"], 3)
        self.assertEqual(res["dr_label"], "Severe DR")
        self.assertEqual(res["risk_level"], "High Risk")
        self.assertEqual(res["referral_priority"], "Priority")
        self.assertEqual(res["recommendation"], "Priority ophthalmologist referral")
        self.assertEqual(res["decision_status"], "VALID")
        self.assertIn("DR Grade 3–4", res["why_recommendation"])

    # -------------------------------------------------------------------------
    # Test 5: Grade 4 -> High Risk, Priority
    # -------------------------------------------------------------------------
    def test_grade_4_proliferative_dr(self):
        res = generate_referral_decision(
            dr_grade=4,
            dr_confidence=0.93,
            image_quality_status="ACCEPTED",
            image_quality_confidence=0.91
        )
        self.assertTrue(res["screening_allowed"])
        self.assertEqual(res["dr_grade"], 4)
        self.assertEqual(res["dr_label"], "Proliferative DR")
        self.assertEqual(res["risk_level"], "High Risk")
        self.assertEqual(res["referral_priority"], "Priority")
        self.assertEqual(res["recommendation"], "Priority ophthalmologist referral")
        self.assertEqual(res["decision_status"], "VALID")
        self.assertIn("DR Grade 3–4", res["why_recommendation"])

    # -------------------------------------------------------------------------
    # Test 6: Poor Image Quality -> Screening Blocked
    # -------------------------------------------------------------------------
    def test_poor_quality_blocks_screening(self):
        res = generate_referral_decision(
            dr_grade=2,
            dr_confidence=0.85,
            image_quality_status="Poor Quality",
            image_quality_confidence=0.80
        )
        self.assertFalse(res["screening_allowed"])
        self.assertEqual(res["decision_status"], "BLOCKED")
        self.assertIsNone(res["dr_prediction"])
        self.assertIsNone(res["dr_grade"])
        self.assertIsNone(res["risk_level"])
        self.assertIsNone(res["referral_priority"])
        self.assertEqual(res["recommendation"], "Please capture another retinal image.")
        self.assertIn("recapture", res["why_recommendation"].lower())

    # -------------------------------------------------------------------------
    # Test 7: Unusable Image Quality -> Screening Blocked
    # -------------------------------------------------------------------------
    def test_unusable_quality_blocks_screening(self):
        res = generate_referral_decision(
            dr_grade=0,
            dr_confidence=0.95,
            image_quality_status="Unusable",
            image_quality_confidence=0.88
        )
        self.assertFalse(res["screening_allowed"])
        self.assertEqual(res["decision_status"], "BLOCKED")
        self.assertIsNone(res["dr_prediction"])
        self.assertIsNone(res["risk_level"])
        self.assertIsNone(res["referral_priority"])
        self.assertEqual(res["recommendation"], "Please capture another retinal image.")

    # -------------------------------------------------------------------------
    # Test 8: Low-Confidence DR Prediction (Grade 0, 1, 2)
    # -------------------------------------------------------------------------
    def test_low_confidence_dr_prediction(self):
        res = generate_referral_decision(
            dr_grade=2,
            dr_confidence=0.42,
            image_quality_status="ACCEPTED",
            image_quality_confidence=0.95,
            minimum_dr_confidence=0.50
        )
        self.assertTrue(res["screening_allowed"])
        self.assertEqual(res["decision_status"], "LOW_CONFIDENCE")
        self.assertEqual(res["recommendation"], "Professional review recommended")
        self.assertIn("low confidence", res["why_recommendation"].lower())

    # -------------------------------------------------------------------------
    # Test 9: High-Severity Safety Rule (Grade 3 & 4 NEVER downgraded)
    # -------------------------------------------------------------------------
    def test_low_confidence_severe_dr_not_downgraded(self):
        res_grade3 = generate_referral_decision(
            dr_grade=3,
            dr_confidence=0.42,
            image_quality_status="ACCEPTED",
            image_quality_confidence=0.90,
            minimum_dr_confidence=0.50
        )
        self.assertTrue(res_grade3["screening_allowed"])
        self.assertEqual(res_grade3["decision_status"], "LOW_CONFIDENCE")
        # Must NOT be downgraded to Low Risk or Moderate Risk!
        self.assertEqual(res_grade3["risk_level"], "High Risk")
        self.assertEqual(res_grade3["referral_priority"], "Priority")
        self.assertEqual(res_grade3["recommendation"], "Priority ophthalmologist referral")
        self.assertIn("not downgraded", res_grade3["why_recommendation"].lower())

        res_grade4 = generate_referral_decision(
            dr_grade=4,
            dr_confidence=0.38,
            image_quality_status="ACCEPTED",
            image_quality_confidence=0.92,
            minimum_dr_confidence=0.50
        )
        self.assertEqual(res_grade4["risk_level"], "High Risk")
        self.assertEqual(res_grade4["referral_priority"], "Priority")
        self.assertEqual(res_grade4["recommendation"], "Priority ophthalmologist referral")

    # -------------------------------------------------------------------------
    # Test 10: Invalid DR Grade Validation
    # -------------------------------------------------------------------------
    def test_invalid_dr_grade_validation(self):
        invalid_inputs = [-1, 5, 99, "Severe", None]
        for inv in invalid_inputs:
            res = generate_referral_decision(
                dr_grade=inv,
                dr_confidence=0.85,
                image_quality_status="ACCEPTED",
                image_quality_confidence=0.95
            )
            self.assertFalse(res["screening_allowed"])
            self.assertEqual(res["decision_status"], "INVALID_INPUT")
            self.assertIn("error", res)
            self.assertIsNone(res["risk_level"])
            self.assertIsNone(res["referral_priority"])

    # -------------------------------------------------------------------------
    # Test 11: Missing DR Prediction
    # -------------------------------------------------------------------------
    def test_missing_dr_prediction(self):
        res = generate_referral_decision(
            dr_grade=None,
            dr_confidence=None,
            image_quality_status="ACCEPTED",
            image_quality_confidence=0.95
        )
        self.assertFalse(res["screening_allowed"])
        self.assertEqual(res["decision_status"], "INVALID_INPUT")
        self.assertIsNone(res["dr_grade"])
        self.assertIsNone(res["risk_level"])
        self.assertIn("Unable to generate referral recommendation", res["error"])

    # -------------------------------------------------------------------------
    # Test 12: Integrated Clinical Summary and Medical Report Context
    # -------------------------------------------------------------------------
    def test_clinical_summary_with_medical_report(self):
        ref_decision = generate_referral_decision(
            dr_grade=2,
            dr_confidence=0.78,
            image_quality_status="ACCEPTED",
            image_quality_confidence=0.93
        )
        med_report = {
            "glucose": "184 mg/dL",
            "hba1c": "8.6%",
            "blood_pressure": "142/90 mmHg"
        }
        summary = format_clinical_screening_summary(
            referral_decision=ref_decision,
            image_quality={"status": "ACCEPTED", "predicted_class": "Good Quality", "confidence": 0.93},
            medical_report_data=med_report
        )
        self.assertTrue(summary["screening_allowed"])
        self.assertEqual(summary["dr_screening"]["dr_grade"], 2)
        self.assertEqual(summary["risk_referral"]["risk_level"], "Moderate Risk")
        self.assertIn("hba1c", summary["additional_clinical_context"])
        self.assertEqual(summary["additional_clinical_context"]["hba1c"], "8.6%")
        # Ensure text summary matches Section 25
        text_out = format_screening_summary_text(
            referral_decision=ref_decision,
            image_quality={"class": "Good Quality", "confidence": 0.93},
            medical_report_data=med_report
        )
        self.assertIn("IMAGE QUALITY", text_out)
        self.assertIn("DR SCREENING", text_out)
        self.assertIn("RISK / REFERRAL", text_out)
        self.assertIn("Moderate Risk", text_out)
        self.assertIn("HbA1c", text_out)
        self.assertIn("DISCLAIMER", text_out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
