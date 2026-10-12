"""
Health Report Analysis and Patient Screening Orchestrator
AI DR Screen — Patient Health Report Analysis

Orchestrates document ingestion, OCR/text extraction, contextual parsing,
physiological validation, manual entry merging, and combined patient screening summary assembly.
"""

from typing import Dict, Any, Optional, Union, List
from report_analysis.extractor import ReportExtractor
from report_analysis.parser import ReportParser
from report_analysis.validator import (
    validate_glucose,
    validate_hba1c,
    validate_blood_pressure,
    CLINICAL_DISCLAIMER
)
from report_analysis.converter import convert_glucose


class HealthReportAnalyzer:
    """
    High-level analyzer for patient medical reports and clinical triage summary generation.
    """

    def __init__(self):
        self.extractor = ReportExtractor()
        self.parser = ReportParser()

    def analyze_report(
        self,
        file_input: Union[str, bytes],
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        End-to-end report document analysis (PDF or Image).
        """
        # Step 1: Extraction
        ext_res = self.extractor.extract_from_file(file_input, filename=filename)
        if not ext_res.get("success", False):
            return {
                "success": False,
                "error": ext_res.get("error", "Unable to extract text from document."),
                "filename": ext_res.get("filename", filename),
                "file_type": ext_res.get("file_type", "UNKNOWN"),
                "glucose": {},
                "blood_pressure": None,
                "disclaimer": CLINICAL_DISCLAIMER
            }

        # Step 2: Contextual Parsing
        parse_res = self.parser.parse_text(
            text=ext_res.get("extracted_text", ""),
            lines=ext_res.get("lines", []),
            base_confidence=ext_res.get("confidence", 0.95)
        )

        glucose_raw = parse_res.get("glucose", {})
        bp_raw = parse_res.get("blood_pressure")

        # Step 3: Validation and Status Flagging
        validated_glucose: Dict[str, Any] = {}
        for key, g_item in glucose_raw.items():
            if key == "hba1c":
                v_res = validate_hba1c(
                    value=g_item["value"],
                    ref_low=g_item.get("ref_low"),
                    ref_high=g_item.get("ref_high")
                )
            else:
                v_res = validate_glucose(
                    value=g_item["value"],
                    unit=g_item["unit"],
                    ref_low=g_item.get("ref_low"),
                    ref_high=g_item.get("ref_high")
                )
            # Merge parsed metadata and validation
            g_combined = dict(g_item)
            g_combined.update(v_res)
            g_combined["source"] = "Uploaded Report"
            validated_glucose[key] = g_combined

        validated_bp = None
        if bp_raw:
            bp_val = validate_blood_pressure(
                systolic=bp_raw["systolic"],
                diastolic=bp_raw["diastolic"],
                unit=bp_raw.get("unit", "mmHg")
            )
            validated_bp = dict(bp_raw)
            validated_bp.update(bp_val)
            validated_bp["source"] = "Uploaded Report"

        overall_confidence = ext_res.get("confidence", 0.95)
        is_low_conf = overall_confidence < 0.70

        return {
            "success": True,
            "filename": ext_res.get("filename", filename),
            "file_type": ext_res.get("file_type", "DOCUMENT"),
            "extraction_method": ext_res.get("extraction_method", "unknown"),
            "extraction_confidence": round(overall_confidence, 2),
            "is_low_confidence": is_low_conf,
            "confidence_warning": (
                "⚠ Low-confidence extraction: Please verify extracted numbers against the uploaded report."
                if is_low_conf else None
            ),
            "glucose": validated_glucose,
            "blood_pressure": validated_bp,
            "has_glucose": len(validated_glucose) > 0,
            "has_blood_pressure": validated_bp is not None,
            "parameters_detected": parse_res.get("parameters_detected", []),
            "raw_lines": ext_res.get("lines", [])[:40],  # Sample lines for user preview
            "disclaimer": CLINICAL_DISCLAIMER
        }

    def merge_with_manual_entries(
        self,
        report_data: Optional[Dict[str, Any]] = None,
        manual_glucose: Optional[Dict[str, Any]] = None,
        manual_bp: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Merge report-extracted data with optional manual entries (Section 11 & 12).
        Preserves and clearly indicates data sources for each entry.
        """
        merged_glucose: Dict[str, Any] = {}
        merged_bp: Optional[Dict[str, Any]] = None
        report_bp: Optional[Dict[str, Any]] = None
        manual_bp_validated: Optional[Dict[str, Any]] = None

        if report_data and report_data.get("glucose"):
            for k, v in report_data["glucose"].items():
                item = dict(v)
                item["source"] = "Uploaded Report"
                merged_glucose[k] = item

        if report_data and report_data.get("blood_pressure"):
            report_bp = dict(report_data["blood_pressure"])
            report_bp["source"] = "Uploaded Report"
            merged_bp = report_bp

        # Process manual glucose
        if manual_glucose and "value" in manual_glucose and manual_glucose["value"] is not None:
            try:
                g_val = float(manual_glucose["value"])
                g_type = str(manual_glucose.get("test_type", "fasting")).lower()
                g_unit = str(manual_glucose.get("unit", "mg/dL"))

                if g_type == "hba1c":
                    val_res = validate_hba1c(g_val)
                    p_name = "HbA1c"
                else:
                    val_res = validate_glucose(g_val, g_unit)
                    p_name = f"{g_type.capitalize()} Blood Glucose"

                conv = convert_glucose(g_val, g_unit) if g_type != "hba1c" else {}

                manual_entry = {
                    "parameter": p_name,
                    "category": g_type,
                    "value": g_val,
                    "unit": g_unit if g_type != "hba1c" else "%",
                    "source": "Manual Entry",
                    "confidence": 1.0,
                    **val_res,
                    **conv
                }
                merged_glucose[f"manual_{g_type}"] = manual_entry
            except (ValueError, TypeError):
                pass

        # Process manual blood pressure
        if manual_bp and "systolic" in manual_bp and "diastolic" in manual_bp:
            try:
                sys_v = int(manual_bp["systolic"])
                dia_v = int(manual_bp["diastolic"])
                bp_val = validate_blood_pressure(sys_v, dia_v, manual_bp.get("unit", "mmHg"))
                manual_bp_validated = {
                    "systolic": sys_v,
                    "diastolic": dia_v,
                    "unit": manual_bp.get("unit", "mmHg"),
                    "formatted": f"{sys_v}/{dia_v} mmHg",
                    "source": "Manual Entry",
                    "confidence": 1.0,
                    **bp_val
                }
            except (ValueError, TypeError):
                pass

        return {
            "glucose": merged_glucose,
            "blood_pressure": manual_bp_validated or report_bp,
            "report_blood_pressure": report_bp,
            "manual_blood_pressure": manual_bp_validated,
            "has_glucose": len(merged_glucose) > 0,
            "has_blood_pressure": bool(manual_bp_validated or report_bp),
            "disclaimer": CLINICAL_DISCLAIMER
        }

    def build_combined_screening_summary(
        self,
        dr_prediction: Optional[Dict[str, Any]] = None,
        health_data: Optional[Dict[str, Any]] = None,
        lesions_data: Optional[Dict[str, Any]] = None,
        referral_decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assemble the unified patient screening summary matching Section 16 & 24.
        Combines AI retinal findings and extracted health measurements without
        any black-box mathematical score formulas.
        """
        # DR Section
        dr_summary = {
            "dr_grade": dr_prediction.get("prediction", dr_prediction.get("class_id")) if dr_prediction else None,
            "dr_label": dr_prediction.get("severity", "Not Assessed") if dr_prediction else "Not Assessed",
            "dr_confidence": dr_prediction.get("confidence", 0.0) if dr_prediction else 0.0,
            "status": "COMPLETED" if dr_prediction else "NOT_AVAILABLE"
        }

        # Lesions Section
        lesions_summary = {
            "microaneurysms": lesions_data.get("microaneurysms", "Not detected") if lesions_data else "Not detected",
            "hemorrhages": lesions_data.get("hemorrhages", "Not detected") if lesions_data else "Not detected",
            "hard_exudates": lesions_data.get("hard_exudates", "Not detected") if lesions_data else "Not detected",
            "soft_exudates": lesions_data.get("soft_exudates", "Not detected") if lesions_data else "Not detected"
        }

        # Health Report Section
        health_summary = {
            "glucose": health_data.get("glucose", {}) if health_data else {},
            "blood_pressure": health_data.get("blood_pressure") if health_data else None,
            "report_blood_pressure": health_data.get("report_blood_pressure") if health_data else None,
            "manual_blood_pressure": health_data.get("manual_blood_pressure") if health_data else None,
            "has_report": bool(health_data and (health_data.get("glucose") or health_data.get("blood_pressure")))
        }

        # Referral Section
        referral_summary = {
            "risk_level": referral_decision.get("risk_level", "Unknown") if referral_decision else "Unknown",
            "referral_priority": referral_decision.get("referral_priority", "Unknown") if referral_decision else "Unknown",
            "recommendation": referral_decision.get("recommendation", "Clinical evaluation recommended") if referral_decision else "Clinical evaluation recommended",
            "why_recommendation": referral_decision.get("why_recommendation", "") if referral_decision else ""
        }

        explanation = (
            "The retinal AI model evaluated the fundus image for diabetic retinopathy features. "
            "Additional glucose and blood pressure information was extracted from the patient's "
            "health report/entry to provide comprehensive clinical context. "
            "Referral recommendations follow transparent, deterministic screening rules."
        )

        return {
            "title": "PATIENT SCREENING SUMMARY",
            "dr_classification": dr_summary,
            "retinal_lesions": lesions_summary,
            "health_report": health_summary,
            "referral_recommendation": referral_summary,
            "explanation": explanation,
            "disclaimer": CLINICAL_DISCLAIMER
        }


# Module-level convenience functions
_analyzer_instance = None

def get_health_analyzer() -> HealthReportAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = HealthReportAnalyzer()
    return _analyzer_instance

def analyze_report(file_input: Union[str, bytes], filename: Optional[str] = None) -> Dict[str, Any]:
    return get_health_analyzer().analyze_report(file_input, filename=filename)
