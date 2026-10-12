"""
Patient Health Report Analysis Package
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Module for ingesting medical reports (PDF/images), extracting glucose and blood pressure,
validating units, and producing unified patient screening summaries.
"""

from report_analysis.converter import convert_glucose, GLUCOSE_MMOL_TO_MGDL_FACTOR
from report_analysis.validator import (
    validate_glucose,
    validate_hba1c,
    validate_blood_pressure,
    CLINICAL_DISCLAIMER
)
from report_analysis.extractor import ReportExtractor, SUPPORTED_EXTENSIONS
from report_analysis.parser import ReportParser
from report_analysis.analyzer import (
    HealthReportAnalyzer,
    get_health_analyzer,
    analyze_report
)

__all__ = [
    "convert_glucose",
    "GLUCOSE_MMOL_TO_MGDL_FACTOR",
    "validate_glucose",
    "validate_hba1c",
    "validate_blood_pressure",
    "CLINICAL_DISCLAIMER",
    "ReportExtractor",
    "SUPPORTED_EXTENSIONS",
    "ReportParser",
    "HealthReportAnalyzer",
    "get_health_analyzer",
    "analyze_report"
]
