"""
Health Parameter Validation and Clinical Interpretation Module
AI DR Screen — Patient Health Report Analysis

Validates physiological plausibility of extracted glucose and blood pressure values,
compares against extracted laboratory reference ranges, and provides cautious,
non-alarmist screening interpretations.
"""

from typing import Dict, Any, Optional, Tuple

# Physiological Bounds for Validation
BOUNDS = {
    "glucose_mg_dl": {"min": 20.0, "max": 1000.0},
    "glucose_mmol_l": {"min": 1.1, "max": 55.5},
    "hba1c": {"min": 3.0, "max": 25.0},
    "systolic_bp": {"min": 60, "max": 260},
    "diastolic_bp": {"min": 40, "max": 160}
}

CLINICAL_DISCLAIMER = (
    "This system is a research and screening prototype. "
    "It does not replace professional medical diagnosis or advice. "
    "Laboratory values are extracted from the uploaded report and should be "
    "verified against the original report."
)


def validate_glucose(
    value: float,
    unit: str = "mg/dL",
    ref_low: Optional[float] = None,
    ref_high: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate a glucose value against physiological plausibility and report reference ranges.
    """
    unit_lower = unit.lower()
    is_mmol = "mmol" in unit_lower
    bounds = BOUNDS["glucose_mmol_l"] if is_mmol else BOUNDS["glucose_mg_dl"]

    # Check physiological plausibility
    is_plausible = bounds["min"] <= value <= bounds["max"]
    if not is_plausible:
        return {
            "valid": False,
            "value": value,
            "unit": unit,
            "error": f"Glucose value {value} {unit} is outside physiologically plausible range ({bounds['min']}-{bounds['max']} {unit}).",
            "interpretation": "Value implausible; please verify original laboratory report.",
            "status": "IMPLAUSIBLE"
        }

    # Reference range evaluation
    status = "RECORDED"
    interpretation = "Recorded from report."
    has_ref = ref_low is not None and ref_high is not None

    if has_ref:
        if value < ref_low:
            status = "BELOW_RANGE"
            interpretation = (
                f"The reported value ({value} {unit}) is below the reference range "
                f"({ref_low}–{ref_high} {unit}) shown in the uploaded report. "
                f"Please discuss this result with a qualified healthcare professional."
            )
        elif value > ref_high:
            status = "ABOVE_RANGE"
            interpretation = (
                f"The reported value ({value} {unit}) is above the reference range "
                f"({ref_low}–{ref_high} {unit}) shown in the uploaded report. "
                f"Please discuss this result with a qualified healthcare professional."
            )
        else:
            status = "WITHIN_RANGE"
            interpretation = f"The reported value is within the laboratory reference range ({ref_low}–{ref_high} {unit})."
    else:
        interpretation = (
            "Reference range was not provided in the uploaded report. "
            "This result should be interpreted using the laboratory's reference range and clinical context."
        )

    return {
        "valid": True,
        "value": value,
        "unit": unit,
        "reference_range": f"{ref_low}–{ref_high} {unit}" if has_ref else None,
        "has_reference_range": has_ref,
        "status": status,
        "interpretation": interpretation
    }


def validate_hba1c(
    value: float,
    ref_low: Optional[float] = None,
    ref_high: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate an HbA1c percentage.
    """
    bounds = BOUNDS["hba1c"]
    is_plausible = bounds["min"] <= value <= bounds["max"]

    if not is_plausible:
        return {
            "valid": False,
            "value": value,
            "unit": "%",
            "error": f"HbA1c value {value}% is outside plausible range ({bounds['min']}%–{bounds['max']}%).",
            "interpretation": "Value implausible; please verify original laboratory report.",
            "status": "IMPLAUSIBLE"
        }

    has_ref = ref_low is not None and ref_high is not None
    if has_ref:
        if value > ref_high:
            status = "ABOVE_RANGE"
            interpretation = (
                f"The reported HbA1c ({value}%) is above the reference range "
                f"({ref_low}–{ref_high}%) shown in the uploaded report. "
                f"Please discuss this finding with a doctor."
            )
        elif value < ref_low:
            status = "BELOW_RANGE"
            interpretation = f"The reported HbA1c ({value}%) is below the reference range shown in the report."
        else:
            status = "WITHIN_RANGE"
            interpretation = f"The reported HbA1c ({value}%) is within the laboratory reference range."
    else:
        status = "RECORDED"
        interpretation = (
            "Reference range was not provided in the uploaded report. "
            "This result should be interpreted using the laboratory's reference range and clinical context."
        )

    return {
        "valid": True,
        "value": value,
        "unit": "%",
        "reference_range": f"{ref_low}–{ref_high} %" if has_ref else None,
        "has_reference_range": has_ref,
        "status": status,
        "interpretation": interpretation
    }


def validate_blood_pressure(
    systolic: int,
    diastolic: int,
    unit: str = "mmHg"
) -> Dict[str, Any]:
    """
    Validate systolic and diastolic blood pressure values.
    """
    s_bounds = BOUNDS["systolic_bp"]
    d_bounds = BOUNDS["diastolic_bp"]

    s_valid = s_bounds["min"] <= systolic <= s_bounds["max"]
    d_valid = d_bounds["min"] <= diastolic <= d_bounds["max"]

    if not s_valid or not d_valid:
        return {
            "valid": False,
            "systolic": systolic,
            "diastolic": diastolic,
            "unit": unit,
            "error": f"Blood pressure {systolic}/{diastolic} {unit} is outside plausible physiological bounds (Systolic: {s_bounds['min']}-{s_bounds['max']}, Diastolic: {d_bounds['min']}-{d_bounds['max']}).",
            "status": "IMPLAUSIBLE"
        }

    if systolic <= diastolic:
        return {
            "valid": False,
            "systolic": systolic,
            "diastolic": diastolic,
            "unit": unit,
            "error": f"Invalid BP reading: Systolic ({systolic}) must be greater than Diastolic ({diastolic}).",
            "status": "INVALID_RELATION"
        }

    # Cautious clinical observation (standard reference thresholds for screening context)
    # Systolic < 120 and Diastolic < 80 is standard normal; >= 140 or >= 90 is elevated
    if systolic >= 140 or diastolic >= 90:
        status = "ELEVATED"
        interpretation = (
            f"The reported blood pressure ({systolic}/{diastolic} {unit}) is elevated. "
            f"Please have your blood pressure evaluated by a healthcare professional."
        )
    elif systolic >= 130 or diastolic >= 80:
        status = "BORDERLINE"
        interpretation = (
            f"The reported blood pressure ({systolic}/{diastolic} {unit}) is in the borderline/elevated range. "
            f"Routine clinical monitoring is advised."
        )
    else:
        status = "STANDARD"
        interpretation = f"The reported blood pressure ({systolic}/{diastolic} {unit}) is within typical standard ranges."

    return {
        "valid": True,
        "systolic": systolic,
        "diastolic": diastolic,
        "unit": unit,
        "status": status,
        "interpretation": interpretation
    }
