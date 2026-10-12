"""
Glucose and Blood Pressure Unit Conversion Module
AI DR Screen — Patient Health Report Analysis

Provides accurate, bi-directional unit conversion between mg/dL and mmol/L
for glucose measurements without losing or altering original values.
"""

from typing import Dict, Any, Optional, Tuple

# Molecular weight of glucose is ~180.16 g/mol, so 1 mmol/L = 18.0182 mg/dL
# Standard clinical conversion factor:
GLUCOSE_MMOL_TO_MGDL_FACTOR: float = 18.0182


def convert_glucose(
    value: float,
    source_unit: str
) -> Dict[str, Any]:
    """
    Convert glucose value to both mg/dL and mmol/L.
    
    Parameters:
        value (float): Numerical glucose value.
        source_unit (str): Unit of the value ("mg/dL", "mg/dl", "mmol/L", "mmol/l").
        
    Returns:
        Dict[str, Any]: Contains original value/unit, normalized values in both units.
    """
    unit_clean = source_unit.strip()
    unit_lower = unit_clean.lower()

    if "mmol" in unit_lower:
        mg_dl = value * GLUCOSE_MMOL_TO_MGDL_FACTOR
        mmol_l = value
        normalized_unit = "mg/dL"
        normalized_val = round(mg_dl, 1)
    else:  # default or mg/dL
        mg_dl = value
        mmol_l = value / GLUCOSE_MMOL_TO_MGDL_FACTOR
        normalized_unit = "mg/dL"
        normalized_val = round(mg_dl, 1)

    return {
        "original_value": value,
        "original_unit": unit_clean,
        "value_mg_dl": round(mg_dl, 1),
        "value_mmol_l": round(mmol_l, 2),
        "display_mg_dl": f"{round(mg_dl, 1)} mg/dL",
        "display_mmol_l": f"{round(mmol_l, 2)} mmol/L",
        "normalized_value": normalized_val,
        "normalized_unit": normalized_unit
    }
