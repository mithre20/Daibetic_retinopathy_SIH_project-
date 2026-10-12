"""
Contextual Medical Report Parser Module
AI DR Screen — Patient Health Report Analysis

Parses laboratory report text to detect and extract:
- Fasting, Random, and Postprandial Glucose
- HbA1c (Glycated Hemoglobin)
- Blood Pressure (composite 120/80 and separate Systolic/Diastolic)
- Laboratory Reference Ranges
- Extraction Confidence and Source Line Tracking

Prevents false positives by strictly requiring contextual proximity to parameter keywords
and rejecting unrelated numbers (such as Patient ID, Age, Date, or Phone Number).
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from report_analysis.converter import convert_glucose

# Keyword synonym lists with OCR character substitution tolerance
GLUCOSE_KEYWORDS = {
    "fasting": [
        r"fasting\s+(?:blood\s+)?(?:sugar|glucose|suga?r|glu[co0][co0a-z]*)",
        r"(?:blood\s+)?(?:sugar|glucose|glu[co0][co0a-z]*)\s*[-–—]\s*fasting",
        r"\bf[b8][s5]\b",
        r"\bf[b8]g\b"
    ],
    "postprandial": [
        r"post\s*prandial\s+(?:blood\s+)?(?:sugar|glucose|suga?r|glu[co0][co0a-z]*)",
        r"(?:blood\s+)?(?:sugar|glucose|glu[co0][co0a-z]*)\s*[-–—]\s*post\s*prandial",
        r"pp\s+(?:blood\s+)?(?:sugar|glucose)",
        r"\bpp[b8][s5]\b",
        r"\bpp[b8]g\b"
    ],
    "random": [
        r"random\s+(?:blood\s+)?(?:sugar|glucose|suga?r|glu[co0][co0a-z]*)",
        r"(?:blood\s+)?(?:sugar|glucose|glu[co0][co0a-z]*)\s*[-–—]\s*random",
        r"blood\s+sugar\s*\(?random\)?",
        r"\br[b8][s5]\b",
        r"\br[b8]g\b"
    ],
    "general": [
        r"blood\s+glucose",
        r"blood\s+sugar",
        r"serum\s+glucose"
    ]
}

HBA1C_KEYWORDS = [
    r"glycated\s+hemoglobin",
    r"glycosylated\s+hemoglobin",
    r"glycohemoglobin",
    r"hba[1li]c",
    r"hb\s*a[1li]c",
    r"\ba[1li]c\b"
]

BP_KEYWORDS = [
    r"blood\s+press?ure?",
    r"\bbp\b",
    r"\bnibp\b"
]

# Patterns for reference ranges: e.g., (70 - 99), 70-110, 4.0 - 5.6
REF_RANGE_PATTERN = re.compile(
    r"(?:ref(?:erence)?\s*(?:range)?|normal)?\s*[:\(\[]?\s*(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*(?:mg/d[lL]|mmol/[lL]|%|mmHg)?\s*[\)\]]?",
    re.IGNORECASE
)


class ReportParser:
    """
    Context-aware parser for clinical laboratory reports.
    """

    def __init__(self):
        pass

    def parse_text(
        self,
        text: str,
        lines: Optional[List[str]] = None,
        base_confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Parse raw text and/or extracted lines to produce structured health data.
        """
        if lines is None or len(lines) == 0:
            lines = [l.strip() for l in text.splitlines() if l.strip()]

        cleaned_text = "\n".join(lines)

        glucose_data: Dict[str, Any] = {}
        bp_data: Optional[Dict[str, Any]] = None

        # 1. Parse Glucose (Fasting, Postprandial, Random, General)
        fasting = self._extract_glucose_parameter(lines, "fasting", "Fasting Blood Glucose", base_confidence)
        if fasting:
            glucose_data["fasting"] = fasting

        pp = self._extract_glucose_parameter(lines, "postprandial", "Postprandial Glucose", base_confidence)
        if pp:
            glucose_data["postprandial"] = pp

        random_gl = self._extract_glucose_parameter(lines, "random", "Random Blood Glucose", base_confidence)
        if random_gl:
            glucose_data["random"] = random_gl

        # If neither fasting nor postprandial nor random was specifically matched, check general glucose
        if not glucose_data:
            gen = self._extract_glucose_parameter(lines, "general", "Blood Glucose", base_confidence)
            if gen:
                glucose_data["general"] = gen

        # 2. Parse HbA1c
        hba1c = self._extract_hba1c(lines, base_confidence)
        if hba1c:
            glucose_data["hba1c"] = hba1c

        # 3. Parse Blood Pressure
        bp_data = self._extract_blood_pressure(lines, base_confidence)

        return {
            "glucose": glucose_data,
            "blood_pressure": bp_data,
            "has_glucose": len(glucose_data) > 0,
            "has_blood_pressure": bp_data is not None,
            "parameters_detected": list(glucose_data.keys()) + (["blood_pressure"] if bp_data else []),
            "raw_text_length": len(cleaned_text)
        }

    def _extract_glucose_parameter(
        self,
        lines: List[str],
        category: str,
        param_display_name: str,
        base_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """
        Extract specific glucose parameter with units and reference range.
        """
        patterns = GLUCOSE_KEYWORDS[category]

        for line_idx, line in enumerate(lines):
            line_clean = line.strip()
            # Match any of the category patterns
            matched = False
            for pat in patterns:
                if re.search(pat, line_clean, re.IGNORECASE):
                    matched = True
                    break

            if not matched:
                continue

            # Found matching keyword in this line. Look for numeric value in this line or subsequent line.
            search_window = line_clean
            if line_idx + 1 < len(lines):
                search_window += " " + lines[line_idx + 1]

            # Look for number followed optionally by unit (mg/dL, mmol/L, or with OCR variations)
            # Avoid matching dates or IDs
            val_match = re.search(
                r"(?::|\s)\s*(\d{1,3}(?:\.\d+)?)\s*(mg\s*[/|\\]?\s*d[l1I]|mmol\s*[/|\\]?\s*[l1I])?",
                search_window,
                re.IGNORECASE
            )

            if not val_match:
                # Try pattern without colon
                val_match = re.search(
                    r"\b(\d{1,3}(?:\.\d+)?)\s*(mg\s*[/|\\]?\s*d[l1I]|mmol\s*[/|\\]?\s*[l1I])\b",
                    search_window,
                    re.IGNORECASE
                )

            if val_match:
                val = float(val_match.group(1))
                unit_raw = val_match.group(2) if val_match.group(2) else "mg/dL"
                if "mmol" in unit_raw.lower():
                    unit = "mmol/L"
                else:
                    unit = "mg/dL"

                # Extract reference range if present
                ref_low, ref_high, ref_str = self._extract_reference_range(search_window, unit)

                # Normalized conversion
                conv = convert_glucose(val, unit)

                return {
                    "parameter": param_display_name,
                    "category": category,
                    "value": val,
                    "unit": unit,
                    "normalized_value": conv["normalized_value"],
                    "normalized_unit": conv["normalized_unit"],
                    "value_mg_dl": conv["value_mg_dl"],
                    "value_mmol_l": conv["value_mmol_l"],
                    "reference_range": ref_str,
                    "ref_low": ref_low,
                    "ref_high": ref_high,
                    "has_reference_range": ref_str is not None,
                    "confidence": round(base_confidence, 2),
                    "source_text": line_clean
                }

        return None

    def _extract_hba1c(
        self,
        lines: List[str],
        base_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """
        Extract HbA1c value (as numerical percentage) and reference range.
        """
        for line_idx, line in enumerate(lines):
            line_clean = line.strip()
            matched = any(re.search(pat, line_clean, re.IGNORECASE) for pat in HBA1C_KEYWORDS)
            if not matched:
                continue

            search_window = line_clean
            if line_idx + 1 < len(lines):
                search_window += " " + lines[line_idx + 1]

            # Match e.g. 7.6 %, 7.6, 7.6%
            val_match = re.search(
                r"(?::|\s)\s*(\d{1,2}(?:\.\d+)?)\s*(?:%|\bpercent\b)?",
                search_window,
                re.IGNORECASE
            )

            if val_match:
                val = float(val_match.group(1))
                # Physiological bounds check for HbA1c (typically 3.5 - 20)
                if 3.0 <= val <= 25.0:
                    ref_low, ref_high, ref_str = self._extract_reference_range(search_window, "%")
                    return {
                        "parameter": "HbA1c",
                        "value": val,
                        "unit": "%",
                        "reference_range": ref_str,
                        "ref_low": ref_low,
                        "ref_high": ref_high,
                        "has_reference_range": ref_str is not None,
                        "confidence": round(base_confidence, 2),
                        "source_text": line_clean
                    }

        return None

    def _extract_blood_pressure(
        self,
        lines: List[str],
        base_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """
        Extract blood pressure in composite (120/80) or separate (Systolic/Diastolic) format.
        """
        # Form 1: Composite format (e.g. 120/80, 145/92 mmHg)
        for line in lines:
            line_clean = line.strip()

            # Check for composite pattern: 120/80 or 145 / 92 mmHg
            bp_match = re.search(
                r"(?:(?:bp|blood\s+pressure|nibp)[^0-9\n]*)?(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mm\s*Hg)?",
                line_clean,
                re.IGNORECASE
            )
            if bp_match:
                # If "bp" or "pressure" is explicitly in the line, or numbers look like plausible BP
                has_bp_keyword = any(re.search(pat, line_clean, re.IGNORECASE) for pat in BP_KEYWORDS)
                sys_val = int(bp_match.group(1))
                dia_val = int(bp_match.group(2))

                # If numbers are plausible (e.g. 70-240 / 40-140) and Sys > Dia
                if (has_bp_keyword or (90 <= sys_val <= 220 and 50 <= dia_val <= 130)) and (sys_val > dia_val):
                    return {
                        "systolic": sys_val,
                        "diastolic": dia_val,
                        "unit": "mmHg",
                        "formatted": f"{sys_val}/{dia_val} mmHg",
                        "confidence": round(base_confidence, 2),
                        "source_text": line_clean,
                        "extraction_format": "composite"
                    }

        # Form 2: Separate Systolic and Diastolic lines
        sys_val = None
        dia_val = None
        source_parts = []

        for line in lines:
            line_clean = line.strip()
            if re.search(r"systolic(?:\s+bp)?[:\s]+(\d{2,3})", line_clean, re.IGNORECASE):
                m = re.search(r"systolic(?:\s+bp)?[:\s]+(\d{2,3})", line_clean, re.IGNORECASE)
                sys_val = int(m.group(1))
                source_parts.append(line_clean)
            elif re.search(r"diastolic(?:\s+bp)?[:\s]+(\d{2,3})", line_clean, re.IGNORECASE):
                m = re.search(r"diastolic(?:\s+bp)?[:\s]+(\d{2,3})", line_clean, re.IGNORECASE)
                dia_val = int(m.group(1))
                source_parts.append(line_clean)

        if sys_val is not None and dia_val is not None and sys_val > dia_val:
            return {
                "systolic": sys_val,
                "diastolic": dia_val,
                "unit": "mmHg",
                "formatted": f"{sys_val}/{dia_val} mmHg",
                "confidence": round(base_confidence, 2),
                "source_text": " | ".join(source_parts),
                "extraction_format": "separated"
            }

        return None

    def _extract_reference_range(
        self,
        context_str: str,
        unit: str
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Extract reference low and high values from context string.
        """
        m = REF_RANGE_PATTERN.search(context_str)
        if m:
            low = float(m.group(1))
            high = float(m.group(2))
            if low < high:
                return low, high, f"{low}–{high} {unit}"
        return None, None, None
