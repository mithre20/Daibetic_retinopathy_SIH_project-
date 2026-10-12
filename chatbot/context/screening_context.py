"""
Screening Context Ingestion and Explanation
Reads structured screening session telemetry and explains current findings
using deterministic decision engine rules.
"""

from typing import Dict, Any, Optional
from decision.referral_rules import REFERRAL_RULES, DR_GRADE_NAMES, DR_GRADE_FULL_NAMES

def parse_screening_context(raw_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Standardizes and validates incoming screening session context.
    """
    if not raw_context or not isinstance(raw_context, dict):
        return {
            "has_screening": False,
            "dr_grade": None,
            "dr_label": None,
            "dr_confidence": None,
            "risk_level": None,
            "referral_priority": None,
            "iqa_status": None,
            "screening_allowed": None,
            "unet_findings": None
        }

    dr_grade = raw_context.get("dr_grade")
    if dr_grade is not None:
        try:
            dr_grade = int(dr_grade)
        except (ValueError, TypeError):
            dr_grade = None

    dr_confidence = raw_context.get("dr_confidence")
    if dr_confidence is not None:
        try:
            dr_confidence = float(dr_confidence)
        except (ValueError, TypeError):
            dr_confidence = None

    dr_label = raw_context.get("dr_label")
    if dr_grade is not None and dr_grade in DR_GRADE_NAMES and not dr_label:
        dr_label = DR_GRADE_NAMES[dr_grade]

    risk_level = raw_context.get("risk_level")
    referral_priority = raw_context.get("referral_priority")
    if dr_grade is not None and dr_grade in REFERRAL_RULES:
        rule = REFERRAL_RULES[dr_grade]
        if not risk_level:
            risk_level = rule["risk_level"]
        if not referral_priority:
            referral_priority = rule["referral_priority"]

    iqa_status = raw_context.get("iqa_status")
    screening_allowed = raw_context.get("screening_allowed", True)
    if iqa_status and str(iqa_status).lower() in ["poor", "unusable", "rejected"]:
        screening_allowed = False

    has_screening = (dr_grade is not None) or (iqa_status is not None)

    return {
        "has_screening": has_screening,
        "dr_grade": dr_grade,
        "dr_label": dr_label,
        "dr_confidence": dr_confidence,
        "risk_level": risk_level,
        "referral_priority": referral_priority,
        "iqa_status": iqa_status,
        "screening_allowed": screening_allowed,
        "unet_findings": raw_context.get("unet_findings")
    }

def explain_screening_result(context: Dict[str, Any], language: str = "en") -> str:
    """
    Generates an exact, deterministic natural-language explanation of the current screening result
    grounded directly in the active referral rules and IQA gatekeeper findings.
    """
    parsed = parse_screening_context(context)

    # Case 1: Image Quality blocked
    if parsed["screening_allowed"] is False and parsed["iqa_status"]:
        status = parsed["iqa_status"]
        if language == "ta":
            return (
                f"உங்கள் விழித்திரை படம் தரக் கட்டுப்பாட்டு சோதனையில் (IQA Gate) "
                f"'{status}' என குறிக்கப்பட்டுள்ளது. இதனால் பரிசோதனை நிறுத்தப்பட்டது. "
                "தெளிவற்ற படத்தில் நம்பகமான முடிவுகளை எடுக்க முடியாது. "
                "தயவுசெய்து கண்களை அசைக்காமல், சரியான வெளிச்சத்தில் மீண்டும் ஒரு புகைப்படம் எடுக்கவும்."
            )
        elif language == "hi":
            return (
                f"आपकी रेटिना छवि गुणवत्ता जांच (IQA Gate) में '{status}' पाई गई है, "
                "जिसके कारण स्क्रीनिंग को रोक दिया गया है। अस्पष्ट फोटो पर AI सही परिणाम नहीं दे सकता। "
                "कृपया बेहतर फोकस और बिना चकाचौंध के पुनः एक स्पष्ट फोटो लें।"
            )
        return (
            f"Your retinal image was flagged as '{status}' by the automated Image Quality Assessment (IQA) gate. "
            "Because high-resolution optical clarity is required to prevent diagnostic mistakes, screening was stopped. "
            "Please recapture another retinal photograph with steady focus and no glare."
        )

    # Case 2: No screening performed yet
    if parsed["dr_grade"] is None:
        if language == "ta":
            return "தற்போது உங்களிடம் செயலில் உள்ள பரிசோதனை முடிவுகள் எதுவும் இல்லை. படத்தைப் பதிவேற்றி 'Run Screening' என்பதை அழுத்தவும்."
        elif language == "hi":
            return "वर्तमान सत्र में कोई स्क्रीनिंग परिणाम उपलब्ध नहीं है। कृपया रेटिना फोटो अपलोड करें और 'Run Screening' पर क्लिक करें।"
        return "There is no active screening result in this session. Please upload a retinal fundus scan and click 'Run Screening Pipeline'."

    grade = parsed["dr_grade"]
    conf = parsed["dr_confidence"]
    conf_str = f"{conf * 100:.1f}%" if conf is not None else "N/A"
    rule = REFERRAL_RULES.get(grade, {})
    label = parsed["dr_label"] or rule.get("dr_label", f"Grade {grade}")
    risk = parsed["risk_level"] or rule.get("risk_level", "Unknown")
    priority = parsed["referral_priority"] or rule.get("referral_priority", "Standard")
    rec = rule.get("recommendation", "Clinical evaluation")
    action = rule.get("clinical_action", "")

    # Explain deterministic rule
    if language == "ta":
        return (
            f"**தற்போதைய பரிசோதனை முடிவு:**\n"
            f"- **நிலை:** {label} (நிலை {grade})\n"
            f"- **AI நம்பகத்தன்மை:** {conf_str}\n"
            f"- **ஆபத்து வகை:** {risk}\n"
            f"- **பரிந்துரை முன்னுரிமை:** {priority} ({rec})\n\n"
            f"**மருத்துவ பரிந்துரை:** {action}\n\n"
            f"_குறிப்பு: இது AI திரையிடல் வழிகாட்டுதல் மட்டுமே. முழுமையான பரிசோதனைக்கு கண் மருத்துவரை அணுகவும்._"
        )
    elif language == "hi":
        return (
            f"**वर्तमान स्क्रीनिंग परिणाम:**\n"
            f"- **ग्रेड:** {label} (ग्रेड {grade})\n"
            f"- **AI सटीकता/विश्वास:** {conf_str}\n"
            f"- **जोखिम स्तर:** {risk}\n"
            f"- **रेफरल प्राथमिकता:** {priority} ({rec})\n\n"
            f"**चिकित्सीय सलाह:** {action}\n\n"
            f"_नोट: यह केवल AI स्क्रीनिंग मार्गदर्शन है, अंतिम चिकित्सीय निदान नहीं। कृपया नेत्र विशेषज्ञ से परामर्श लें।_"
        )

    return (
        f"**Current Screening Result:**\n"
        f"- **Detected Severity:** {label} (Grade {grade})\n"
        f"- **Model Confidence:** {conf_str}\n"
        f"- **Risk Category:** {risk}\n"
        f"- **Referral Priority:** {priority} ({rec})\n\n"
        f"**Recommended Action According to AI DR Screen Rules:**\n"
        f"{action}\n\n"
        f"*(Note: AI screening results are not a definitive medical diagnosis. Clinical confirmation by an eye specialist is always recommended.)*"
    )
