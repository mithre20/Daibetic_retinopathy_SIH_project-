"""
Emergency Symptoms Detection and Escalation
Identifies acute, sight-threatening visual emergencies and provides immediate urgent-care guidance.
"""

import re
from typing import Dict, Any, Optional

EMERGENCY_PATTERNS = [
    r"sudden.*(?:loss of vision|vision loss|blindness|can'?t see|went blind|lost.*vision|lost.*sight)",
    r"(?:blindness|vision loss|lost.*vision).*sudden",
    r"(?:curtain|shadow|veil|dark sheet).*(?:falling|pulling|across|over).*(?:eye|vision)",
    r"sudden.*(?:flashes of light|lightning streaks|shower of floaters|black spots)",
    r"severe.*(?:eye pain|pain in eye).*(?:vomiting|nausea|headache)",
    r"(?:chemical|acid|alkali|bleach|detergent).*(?:in eye|splash|burned my eye)",
    r"(?:injury|trauma|hit in the eye|punctured|penetrating).*(?:eye|bleeding)",
    r"(?:திடீர்|திடீரென).*(?:பார்வை இழப்பு|பார்வை போய்விட்டது|கண் தெரியவில்லை|பார்வை)",
    r"(?:பார்வை இழப்பு|கண் தெரியவில்லை).*(?:திடீர்|திடீரென)",
    r"(?:अचानक|तुरंत).*(?:अंधापन|दृष्टि चली गई|दिखाई नहीं दे रहा|आंख की रोशनी|रोशनी चली गई)",
    r"(?:आंख की रोशनी|दृष्टि|रोशनी).*(?:अचानक|तुरंत|चली गई)"
]

EMERGENCY_RESPONSES = {
    "en": (
        "🚨 **URGENT MEDICAL ALERT — IMMEDIATE ATTENTION REQUIRED** 🚨\n\n"
        "The symptoms you described (such as sudden vision loss, a dark curtain across your visual field, "
        "sudden flashes of light, or severe acute eye pain) are signs of a **potential ophthalmic emergency** "
        "(such as retinal detachment, acute angle-closure glaucoma, or central retinal artery occlusion).\n\n"
        "**What you must do immediately:**\n"
        "1. **Go directly to the nearest Eye Hospital Emergency Department or District Hospital.**\n"
        "2. Do not wait for a routine screening appointment or clinic follow-up.\n"
        "3. Do not rub, press, or apply unprescribed drops to the eye.\n"
        "4. Have someone accompany or drive you immediately to emergency medical care.\n\n"
        "Time is critical in protecting your vision."
    ),
    "ta": (
        "🚨 **அவசர மருத்துவ எச்சரிக்கை — உடனடியாக மருத்துவரை அணுகவும்** 🚨\n\n"
        "நீங்கள் குறிப்பிட்டுள்ள அறிகுறிகள் (திடீர் பார்வை இழப்பு, கண் முன்னே கருப்பு திரை விழுதல், "
        "மின்னல் போன்ற ஒளி அல்லது கடுமையான கண் வலி) **அவசர கண் பாதிப்பின்** அறிகுறிகளாக இருக்கலாம்.\n\n"
        "**உடனடியாக செய்ய வேண்டியவை:**\n"
        "1. **உடனடியாக அருகில் உள்ள கண் மருத்துவமனை அவசர சிகிச்சைப் பிரிவிற்கு (Emergency Ward) செல்லவும்.**\n"
        "2. சாதாரண பரிசோதனைக்காக காத்திருக்க வேண்டாம்.\n"
        "3. கண்களைத் தேய்க்கவோ, தாங்களாக எந்த கண் சொட்டு மருந்தையும் போடவோ வேண்டாம்.\n"
        "4. உடனடியாக குடும்பத்தினர் உதவியுடன் மருத்துவமனைக்கு செல்லவும்."
    ),
    "hi": (
        "🚨 **आपातकालीन चिकित्सा चेतावनी — तुरंत डॉक्टर से संपर्क करें** 🚨\n\n"
        "आपके द्वारा बताए गए लक्षण (जैसे अचानक दृष्टि का चले जाना, आंखों के आगे काला पर्दा आना, "
        "रोशनी की चमक या अत्यधिक तेज दर्द) एक **गंभीर नेत्र आपातकाल** के संकेत हो सकते हैं।\n\n"
        "**आपको तुरंत क्या करना चाहिए:**\n"
        "1. **बिना किसी देरी के निकटतम नेत्र अस्पताल या जिला अस्पताल के आपातकालीन विभाग में जाएं।**\n"
        "2. किसी नियमित जांच या सामान्य समय का इंतजार न करें।\n"
        "3. आंख को बिल्कुल न मलें और कोई अनजानी दवाई न डालें।\n"
        "4. तुरंत किसी परिजन के साथ अस्पताल जाएं।"
    )
}

def check_emergency_symptoms(message: str, language: str = "en") -> Optional[Dict[str, Any]]:
    """
    Evaluates user message for acute sight-threatening emergency indicators.
    Returns structured emergency response if detected, or None if safe.
    """
    if not message or not isinstance(message, str):
        return None

    clean_text = message.strip().lower()

    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, clean_text, re.IGNORECASE):
            lang_key = language if language in EMERGENCY_RESPONSES else "en"
            return {
                "is_emergency": True,
                "intent": "emergency_symptoms",
                "answer": EMERGENCY_RESPONSES[lang_key],
                "safety_flag": True,
                "sources": ["knowledge/safety/medical_safety.md"]
            }

    return None
