"""
Medical Safety and Adversarial Guardrails
Deterministic pre-LLM filters enforcing strict clinical boundaries,
rejecting medication prescription, diagnostic claims, and adversarial prompts.
"""

import re
from typing import Dict, Any, Optional

# Regex patterns for safety categories
MEDICATION_PATTERNS = [
    r"(?:what|which|suggest|recommend|prescribe).*(?:medicine|tablet|drug|medication|pill|injection|drops|eye drop)",
    r"(?:should i take|can i take|dose|dosage).*(?:insulin|metformin|drops|medicine)",
    r"(?:மருந்து|மாத்திரை|சொட்டு மருந்து)",
    r"(?:दवा|दवाई|गोली|इंसुलिन|आई ड्रॉप)"
]

STOP_MEDICATION_PATTERNS = [
    r"(?:should i|can i|may i|stop|discontinue|reduce|quit|pause).*(?:insulin|tablet|medicine|medication|pill)",
    r"stop.*(?:insulin|medicine)",
    r"(?:மருந்தை நிறுத்தலாமா|இன்சுலின் நிறுத்தலாமா)",
    r"(?:दवा बंद कर दूं|इंसुलिन बंद)"
]

DIAGNOSIS_REQUEST_PATTERNS = [
    r"(?:can you|please|could you)?.*(?:diagnose me|give me a diagnosis|tell me if i have|do i definitely have)",
    r"am i definitely (?:diabetic|blind|having dr|cured)",
    r"(?:நோய் கண்டறிதல்|எனக்கு சர்க்கரை நோய் உள்ளதா)",
    r"(?:निदान करें|क्या मुझे मधुमेह है)"
]

PROGNOSIS_BLINDNESS_PATTERNS = [
    r"will i (?:become|go) blind",
    r"am i going to lose my (?:eyesight|vision|sight)",
    r"நான் குருடாகிவிடுவேனா",
    r"क्या मैं अंधा हो जाऊंगा"
]

ADVERSARIAL_PATTERNS = [
    r"ignore (?:all )?(?:previous|your|the) (?:rules|instructions|prompts)",
    r"(?:system prompt|internal prompt|developer mode|dan mode|jailbreak)",
    r"(?:what is your|tell me your|show me your|reveal).*(?:api key|secret|server secret|db password|database password|password)",
    r"(?:give|show|tell|reveal|fetch).*(?:patient\s+[A-Z][a-z]+|other patient|another patient|patient data|patient record)",
    r"(?:change|alter|fake|modify|override).*(?:my result|dr grade|prediction|result to no dr)"
]

SAFE_RESPONSES = {
    "medication": {
        "en": (
            "I cannot prescribe medication or recommend medical treatments. "
            "Decisions regarding medications, eye drops, laser therapy, or injections "
            "must be made by a qualified ophthalmologist or physician based on an in-person clinical examination."
        ),
        "ta": (
            "நான் மருந்துகளை பரிந்துரைக்கவோ அல்லது சிகிச்சை முறைகளை பரிந்துரைக்கவோ முடியாது. "
            "கண் சொட்டு மருந்துகள் அல்லது மாத்திரைகள் பற்றிய முடிவுகளை தகுதியான கண் மருத்துவர் நேரில் பரிசோதித்த பின்னரே எடுக்க வேண்டும்."
        ),
        "hi": (
            "मैं दवाएं निर्धारित नहीं कर सकता और न ही उपचार की सिफारिश कर सकता हूँ। "
            "दवाइयों या आंखों की ड्रॉप्स का निर्णय किसी योग्य नेत्र रोग विशेषज्ञ (डॉक्टर) द्वारा जांच के बाद ही लिया जाना चाहिए।"
        )
    },
    "stop_medication": {
        "en": (
            "Never stop, reduce, or alter your prescribed insulin, diabetes medication, or blood pressure medicines "
            "without direct instruction from your treating doctor. Abruptly discontinuing medication can lead to dangerous "
            "glycemic spikes that accelerate retinal and organ damage."
        ),
        "ta": (
            "உங்கள் மருத்துவரிடம் ஆலோசிக்காமல் இன்சுலின் அல்லது சர்க்கரை மாத்திரைகளை ஒருபோதும் நிறுத்தவோ குறைக்கவோ கூடாது. "
            "மருந்தை திடீரென நிறுத்துவது இரத்த சர்க்கரையை அதிகரித்து கண்களுக்கு கடுமையான பாதிப்பை ஏற்படுத்தலாம்."
        ),
        "hi": (
            "अपने डॉक्टर की सलाह के बिना अपनी निर्धारित इंसुलिन या मधुमेह की दवाएं कभी भी बंद या कम न करें। "
            "दवा बंद करने से ब्लड शुगर अनियंत्रित हो सकता है और आंखों को गंभीर नुकसान पहुंच सकता है।"
        )
    },
    "diagnosis": {
        "en": (
            "I cannot provide a medical diagnosis. AI DR Screen is an automated screening and triage assistant "
            "designed to identify risk levels and facilitate clinical referrals. Please consult an eye specialist "
            "for a comprehensive dilated examination and definitive clinical diagnosis."
        ),
        "ta": (
            "என்னால் மருத்துவ நோயறிதலை (Diagnosis) வழங்க முடியாது. AI DR Screen என்பது ஆபத்து நிலையை கண்டறிந்து பரிந்துரைக்கும் ஒரு திரையிடல் கருவி மட்டுமே. "
            "முழுமையான பரிசோதனைக்கு கண் மருத்துவரை அணுகவும்."
        ),
        "hi": (
            "मैं चिकित्सीय निदान (Diagnosis) प्रदान नहीं कर सकता। AI DR Screen केवल एक स्क्रीनिंग और ट्राइएज सहायक है। "
            "पुख्ता जांच और निदान के लिए कृपया नेत्र रोग विशेषज्ञ से मिलें।"
        )
    },
    "blindness": {
        "en": (
            "Diabetic Retinopathy is manageable, and timely treatment can prevent vision loss in the vast majority of patients. "
            "Early screening, maintaining controlled blood sugar and blood pressure, and timely evaluation by an ophthalmologist "
            "are the most effective steps to protect your sight."
        ),
        "ta": (
            "நீரிழிவு விழித்திரை பாதிப்பு (Diabetic Retinopathy) ஆரம்பத்திலேயே கண்டறியப்பட்டால் சிகிச்சையளிக்கக்கூடியது. "
            "இரத்த சர்க்கரை மற்றும் இரத்த அழுத்தத்தை கட்டுப்பாட்டில் வைத்து, சரியான நேரத்தில் கண் மருத்துவரை அணுகுவது உங்கள் பார்வையை பாதுகாக்கும்."
        ),
        "hi": (
            "डायबिटिक रेटिनोपैथी का समय पर पता चलने पर उचित इलाज संभव है और अधिकांश मामलों में दृष्टि की रक्षा की जा सकती है। "
            "समय पर जांच, शुगर और बीपी पर नियंत्रण तथा डॉक्टर से परामर्श ही सबसे सुरक्षित उपाय है।"
        )
    },
    "adversarial": {
        "en": (
            "I am designed to operate strictly within clinical screening boundaries. I cannot modify screening results, "
            "disclose internal configurations, or access unauthorized patient records."
        ),
        "ta": (
            "நான் மருத்துவ வழிகாட்டுதல் விதிகளுக்கு உட்பட்டு மட்டுமே செயல்பட முடியும். பரிசோதனை முடிவுகளை மாற்றவோ அல்லது பிற தகவல்களை வெளியிடவோ இயலாது."
        ),
        "hi": (
            "मैं पूरी तरह से चिकित्सा सुरक्षा नियमों के तहत कार्य करता हूँ। मैं परिणामों में बदलाव नहीं कर सकता और न ही गोपनीय जानकारी साझा कर सकता हूँ।"
        )
    }
}

def check_medical_safety(message: str, language: str = "en") -> Optional[Dict[str, Any]]:
    """
    Evaluates user message against deterministic medical safety and security rules.
    Returns safe structured refusal dictionary if triggered, or None if safe.
    """
    if not message or not isinstance(message, str):
        return None

    clean_text = message.strip().lower()
    lang = language if language in ["en", "ta", "hi"] else "en"

    # 1. Adversarial & jailbreak check
    for p in ADVERSARIAL_PATTERNS:
        if re.search(p, clean_text, re.IGNORECASE):
            return {
                "safe": False,
                "intent": "medical_safety",
                "answer": SAFE_RESPONSES["adversarial"][lang],
                "safety_flag": True,
                "category": "adversarial"
            }

    # 2. Stop medication check
    for p in STOP_MEDICATION_PATTERNS:
        if re.search(p, clean_text, re.IGNORECASE):
            return {
                "safe": False,
                "intent": "medical_safety",
                "answer": SAFE_RESPONSES["stop_medication"][lang],
                "safety_flag": True,
                "category": "stop_medication"
            }

    # 3. Prescribe medication check
    for p in MEDICATION_PATTERNS:
        if re.search(p, clean_text, re.IGNORECASE):
            return {
                "safe": False,
                "intent": "medical_safety",
                "answer": SAFE_RESPONSES["medication"][lang],
                "safety_flag": True,
                "category": "medication_prescribe"
            }

    # 4. Direct diagnosis request
    for p in DIAGNOSIS_REQUEST_PATTERNS:
        if re.search(p, clean_text, re.IGNORECASE):
            return {
                "safe": False,
                "intent": "medical_safety",
                "answer": SAFE_RESPONSES["diagnosis"][lang],
                "safety_flag": True,
                "category": "diagnosis_demand"
            }

    # 5. Prognosis / Blindness concern
    for p in PROGNOSIS_BLINDNESS_PATTERNS:
        if re.search(p, clean_text, re.IGNORECASE):
            return {
                "safe": False,
                "intent": "medical_safety",
                "answer": SAFE_RESPONSES["blindness"][lang],
                "safety_flag": True,
                "category": "prognosis_blindness"
            }

    return None
