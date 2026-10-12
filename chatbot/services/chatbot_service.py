"""
Core Chatbot Service and Multi-Stage Conversational Pipeline Coordinator
Orchestrates:
1. Emergency Triage (Sight-threatening symptoms)
2. Medical Safety & Adversarial Guardrails (Refusal of diagnosis/prescriptions)
3. Input Normalization (Typo correction, slang expansion, repeated character squashing)
4. Friendly Conversational Intent Layer (Greetings, how are you, who are you, capabilities, thanks, goodbye, help)
5. Domain Intent Routing with Confidence Scoring (Screening, Upload, IQA, AI, Referral, Navigation, Healthcare)
6. Structured Action & Quick Replies Generation
7. MongoDB Message & Session Logging (with graceful offline fallback)
8. RAG Knowledge Retrieval & LLM Generation (with local deterministic knowledge fallback)
"""

import re
import uuid
from typing import Dict, Any, Optional, List, Tuple

from chatbot.safety.safety import check_medical_safety
from chatbot.safety.emergency import check_emergency_symptoms
from chatbot.services.normalizer import normalize_text
from chatbot.context.context_manager import get_context_manager
from chatbot.context.screening_context import parse_screening_context, explain_screening_result
from chatbot.rag.retriever import retrieve_relevant_knowledge
from chatbot.services.healthcare_service import get_healthcare_service
from chatbot.services.llm_service import get_llm_service
from chatbot.data.mongo_db import get_mongo_store

# Conversational & Domain Intent Pattern Matchers
INTENT_PATTERNS: Dict[str, List[str]] = {
    # Emergency & Safety (also checked deterministically via safety modules)
    "emergency_symptoms": [
        r"sudden.*(?:loss|blindness|can'?t see)",
        r"curtain.*(?:falling|across|over)",
        r"flashes.*(?:light|floaters)",
        r"severe.*eye pain"
    ],
    "medical_safety": [
        r"(?:prescribe|suggest|what).*(?:medicine|tablet|drug|drops)",
        r"stop.*(?:insulin|medicine)",
        r"diagnose me",
        r"will i (?:become|go) blind"
    ],

    # Conversational Intents (Step 2 to 9)
    "greeting": [
        r"^(?:hi|hello|hey|hii|hiii|helo|hlw|howdy|greetings|hey there|hello there|good day)[!.,? ]*$",
        r"good\s+(?:morning|afternoon|evening|night)",
        r"^(?:வணக்கம்|நல்வரவு)",
        r"^(?:नमस्ते|नमस्कार|प्रणाम|शुभ प्रभात|शुभ संध्या)"
    ],
    "welcome": [
        r"^(?:welcome|you are welcome|youre welcome|welcome back)[!.,? ]*$"
    ],
    "how_are_you": [
        r"\bhow (?:are|r) (?:you|u)\b",
        r"\bhow(?:'s|s|\s+is) it going\b",
        r"\bhow are things\b",
        r"\bare you okay\b",
        r"\bhow do you do\b"
    ],
    "identity": [
        r"\bwho (?:are|r) (?:you|u)\b",
        r"\bwhat (?:are|r) (?:you|u)\b",
        r"\bare you (?:ai|a bot|a chatbot|an assistant|human|oculyn)\b",
        r"\bwhat is this (?:bot|chatbot|assistant)\b",
        r"\b(?:who|what) is oculyn\b",
        r"\boculyn\b",
        r"\bwho made you\b",
        r"\btell me about yourself\b"
    ],
    "capabilities": [
        r"\bwhat can (?:you|u) do\b",
        r"\bwhat do (?:you|u) do\b",
        r"\bwhat are your (?:features|capabilities|functions)\b",
        r"help me understand what you can do",
        r"list your capabilities",
        r"show (?:me )?features",
        r"what is your role"
    ],
    "thanks": [
        r"^(?:thanks|thank you|thank u|thx|thnx|thanx|thank you so much|thanks a lot|great thanks|that was helpful|thats helpful|that's helpful|perfect thanks|appreciated)[!., ]*$",
        r"\bthat(?:'s|s|\s+was|\s+is)\s+helpful\b"
    ],
    "goodbye": [
        r"^(?:bye|goodbye|see you|see ya|cya|bye bye|talk to you later|see you later|i am leaving|im leaving|that is all|thats all|that's all|exit)[!., ]*$"
    ],
    "casual_conversation": [
        r"^(?:nice|great|okay|ok|okk|cool|awesome|fine|good|perfect|understood|got it|sounds good)[!., ]*$",
        r"(?:nice|good|helpful|smart|cool|great) (?:chatbot|bot|assistant)",
        r"you(?:'re| are) (?:helpful|smart|great|good|awesome|nice)"
    ],
    "help": [
        r"^(?:help|i need help|can you help me|help me|i am confused|im confused|what should i do|what to do next|guide me|confused)[!.,? ]*$",
        r"\bcan you help\b",
        r"\bwhat should i do\b",
        r"\bi am confused\b",
        r"\bi need guidance\b"
    ],


    # Domain Intents (Step 11 to 19)
    "start_screening": [
        r"start.*screen",
        r"how.*to.*screen",
        r"begin.*screen",
        r"new.*screen",
        r"check my eye",
        r"check eye",
        r"i want to check my eye"
    ],
    "image_upload": [
        r"upload.*image",
        r"upload.*scan",
        r"replace.*image",
        r"upload.*eye",
        r"file format",
        r"file size",
        r"selfie",
        r"normal eye photo",
        r"choose image",
        r"how do i upload an eye image"
    ],
    "image_quality": [
        r"image quality",
        r"iqa",
        r"rejected",
        r"blurry",
        r"too dark",
        r"glare",
        r"poor quality",
        r"unusable",
        r"quality assessment",
        r"why can't i continue",
        r"why was my image rejected",
        r"why does the system stop"
    ],
    "image_capture": [
        r"capture.*better",
        r"camera.*steady",
        r"how to capture",
        r"capture guidance",
        r"avoid glare",
        r"position.*eye",
        r"blink"
    ],
    "report_upload": [
        r"medical report",
        r"upload.*report",
        r"upload.*pdf",
        r"ocr",
        r"blood report",
        r"extract report",
        r"enter glucose"
    ],
    "health_parameters": [
        r"glucose",
        r"fbs",
        r"rbs",
        r"ppbs",
        r"hba1c",
        r"blood pressure",
        r"systolic",
        r"diastolic",
        r"measure glucose"
    ],
    "explainable_ai": [
        r"grad-cam",
        r"gradcam",
        r"heatmap",
        r"shap",
        r"u-net",
        r"unet",
        r"segmentation",
        r"microaneurysm",
        r"hemorrhage",
        r"exudate",
        r"positive contribution",
        r"negative contribution",
        r"why is this part highlighted"
    ],
    "screening_result": [
        r"dr grade",
        r"grade [0-4]",
        r"mild dr",
        r"moderate dr",
        r"severe dr",
        r"proliferative",
        r"what does my result mean",
        r"understand result",
        r"confidence",
        r"efficientnet"
    ],
    "risk_referral": [
        r"low risk",
        r"moderate risk",
        r"high risk",
        r"why am i moderate risk",
        r"why am i high risk",
        r"referral",
        r"priority",
        r"why am i",
        r"ophthalmologist",
        r"download.*result",
        r"what should i do after screening"
    ],
    "navigation": [
        r"go to",
        r"open page",
        r"navigate",
        r"switch to",
        r"open hero",
        r"open login",
        r"open dashboard",
        r"open medivision",
        r"open pipeline",
        r"where is the (?:screening|upload|report|dashboard|login|medivision|home|hero|results) page",
        r"where do i upload",
        r"where can i see my results",
        r"where is the home page",
        r"shortcuts",
        r"keyboard shortcuts"
    ],
    "healthcare_contact": [
        r"hospital",
        r"clinic",
        r"doctor",
        r"center",
        r"phc",
        r"nearest eye hospital",
        r"appointment",
        r"apollo hospital",
        r"contact thirunallur",
        r"take to my eye appointment"
    ],
    "privacy_security": [
        r"privacy",
        r"confidential",
        r"data security",
        r"other patient",
        r"safe",
        r"protect",
        r"public internet"
    ],
    "technical_support": [
        r"server",
        r"offline",
        r"not working",
        r"stutter",
        r"failed to fetch",
        r"timeout",
        r"technical error",
        r"connection offline"
    ],
    "general_info": [
        r"what is ai dr screen",
        r"what does ai dr screen do",
        r"who can use",
        r"how it works",
        r"is it free",
        r"diagnostic system",
        r"xairis",
        r"abdm",
        r"station 01",
        r"mobile phone",
        r"which browsers",
        r"how long does screening take",
        r"do i need an account",
        r"what does the chatbot do",
        r"what can the chatbot help me with",
        r"tell me about xairis",
        r"what is diabetic retinopathy",
        r"why is early screening important",
        r"does ai replace eye doctors"
    ]
}

# Standard quick reply chip sets
QUICK_REPLIES_DEFAULT = [
    "Start Screening",
    "Upload Eye Image",
    "Upload Report",
    "Understand Result"
]

QUICK_REPLIES_CAPABILITIES = [
    "👁 Start Screening",
    "📷 Upload Eye Image",
    "📄 Upload Report",
    "📊 Understand Result",
    "🔍 Explain AI Result",
    "🏥 Find Healthcare"
]

QUICK_REPLIES_RESULTS = [
    "Understand Result",
    "Explain AI Result",
    "Find Healthcare",
    "Download Report"
]

QUICK_REPLIES_AI = [
    "Explain Grad-CAM",
    "Explain SHAP",
    "Explain U-Net",
    "Understand Result"
]

class ChatbotService:
    def __init__(self):
        self.context_mgr = get_context_manager()
        self.healthcare_srv = get_healthcare_service()
        self.llm_srv = get_llm_service()
        self.mongo_store = get_mongo_store()

    def route_intent_with_confidence(self, message: str) -> Tuple[str, float]:
        """
        Classifies normalized user query into one of the 26 standardized intents
        with associated confidence score.
        """
        msg_norm = normalize_text(message)

        # 1. Exact priority conversational matches
        if re.search(r"good\s+morning", msg_norm):
            return "greeting", 0.99
        if re.search(r"good\s+afternoon", msg_norm):
            return "greeting", 0.99
        if re.search(r"good\s+evening", msg_norm):
            return "greeting", 0.99
        if re.search(r"good\s+night", msg_norm):
            return "greeting", 0.98

        # 2. Check each intent in prioritized order
        for intent, patterns in INTENT_PATTERNS.items():
            for p in patterns:
                if re.search(p, msg_norm, re.IGNORECASE):
                    # Higher confidence for exact anchored matches
                    if p.startswith("^") and p.endswith("$"):
                        return intent, 0.99
                    return intent, 0.95

        # 3. High-level keyword heuristics with word boundaries
        if re.search(r"\b(hi|hello|hey|greetings)\b", msg_norm):
            return "greeting", 0.92
        if re.search(r"\b(thank|thanks|thx|appreciate)\b", msg_norm):
            return "thanks", 0.94
        if re.search(r"\b(bye|goodbye|farewell)\b", msg_norm):
            return "goodbye", 0.94
        if re.search(r"\b(help|confused|guide)\b", msg_norm):
            return "help", 0.92

        # 4. Fallback: query knowledge base to see if semantic match exists
        retrieved_context, sources = retrieve_relevant_knowledge(msg_norm, top_k=1)
        if retrieved_context and "No specific knowledge base documents matched" not in retrieved_context:
            return "general_info", 0.80

        # 5. Unknown intent
        return "unknown", 0.50

    def route_intent(self, message: str) -> str:
        """
        Backwards-compatible route_intent returning only the intent string.
        """
        intent, _ = self.route_intent_with_confidence(message)
        return intent

    def _generate_conversational_response(self, intent: str, raw_msg: str, language: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """
        Produces natural, friendly, non-robotic responses for conversational intents.
        """
        lower = normalize_text(raw_msg)
        actions: List[Dict[str, Any]] = []
        quick_replies = QUICK_REPLIES_DEFAULT

        if intent == "greeting":
            if "morning" in lower:
                if language == "ta":
                    msg = "காலை வணக்கம்! ☀️ AI DR Screen-க்கு உங்களை வரவேற்கிறோம். இன்று நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?"
                elif language == "hi":
                    msg = "शुभ प्रभात! ☀️ AI DR Screen में आपका स्वागत है। आज मैं आपकी क्या सहायता कर सकता हूँ?"
                else:
                    msg = "Good morning! ☀️ Welcome to AI DR Screen. How can I help you today?"
            elif "afternoon" in lower:
                if language == "ta":
                    msg = "மதிய வணக்கம்! ☀️ AI DR Screen-க்கு உங்களை வரவேற்கிறோம். இன்று நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?"
                elif language == "hi":
                    msg = "शुभ दोपहर! ☀️ AI DR Screen में आपका स्वागत है। मैं आपकी क्या मदद कर सकता हूँ?"
                else:
                    msg = "Good afternoon! ☀️ Welcome to AI DR Screen. How can I help you today?"
            elif "evening" in lower or "night" in lower:
                if language == "ta":
                    msg = "மாலை வணக்கம்! 🌙 AI DR Screen-க்கு உங்களை வரவேற்கிறோம். இன்று நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?"
                elif language == "hi":
                    msg = "शुभ संध्या! 🌙 AI DR Screen में आपका स्वागत है। मैं आपकी क्या मदद कर सकता हूँ?"
                else:
                    msg = "Good evening! 🌙 How can I help you with AI DR Screen today?"
            else:
                if language == "ta":
                    msg = "வணக்கம்! 👋 AI DR Screen-க்கு உங்களை வரவேற்கிறோம். விழித்திரை பரிசோதனை, பட பதிவேற்றம், மருத்துவ அறிக்கைகள் அல்லது இணையதள வழிகாட்டலுக்கு நான் உங்களுக்கு உதவ முடியும். நீங்கள் என்ன செய்ய விரும்புகிறீர்கள்?"
                elif language == "hi":
                    msg = "नमस्ते! 👋 AI DR Screen में आपका स्वागत है। मैं रेटिना स्क्रीनिंग, इमेज अपलोड, मेडिकल रिपोर्ट और वेबसाइट नेविगेशन में आपकी मदद कर सकता हूँ। आप क्या करना चाहते हैं?"
                else:
                    msg = "Hi! 👋 Welcome to AI DR Screen. I'm here to help you with retinal screening, image uploads, medical reports, results, and navigating the website. What would you like to do?"
            return msg, actions, quick_replies

        if intent == "welcome":
            if language == "ta":
                msg = "மிக்க நன்றி! 😊 AI DR Screen-ஐ பயன்படுத்த நான் உங்களுக்கு உதவ இங்கே இருக்கிறேன். பரிசோதனை, பட பதிவேற்றம் அல்லது முடிவுகள் பற்றி நீங்கள் கேட்கலாம்."
            elif language == "hi":
                msg = "बहुत धन्यवाद! 😊 मैं AI DR Screen का उपयोग करने में आपकी सहायता के लिए यहाँ हूँ। आप स्क्रीनिंग, इमेज अपलोड या परिणामों के बारे में पूछ सकते हैं।"
            else:
                msg = "Thank you! 😊 I'm here to help you use AI DR Screen. You can ask me about screening, image uploads, reports, results, or anything about the website."
            return msg, actions, quick_replies

        if intent == "how_are_you":
            if language == "ta":
                msg = "நான் மிகவும் நலமாக இருக்கிறேன், நன்றி! 😊 AI DR Screen-ல் உங்களுக்கு உதவ நான் தயாராக உள்ளேன். நீங்கள் என்ன தெரிந்து கொள்ள விரும்புகிறீர்கள்?"
            elif language == "hi":
                msg = "मैं बहुत अच्छा हूँ, धन्यवाद! 😊 मैं AI DR Screen में आपकी मदद के लिए पूरी तरह तैयार हूँ। आप क्या जानना चाहते हैं?"
            else:
                msg = "I'm doing great, thank you! 😊 I'm ready to help you with AI DR Screen. What would you like to explore?"
            return msg, actions, quick_replies

        if intent == "identity":
            if language == "ta":
                msg = "வணக்கம்! நான் **Oculyn**, உங்கள் **AI DR Screen உதவியாளர்** 🤖. உங்களுக்கு எப்படி உள்ளது என்று சொல்லுங்கள், நான் உதவ தயாராக உள்ளேன்! இணையதளத்தை இயக்கவும், விழித்திரை பரிசோதனை செயல்முறையை புரிந்துகொள்ளவும், படங்கள் மற்றும் அறிக்கைகளை பதிவேற்றவும், AI முடிவுகளை புரிந்துகொள்ளவும் நான் வழிகாட்டுகிறேன். நான் வழிகாட்டுதல் மட்டுமே வழங்குகிறேன், இறுதி மருத்துவ நோயறிதல் அல்ல."
            elif language == "hi":
                msg = "नमस्ते! मैं **Oculyn**, आपका **AI DR Screen सहायक** 🤖 हूँ। मुझे बताएं कि आप कैसा महसूस कर रहे हैं, मैं आपकी मदद के लिए यहाँ हूँ! मैं प्लेटफॉर्म नेविगेशन, स्क्रीनिंग प्रक्रिया को समझने, रेटिना इमेज या रिपोर्ट अपलोड करने और AI परिणामों को समझाने में आपकी मदद करता हूँ।"
            else:
                msg = "Hi! I'm **Oculyn**, your AI DR Screen Assistant 🤖. Tell me how you feel, I'm here to help! I can help you navigate the platform, understand the screening process, upload retinal images or reports, explain AI results (Grad-CAM, SHAP, U-Net), and guide you to nearby eye hospitals. I provide information and screening guidance, not a medical diagnosis."
            return msg, actions, quick_replies

        if intent == "capabilities":
            if language == "ta":
                msg = (
                    "நான் உங்களுக்கு பின்வருவனவற்றில் உதவ முடியும் 😊\n\n"
                    "👁️ விழித்திரை பரிசோதனையைத் தொடங்குங்கள்\n"
                    "📷 கண் புகைப்படத்தை பதிவேற்றுங்கள்\n"
                    "📄 மருத்துவ அறிக்கைகளை பதிவேற்றி ஆய்வு செய்யுங்கள்\n"
                    "📊 பரிசோதனை முடிவுகளைப் புரிந்து கொள்ளுங்கள்\n"
                    "🔍 Grad-CAM, SHAP மற்றும் U-Net முடிவுகளைப் புரிந்து கொள்ளுங்கள்\n"
                    "🏥 அரசு கண் மருத்துவமனைகளைக் கண்டறியுங்கள்\n"
                    "🧭 இணையதளத்தை எளிதாக இயக்குங்கள்\n"
                    "🔐 தனியுரிமை மற்றும் பாதுகாப்பு விதிகளைத் தெரிந்து கொள்ளுங்கள்\n"
                    "❓ AI DR Screen பற்றிய கேள்விகளுக்கு விடை பெறுங்கள்\n\n"
                    "நீங்கள் எதைத் தொடங்க விரும்புகிறீர்கள்?"
                )
            elif language == "hi":
                msg = (
                    "मैं आपकी इन चीज़ों में मदद कर सकता हूँ 😊\n\n"
                    "👁️ रेटिना स्क्रीनिंग शुरू करें\n"
                    "📷 आंख की फोटो अपलोड करें\n"
                    "📄 मेडिकल रिपोर्ट अपलोड और विश्लेषण करें\n"
                    "📊 स्क्रीनिंग परिणाम समझें\n"
                    "🔍 Grad-CAM, SHAP और U-Net परिणाम समझें\n"
                    "🏥 निकटतम नेत्र अस्पताल खोजें\n"
                    "🧭 वेबसाइट नेविगेट करें\n"
                    "🔐 गोपनीयता और सुरक्षा नियम जानें\n"
                    "❓ AI DR Screen के बारे में प्रश्न पूछें\n\n"
                    "आप क्या करना चाहेंगे?"
                )
            else:
                msg = (
                    "I can help you with 😊\n\n"
                    "👁️ Start a retinal screening\n"
                    "📷 Upload or capture an eye image\n"
                    "📄 Upload and understand medical reports\n"
                    "📊 Understand screening results\n"
                    "🔍 Explain Grad-CAM, SHAP and U-Net results\n"
                    "🏥 Find available healthcare centers\n"
                    "🧭 Navigate the website\n"
                    "🔐 Explain privacy and safety\n"
                    "❓ Answer questions about AI DR Screen\n\n"
                    "What would you like to do?"
                )
            return msg, actions, QUICK_REPLIES_CAPABILITIES

        if intent == "thanks":
            if language == "ta":
                msg = "மிக்க மகிழ்ச்சி! 😊 உங்களுக்கு வேறு ஏதேனும் உதவி தேவைப்பட்டால் தயங்காமல் கேளுங்கள்."
            elif language == "hi":
                msg = "आपका स्वागत है! 😊 यदि आपको किसी और चीज़ में सहायता चाहिए, तो अवश्य पूछें।"
            else:
                msg = "You're very welcome! 😊 If you need anything else, just ask."
            return msg, actions, quick_replies

        if intent == "goodbye":
            if language == "ta":
                msg = "வணக்கம்! 👋 கவனித்துக்கொள்ளுங்கள். AI DR Screen-ல் உதவி தேவைப்படும்போது மீண்டும் வாருங்கள்."
            elif language == "hi":
                msg = "अलविदा! 👋 अपना ध्यान रखें, और AI DR Screen में सहायता की आवश्यकता होने पर कभी भी वापस आएं।"
            else:
                msg = "Goodbye! 👋 Take care, and feel free to come back whenever you need help with AI DR Screen."
            return msg, actions, quick_replies

        if intent == "casual_conversation":
            if any(w in lower for w in ["nice", "helpful", "smart", "good bot", "cool"]):
                if language == "ta":
                    msg = "நன்றி! 😊 உங்கள் பாராட்டுக்கு மகிழ்ச்சி. AI DR Screen-ல் அடுத்து என்ன செய்ய விரும்புகிறீர்கள்?"
                elif language == "hi":
                    msg = "धन्यवाद! 😊 यह जानकर खुशी हुई। आप AI DR Screen में आगे क्या करना चाहेंगे?"
                else:
                    msg = "Thank you! 😊 I'm glad to hear that. What would you like to explore next with AI DR Screen?"
            else:
                if language == "ta":
                    msg = "சரி! 😊 பரிசோதனை தொடங்க, படம் பதிவேற்ற அல்லது முடிவுகளை அறிய நீங்கள் என்னை கேட்கலாம்."
                elif language == "hi":
                    msg = "बहुत बढ़िया! 😊 स्क्रीनिंग शुरू करने, फोटो अपलोड करने या नतीजे समझने के लिए मुझे बताएं।"
                else:
                    msg = "Great! 😊 Let me know if you'd like to start a screening, upload an eye image, check a report, or understand a result."
            return msg, actions, quick_replies

        if intent == "help":
            if "confused" in lower:
                if language == "ta":
                    msg = "கவலை வேண்டாம்! 😊 நான் உங்களுக்கு படிப்படியாக வழிகாட்டுகிறேன். நீங்கள் பரிசோதனையைத் தொடங்க விரும்புகிறீர்களா, கண் படத்தை பதிவேற்ற விரும்புகிறீர்களா, அறிக்கையை பதிவேற்றவா அல்லது முடிவுகளை புரிந்து கொள்ளவா?"
                elif language == "hi":
                    msg = "कोई बात नहीं! 😊 मैं आपको एक-एक कदम समझाता हूँ। क्या आप स्क्रीनिंग शुरू करना चाहते हैं, फोटो अपलोड करना चाहते हैं, रिपोर्ट अपलोड करना चाहते हैं या परिणाम समझना चाहते हैं?"
                else:
                    msg = "No problem! 😊 I can guide you step by step. Are you trying to start a screening, upload an eye image, upload a medical report, or understand a result?"
            elif "what should i do" in lower:
                if language == "ta":
                    msg = "நான் வழிகாட்டுகிறேன். நீங்கள் பரிசோதனை செய்ய விரும்பினால், நிலையம் 05-ல் (Station 05 Pipeline) விழித்திரை படத்தை பதிவேற்றுங்கள். அடுத்த படிகளில் நான் உங்களுக்கு உதவுகிறேன்."
                elif language == "hi":
                    msg = "मैं आपका मार्गदर्शन करता हूँ। यदि आप स्क्रीनिंग शुरू कर रहे हैं, तो स्टेशन 05 पर रेटिना फंडस इमेज अपलोड करें। इसके बाद के चरणों में मैं आपकी सहायता करूँगा।"
                else:
                    msg = "I can guide you. If you're starting a screening, begin by uploading or capturing a retinal fundus image on Station 05 (AI Pipeline). I'll help you through the next steps."
                actions.append({"type": "navigate", "target": "pipeline"})
            else:
                if language == "ta":
                    msg = "நிச்சயமாக! 😊 நீங்கள் என்ன செய்ய முயற்சிக்கிறீர்கள் என்று சொல்லுங்கள், நான் உங்களுக்கு வழிகாட்டுகிறேன்."
                elif language == "hi":
                    msg = "बिल्कुल! 😊 आप क्या करने की कोशिश कर रहे हैं, मुझे बताएं और मैं आपका मार्गदर्शन करूँगा।"
                else:
                    msg = "Of course! 😊 Tell me what you're trying to do, and I'll guide you step by step."
            return msg, actions, quick_replies

        # Fallback unknown clarification
        if language == "ta":
            msg = "நீங்கள் என்ன கேட்கிறீர்கள் என்பது முழுமையாக புரியவில்லை 😊. விழித்திரை பரிசோதனை, பட பதிவேற்றம், மருத்துவ அறிக்கைகள், AI விளக்கங்கள், முடிவுகள் அல்லது இணையதள வழிகாட்டலுக்கு என்னால் உதவ முடியும். சிறிது விரிவாக கூற முடியுமா?"
        elif language == "hi":
            msg = "मुझे आपकी बात पूरी तरह समझ नहीं आई 😊। मैं रेटिना स्क्रीनिंग, इमेज अपलोड, मेडिकल रिपोर्ट, AI विश्लेषण या वेबसाइट नेविगेशन में मदद कर सकता हूँ। क्या आप थोड़ा और विस्तार से बता सकते हैं?"
        else:
            msg = "I'm not completely sure what you mean 😊. I can help with retinal screening, image uploads, medical reports, AI explanations, results, healthcare guidance, or website navigation. Could you tell me a little more?"

        return msg, actions, quick_replies

    def _generate_navigation_response(self, raw_msg: str, language: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Provides accurate routing guidance to actual website stations:
        Station 01: Hero Overview ('hero')
        Station 02: Staff Login ('login')
        Station 03: Screening Dashboard ('dashboard')
        Station 04: MediVision Station ('medivision')
        Station 05: 5-Stage Explainable AI Pipeline ('pipeline')
        """
        lower = normalize_text(raw_msg)
        actions = []

        if "dashboard" in lower:
            target = "dashboard"
            station_name = "Station 03: Main Screening Dashboard"
        elif "login" in lower:
            target = "login"
            station_name = "Station 02: Staff Login"
        elif "medivision" in lower:
            target = "medivision"
            station_name = "Station 04: MediVision Clinical Station"
        elif any(w in lower for w in ["hero", "home"]):
            target = "hero"
            station_name = "Station 01: Hero Overview"
        elif any(w in lower for w in ["upload", "pipeline", "screen", "report", "result"]):
            target = "pipeline"
            station_name = "Station 05: 5-Stage Explainable AI Pipeline"
        else:
            target = "pipeline"
            station_name = "Station 05: 5-Stage Explainable AI Pipeline"

        actions.append({"type": "navigate", "target": target})

        if "shortcut" in lower:
            msg = (
                "You can navigate quickly between stations using keyboard shortcuts:\n\n"
                "• **Alt + 1**: Station 01 — Hero Overview\n"
                "• **Alt + 2**: Station 02 — Staff Login\n"
                "• **Alt + 3**: Station 03 — Screening Dashboard\n"
                "• **Alt + 4**: Station 04 — MediVision Station\n"
                "• **Alt + 5**: Station 05 — AI Pipeline (Upload & Results)\n"
                "• **ArrowLeft / ArrowRight**: Step to previous/next station\n\n"
                "You can also click the station tabs along the top navigation bar."
            )
        elif "where do i upload" in lower or "where to upload" in lower:
            msg = (
                "You can upload your retinal image on **Station 05: 5-Stage Explainable AI Pipeline**. "
                "Look for the upload card at Station 05, where you can drag and drop a fundus photograph or click to browse files."
            )
        elif "where can i see my result" in lower or "where is the result" in lower:
            msg = (
                "Screening results are displayed directly on **Station 05: 5-Stage Explainable AI Pipeline** "
                "(in the Results section after running analysis) and in **Station 03: Screening Dashboard** for aggregate patient history."
            )
        else:
            msg = f"Sure! Let's switch to **{station_name}**. You can also use the top navigation bar or Alt+1..5 shortcuts anytime."

        return msg, actions

    def process_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        page: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        language: str = "en",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main end-to-end processing pipeline adhering to the 34-step specification:
        1. Validation
        2. Emergency Symptoms Detection
        3. Medical Safety & Adversarial Pre-Check
        4. Normalization & Intent Routing
        5. Casual/Conversational handling
        6. Navigation & Structured Actions
        7. Active Screening Context Explanation
        8. Healthcare Directory Grounding
        9. RAG Knowledge Retrieval & LLM Generation
        10. Post-Safety Verification
        11. Session Memory & MongoDB Logging
        """
        if not message or not message.strip():
            return {
                "answer": "Please enter a question or select one of the quick actions below 😊.",
                "message": "Please enter a question or select one of the quick actions below 😊.",
                "intent": "general_info",
                "confidence": 0.99,
                "actions": [],
                "quick_replies": QUICK_REPLIES_DEFAULT,
                "sources": [],
                "safety_flag": False
            }

        sess_id = session_id or str(uuid.uuid4())
        raw_msg = message.strip()

        # Language auto-detection / switch check
        norm_msg = normalize_text(raw_msg)
        lang = language if language in ["en", "ta", "hi"] else "en"
        if "தமிழில்" in raw_msg or "tamil" in norm_msg.split():
            lang = "ta"
        elif "हिंदी" in raw_msg or "hindi" in norm_msg.split():
            lang = "hi"

        # -------------------------------------------------------------
        # 1. EMERGENCY SAFETY CHECK (Immediate Priority)
        # -------------------------------------------------------------
        emergency_eval = check_emergency_symptoms(raw_msg, language=lang)
        if emergency_eval:
            self.context_mgr.add_turn(sess_id, "user", raw_msg)
            self.context_mgr.add_turn(sess_id, "assistant", emergency_eval["answer"])
            self.mongo_store.record_message(sess_id, "user", raw_msg, intent="emergency_symptoms", confidence=1.0, user_id=user_id)
            self.mongo_store.record_message(sess_id, "assistant", emergency_eval["answer"], intent="emergency_symptoms", confidence=1.0, user_id=user_id)
            return {
                "answer": emergency_eval["answer"],
                "message": emergency_eval["answer"],
                "intent": "emergency_symptoms",
                "confidence": 1.0,
                "actions": [],
                "quick_replies": ["Nearest Hospital", "Contact PHC"],
                "sources": emergency_eval.get("sources", ["knowledge/safety/medical_safety.md"]),
                "safety_flag": True
            }

        # -------------------------------------------------------------
        # 2. MEDICAL SAFETY & ADVERSARIAL PRE-CHECK
        # -------------------------------------------------------------
        safety_eval = check_medical_safety(raw_msg, language=lang)
        if safety_eval:
            self.context_mgr.add_turn(sess_id, "user", raw_msg)
            self.context_mgr.add_turn(sess_id, "assistant", safety_eval["answer"])
            self.mongo_store.record_message(sess_id, "user", raw_msg, intent=safety_eval["intent"], confidence=0.99, user_id=user_id)
            self.mongo_store.record_message(sess_id, "assistant", safety_eval["answer"], intent=safety_eval["intent"], confidence=0.99, user_id=user_id)
            return {
                "answer": safety_eval["answer"],
                "message": safety_eval["answer"],
                "intent": safety_eval["intent"],
                "confidence": 0.99,
                "actions": [],
                "quick_replies": QUICK_REPLIES_DEFAULT,
                "sources": ["knowledge/safety/medical_safety.md"],
                "safety_flag": True
            }

        # -------------------------------------------------------------
        # 3. NORMALIZATION & INTENT ROUTING
        # -------------------------------------------------------------
        intent, confidence = self.route_intent_with_confidence(raw_msg)
        actions: List[Dict[str, Any]] = []
        sources: List[str] = []
        quick_replies = QUICK_REPLIES_DEFAULT

        # -------------------------------------------------------------
        # 4. CONVERSATIONAL INTENTS LAYER (Step 2 to 9, 20)
        # -------------------------------------------------------------
        conversational_intents = [
            "greeting", "welcome", "how_are_you", "identity",
            "capabilities", "thanks", "goodbye", "casual_conversation",
            "help", "unknown"
        ]
        if intent in conversational_intents:
            answer, actions, quick_replies = self._generate_conversational_response(intent, raw_msg, lang)
            self.context_mgr.add_turn(sess_id, "user", raw_msg)
            self.context_mgr.add_turn(sess_id, "assistant", answer)
            self.mongo_store.record_message(sess_id, "user", raw_msg, intent=intent, confidence=confidence, user_id=user_id)
            self.mongo_store.record_message(sess_id, "assistant", answer, intent=intent, confidence=confidence, user_id=user_id)
            return {
                "answer": answer,
                "message": answer,
                "intent": intent,
                "confidence": confidence,
                "actions": actions,
                "quick_replies": quick_replies,
                "sources": [],
                "safety_flag": False
            }

        # -------------------------------------------------------------
        # 5. NAVIGATION INTENTS (Step 17)
        # -------------------------------------------------------------
        if intent == "navigation":
            answer, actions = self._generate_navigation_response(raw_msg, lang)
            self.context_mgr.add_turn(sess_id, "user", raw_msg)
            self.context_mgr.add_turn(sess_id, "assistant", answer)
            self.mongo_store.record_message(sess_id, "user", raw_msg, intent=intent, confidence=confidence, user_id=user_id)
            self.mongo_store.record_message(sess_id, "assistant", answer, intent=intent, confidence=confidence, user_id=user_id)
            return {
                "answer": answer,
                "message": answer,
                "intent": intent,
                "confidence": confidence,
                "actions": actions,
                "quick_replies": ["Start Screening", "Screening Dashboard", "Keyboard Shortcuts"],
                "sources": ["knowledge/navigation/navigation_shortcuts.md"],
                "safety_flag": False
            }

        # -------------------------------------------------------------
        # 6. CONTEXT INGESTION & SCREENING RESULTS (Step 11, 12, 13)
        # -------------------------------------------------------------
        parsed_context = parse_screening_context(context)

        # Special Case A: IQA / Quality questions when IQA is blocked
        if (intent in ["image_quality", "screening_result"] or "why can't i continue" in norm_msg or "rejected" in norm_msg) and parsed_context["screening_allowed"] is False:
            answer = explain_screening_result(parsed_context, language=lang)
            self.context_mgr.add_turn(sess_id, "user", raw_msg)
            self.context_mgr.add_turn(sess_id, "assistant", answer)
            self.mongo_store.record_message(sess_id, "user", raw_msg, intent="image_quality", confidence=0.98, user_id=user_id)
            self.mongo_store.record_message(sess_id, "assistant", answer, intent="image_quality", confidence=0.98, user_id=user_id)
            return {
                "answer": answer,
                "message": answer,
                "intent": "image_quality",
                "confidence": 0.98,
                "actions": [{"type": "navigate", "target": "pipeline"}],
                "quick_replies": ["How to Capture Better Image", "Supported Image Formats", "Contact Support"],
                "sources": ["knowledge/image/image_quality.md"],
                "safety_flag": False
            }

        # Special Case B: User specifically asking to explain active screening result
        if parsed_context["has_screening"] and parsed_context["dr_grade"] is not None and (
            "my result" in norm_msg or "why am i" in norm_msg or "explain result" in norm_msg or "understand result" in norm_msg or "what does my result mean" in norm_msg
        ):
            answer = explain_screening_result(parsed_context, language=lang)
            self.context_mgr.add_turn(sess_id, "user", raw_msg)
            self.context_mgr.add_turn(sess_id, "assistant", answer)
            self.mongo_store.record_message(sess_id, "user", raw_msg, intent="risk_referral", confidence=0.98, user_id=user_id)
            self.mongo_store.record_message(sess_id, "assistant", answer, intent="risk_referral", confidence=0.98, user_id=user_id)
            return {
                "answer": answer,
                "message": answer,
                "intent": "risk_referral",
                "confidence": 0.98,
                "actions": [],
                "quick_replies": QUICK_REPLIES_RESULTS,
                "sources": ["knowledge/referral/referral_engine.md"],
                "safety_flag": False
            }

        # Special Case C: Healthcare directory inquiry
        if intent == "healthcare_contact" and any(w in norm_msg for w in ["where", "nearest", "find", "hospital", "phc", "center"]):
            answer = self.healthcare_srv.format_centers_for_user(language=lang)
            self.context_mgr.add_turn(sess_id, "user", raw_msg)
            self.context_mgr.add_turn(sess_id, "assistant", answer)
            self.mongo_store.record_message(sess_id, "user", raw_msg, intent="healthcare_contact", confidence=0.97, user_id=user_id)
            self.mongo_store.record_message(sess_id, "assistant", answer, intent="healthcare_contact", confidence=0.97, user_id=user_id)
            return {
                "answer": answer,
                "message": answer,
                "intent": "healthcare_contact",
                "confidence": 0.97,
                "actions": [],
                "quick_replies": ["Thirunallur PHC", "District General Hospital", "What to Take to Appointment"],
                "sources": ["chatbot/data/healthcare_centers.json"],
                "safety_flag": False
            }

        # -------------------------------------------------------------
        # 7. RAG KNOWLEDGE RETRIEVAL (Step 25)
        # -------------------------------------------------------------
        retrieved_context, sources = retrieve_relevant_knowledge(raw_msg, top_k=4)

        # -------------------------------------------------------------
        # 8. SCREENING CONTEXT FORMATTING & MULTI-TURN MEMORY
        # -------------------------------------------------------------
        screening_summary_str = explain_screening_result(parsed_context, language=lang)
        history = self.context_mgr.get_history(sess_id)

        # -------------------------------------------------------------
        # 9. LLM GENERATION & DETERMINISTIC FALLBACK
        # -------------------------------------------------------------
        raw_answer = self.llm_srv.generate_response(
            user_message=raw_msg,
            retrieved_context=retrieved_context,
            screening_context_str=screening_summary_str,
            conversation_history=history,
            language=lang
        )

        # -------------------------------------------------------------
        # 10. POST-SAFETY VERIFICATION (Step 26)
        # -------------------------------------------------------------
        final_answer = raw_answer
        # Never claim eye is healthy if IQA was blocked
        if parsed_context["screening_allowed"] is False and "your eye is healthy" in final_answer.lower():
            final_answer = final_answer.replace("your eye is healthy", "the image was not sufficient for screening")

        # Never prescribe or claim definitive diagnosis in freeform text
        if any(w in final_answer.lower() for w in ["i prescribe", "i diagnose you with", "take this medicine", "stop taking your"]):
            final_answer = "Please consult a qualified ophthalmologist or doctor for clinical examination and prescriptions. AI DR Screen provides screening decision-support only."

        # Dynamic quick replies based on intent
        if intent in ["explainable_ai"]:
            quick_replies = QUICK_REPLIES_AI
        elif intent in ["screening_result", "risk_referral"]:
            quick_replies = QUICK_REPLIES_RESULTS
        elif intent in ["image_upload", "start_screening"]:
            quick_replies = ["Upload Eye Image", "Supported Image Formats", "How to Capture", "IQA Quality Rules"]

        # Navigation action hook if user asked how to start screening
        if intent == "start_screening":
            actions.append({"type": "navigate", "target": "pipeline"})

        # Record turns in memory and Mongo
        self.context_mgr.add_turn(sess_id, "user", raw_msg)
        self.context_mgr.add_turn(sess_id, "assistant", final_answer)
        self.mongo_store.record_message(sess_id, "user", raw_msg, intent=intent, confidence=confidence, user_id=user_id)
        self.mongo_store.record_message(sess_id, "assistant", final_answer, intent=intent, confidence=confidence, user_id=user_id)

        return {
            "answer": final_answer,
            "message": final_answer,
            "intent": intent,
            "confidence": confidence,
            "actions": actions,
            "quick_replies": quick_replies,
            "sources": sources,
            "safety_flag": False
        }

_chatbot_service = None

def get_chatbot_service() -> ChatbotService:
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service
