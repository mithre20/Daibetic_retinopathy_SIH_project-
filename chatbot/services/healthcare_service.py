"""
Healthcare Services Module for AI DR Screen
Provides access to verified healthcare facilities and enforces strict anti-hallucination policies.
"""

import json
from typing import Dict, Any, List, Optional
from chatbot.config import HEALTHCARE_DATA_PATH

class HealthcareService:
    def __init__(self, data_path=HEALTHCARE_DATA_PATH):
        self.data_path = data_path
        self.centers: List[Dict[str, Any]] = []
        self.policy: str = ""
        self._load_data()

    def _load_data(self):
        try:
            if self.data_path.exists():
                with open(self.data_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    self.centers = payload.get("centers", [])
                    self.policy = payload.get("policy", "")
        except Exception as e:
            print(f"Warning: Failed to load healthcare centers data: {e}")
            self.centers = []

    def get_all_centers(self) -> List[Dict[str, Any]]:
        return list(self.centers)

    def find_center_by_query(self, query: str) -> Optional[Dict[str, Any]]:
        if not query:
            return None
        q_lower = query.lower()
        for c in self.centers:
            if c["name"].lower() in q_lower or c["id"].lower() in q_lower or c["district"].lower() in q_lower:
                return c
        return None

    def format_centers_for_user(self, language: str = "en") -> str:
        """
        Formats verified referral facilities with clear contact guidelines.
        """
        if not self.centers:
            if language == "ta":
                return "தற்போது இந்த அமைப்பில் சரிபார்க்கப்பட்ட மருத்துவமனை விவரங்கள் எதுவும் பதிவு செய்யப்படவில்லை. உங்கள் உள்ளூர் ஆரம்ப சுகாதார நிலையத்தை (PHC) தொடர்பு கொள்ளவும்."
            elif language == "hi":
                return "वर्तमान प्रणाली में कोई सत्यापित अस्पताल विवरण कॉन्फ़िगर नहीं है। कृपया अपने स्थानीय प्राथमिक स्वास्थ्य केंद्र (PHC) से संपर्क करें।"
            return "Specific contact details for healthcare centers are not currently configured in this deployment. Please contact your local Primary Health Center (PHC) or district health helpline."

        if language == "ta":
            lines = ["🏥 **சரிபார்க்கப்பட்ட பரிந்துரை மையங்கள் (Verified Healthcare Centers):**\n"]
            for c in self.centers:
                lines.append(f"- **{c['name']}** ({c['type']})")
                lines.append(f"  - மாவட்டம்: {c['district']}, {c['state']}")
                lines.append(f"  - சேவைகள்: {', '.join(c['services'])}")
                lines.append(f"  - பரிந்துரை பாதை: {c.get('referral_pathway', 'District Hospital')}\n")
            lines.append("📌 *குறிப்பு: பிற தனியார் மருத்துவமனைகள் அல்லது மருத்துவர்களின் தனிப்பட்ட தொலைபேசி எண்கள் இந்த அமைப்பில் பதிவு செய்யப்படவில்லை.*")
            return "\n".join(lines)

        elif language == "hi":
            lines = ["🏥 **सत्यापित रेफरल स्वास्थ्य केंद्र (Verified Healthcare Centers):**\n"]
            for c in self.centers:
                lines.append(f"- **{c['name']}** ({c['type']})")
                lines.append(f"  - जिला: {c['district']}, {c['state']}")
                lines.append(f"  - सेवाएं: {', '.join(c['services'])}")
                lines.append(f"  - रेफरल मार्ग: {c.get('referral_pathway', 'District Hospital')}\n")
            lines.append("📌 *नोट: अन्य निजी अस्पतालों या डॉक्टरों के व्यक्तिगत फोन नंबर इस सिस्टम में दर्ज नहीं हैं।*")
            return "\n".join(lines)

        lines = ["🏥 **Verified Healthcare & Referral Facilities:**\n"]
        for c in self.centers:
            lines.append(f"- **{c['name']}** ({c['type']})")
            lines.append(f"  - Location: {c['district']}, {c['state']}")
            lines.append(f"  - Capabilities: {', '.join(c['services'])}")
            lines.append(f"  - Referral Link: {c.get('referral_pathway', 'District Hospital')}\n")
        lines.append("*(Important: External private clinics or personal doctor phone numbers are not maintained in this verified registry. Please consult your local PHC coordinator for local appointments.)*")
        return "\n".join(lines)

_healthcare_service = None

def get_healthcare_service() -> HealthcareService:
    global _healthcare_service
    if _healthcare_service is None:
        _healthcare_service = HealthcareService()
    return _healthcare_service
