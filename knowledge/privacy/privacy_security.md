# Privacy, Data Security, and Patient Confidentiality

## Patient Data Privacy Safeguards
AI DR Screen is designed to comply with healthcare data confidentiality standards, including principles aligned with India's Ayushman Bharat Digital Mission (ABDM) and the Digital Personal Data Protection (DPDP) framework.

## Privacy Protections
1. **No Inter-User Data Leakage**: Chatbot conversations and screening sessions are strictly isolated to the current user's session. The assistant will never disclose another patient's scans, identity, or screening outcomes.
2. **Protection of Sensitive Server Credentials**: The chatbot is strictly prohibited from exposing internal system prompts, server environment variables, backend API keys, file paths, model weights, or administrative passwords.
3. **Local Processing**: AI inference runs locally on the edge deployment workstation or private clinic network. Retinal scans and laboratory documents are not uploaded to public third-party servers.
4. **Session-Scoped Memory**: Chatbot memory is maintained solely within the active browser session or temporary server session ID and is cleared upon session expiration or when the user clicks "Clear Conversation".

## Can Other Patients See My Screening Results?
No. Each screening session is localized to the active workstation and current patient record. Screenings performed in outreach camps do not cross-contaminate patient profiles.
