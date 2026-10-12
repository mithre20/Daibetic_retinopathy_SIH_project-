# AI DR Screen Chatbot — REST API Documentation

## 1. Base URL
All chatbot endpoints are served under `/api/chat`:
`http://localhost:5000/api/chat`

---

## 2. Endpoints

### 2.1 Send Message
**`POST /api/chat`**

#### Request Headers
`Content-Type: application/json`

#### Request Payload
```json
{
  "message": "Why am I moderate risk?",
  "session_id": "sess_user_12345",
  "page": "results",
  "language": "en",
  "context": {
    "dr_grade": 2,
    "dr_label": "Moderate DR",
    "dr_confidence": 0.82,
    "risk_level": "Moderate Risk",
    "referral_priority": "Recommended",
    "iqa_status": "Good Quality",
    "screening_allowed": true
  }
}
```

#### Response (200 OK)
```json
{
  "answer": "**Current Screening Result:**\n- **Detected Severity:** Moderate DR (Grade 2)\n- **Model Confidence:** 82.0%\n- **Risk Category:** Moderate Risk\n- **Referral Priority:** Recommended (Eye examination recommended)\n\n**Recommended Action According to AI DR Screen Rules:**\nComprehensive ophthalmic examination within 3 to 6 months to evaluate progression and macular status.\n\n*(Note: AI screening results are not a definitive medical diagnosis. Clinical confirmation by an eye specialist is always recommended.)*",
  "intent": "risk_referral",
  "safety_flag": false,
  "sources": [
    "knowledge/referral/referral_engine.md"
  ]
}
```

---

### 2.2 Health Check
**`GET /api/chat/health`**

#### Response (200 OK)
```json
{
  "indexed_chunks": 143,
  "service": "AI DR Screen Assistant",
  "status": "healthy",
  "supported_languages": ["en", "ta", "hi"],
  "vector_store_ready": true
}
```

---

### 2.3 Quick Actions
**`GET /api/chat/quick-actions`**

#### Response (200 OK)
Returns the array of 10 standard quick action objects containing icon, label, action ID, and default query text.

---

### 2.4 Clear Conversation History
**`POST /api/chat/clear`**

#### Request Payload
```json
{
  "session_id": "sess_user_12345"
}
```

#### Response (200 OK)
```json
{
  "message": "Conversation history cleared.",
  "success": true
}
```
