# AI DR Screen Chatbot Assistant — Architecture Documentation

## 1. System Overview
The AI DR Screen Assistant (code-named XAIRIS) is a modular, context-aware conversational AI assistant integrated into the AI DR Screen clinical decision-support application. It assists community health workers (ASHAs), optometrists, technicians, and patients in understanding the screening workflow, interpreting AI classifications (EfficientNet-B0), visual explanations (Grad-CAM, SHAP, U-Net), laboratory report extractions, and navigating clinical referral pathways.

## 2. Architectural Blueprint

```
USER
  ↓
FLOATING CHAT UI (chatbot.js + chatbot.css)
  ↓ [POST /api/chat with session_id, current_page, screening_context]
CHATBOT ROUTE (chatbot/api/chat_routes.py)
  ↓
CHATBOT SERVICE (chatbot/services/chatbot_service.py)
  ↓
┌────────────────────────────────────────────────────────┐
│ 1. Emergency Safety Pre-Check (safety/emergency.py)   │
│    - Detects sudden vision loss, curtain, trauma       │
│    - Triggers immediate urgent ophthalmic escalation   │
├────────────────────────────────────────────────────────┤
│ 2. Medical Safety & Adversarial Guard (safety.py)      │
│    - Blocks drug prescription / dosage instructions    │
│    - Blocks stopping insulin / tablets                 │
│    - Blocks diagnosis demands                          │
│    - Defends against prompt injections / data leaks    │
├────────────────────────────────────────────────────────┤
│ 3. Intent Classifier & Query Router                   │
│    - 16 Primary intents (IQA, DR grades, RAG, etc.)    │
├────────────────────────────────────────────────────────┤
│ 4. Context Ingestion (context/screening_context.py)    │
│    - Active DR grade, confidence, IQA status           │
│    - Deterministic referral decision interpretation    │
├────────────────────────────────────────────────────────┤
│ 5. Semantic Knowledge Retrieval (RAG Pipeline)         │
│    - Semantic vector store & Markdown chunk retrieval  │
├────────────────────────────────────────────────────────┤
│ 6. Verified Healthcare Registry (healthcare_service)   │
│    - Real configured centers (PHCs, Mobile Vans, DGH)  │
│    - Strict anti-hallucination policy                  │
├────────────────────────────────────────────────────────┤
│ 7. LLM Response Generator (services/llm_service.py)   │
│    - Gemini / OpenAI / Anthropic or Local RAG Fallback │
├────────────────────────────────────────────────────────┤
│ 8. Response Safety Post-Validation                    │
│    - Guarantees screening rejections != "healthy eye"  │
└────────────────────────────────────────────────────────┘
  ↓
STRUCTURED JSON RESPONSE (answer, intent, sources, safety_flag)
```

## 3. Key Design Principles
1. **Zero Model Disruption**: Existing deep learning inference pipelines (`/api/predict`, `/api/screen`, `/api/referral`, `/api/gradcam`, `/api/shap`) are unchanged.
2. **Deterministic Precedence**: Critical medical safety and emergency escalation rules always take precedence over the generative LLM.
3. **Contextual Continuity**: The assistant retains turn history per session (`context_manager.py`) and seamlessly reads live screening state from `localStorage` (`aidr_screening_context`).
4. **Strict Anti-Hallucination**: Facility contact information is restricted strictly to verified configured records in `chatbot/data/healthcare_centers.json`.
