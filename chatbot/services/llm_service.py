"""
Provider-Agnostic LLM Client and Fallback Engine
Supports Google Gemini, OpenAI, Anthropic, OpenRouter, and a high-fidelity local RAG fallback.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error

from chatbot.config import (
    LLM_PROVIDER,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SYSTEM_PROMPT_PATH
)

class LLMService:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.api_key = LLM_API_KEY
        self.model = LLM_MODEL
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        if SYSTEM_PROMPT_PATH.exists():
            return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        return "You are the AI DR Screen Assistant for rural diabetic retinopathy triage."

    def generate_response(
        self,
        user_message: str,
        retrieved_context: str,
        screening_context_str: str,
        conversation_history: List[Dict[str, str]],
        language: str = "en"
    ) -> str:
        """
        Synthesizes prompt with retrieved knowledge, current clinical context,
        and conversation history, calling the configured LLM or local fallback.
        """
        # If API key is present and provider is configured, attempt LLM call
        if self.api_key:
            try:
                if self.provider == "gemini":
                    res = self._call_gemini(user_message, retrieved_context, screening_context_str, conversation_history, language)
                    if res:
                        return res
                elif self.provider in ["openai", "openrouter"]:
                    res = self._call_openai(user_message, retrieved_context, screening_context_str, conversation_history, language)
                    if res:
                        return res
            except Exception as e:
                print(f"Warning: LLM API call failed ({e}), switching to deterministic knowledge synthesis.")

        # Deterministic RAG Fallback
        return self._deterministic_fallback(user_message, retrieved_context, screening_context_str, language)

    def _call_gemini(
        self,
        user_message: str,
        retrieved_context: str,
        screening_context_str: str,
        conversation_history: List[Dict[str, str]],
        language: str
    ) -> Optional[str]:
        """
        Invokes Google Gemini REST API using standard urllib.
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        contents = []
        # System instructions included as context
        system_instruction = (
            f"{self.system_prompt}\n\n"
            f"TARGET LANGUAGE: {language.upper()} (Respond in this language).\n\n"
            f"ACTIVE SCREENING CONTEXT:\n{screening_context_str}\n\n"
            f"VERIFIED KNOWLEDGE BASE:\n{retrieved_context}\n"
        )

        # Add history
        for turn in conversation_history[-4:]:
            role = "user" if turn["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": turn["content"]}]})

        # Add current user prompt with instructions
        user_prompt_full = f"{user_message}\n\n[Instructions: Answer simply and clearly for rural healthcare users using the verified knowledge above. Do not hallucinate outside the facts.]"
        contents.append({"role": "user", "parts": [{"text": user_prompt_full}]})

        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "temperature": LLM_TEMPERATURE,
                "maxOutputTokens": LLM_MAX_TOKENS
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()

        return None

    def _call_openai(
        self,
        user_message: str,
        retrieved_context: str,
        screening_context_str: str,
        conversation_history: List[Dict[str, str]],
        language: str
    ) -> Optional[str]:
        """
        Invokes OpenAI or OpenRouter REST API using standard urllib.
        """
        endpoint = "https://api.openai.com/v1/chat/completions"
        if self.provider == "openrouter":
            endpoint = "https://openrouter.ai/api/v1/chat/completions"

        messages = [
            {"role": "system", "content": f"{self.system_prompt}\n\nLanguage: {language}\nActive Context:\n{screening_context_str}\nKnowledge:\n{retrieved_context}"}
        ]

        for turn in conversation_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            choices = res_data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()

        return None

    def _deterministic_fallback(
        self,
        user_message: str,
        retrieved_context: str,
        screening_context_str: str,
        language: str
    ) -> str:
        """
        Deterministic, local knowledge synthesizer used when external LLM API is unavailable.
        Extracts verified answers directly from retrieved markdown chunks.
        """
        # If active screening context is specifically queried
        msg_lower = user_message.lower()
        if ("result" in msg_lower or "grade" in msg_lower or "moderate risk" in msg_lower or "high risk" in msg_lower) and "Current Screening Result:" in screening_context_str:
            return screening_context_str

        if not retrieved_context or "No specific knowledge base documents matched" in retrieved_context:
            if language == "ta":
                return "மன்னிக்கவும், AI DR Screen வழிகாட்டுதல் ஆவணங்களில் இதற்கான விவரங்கள் கிடைக்கவில்லை. தயவுசெய்து உங்கள் சுகாதார ஒருங்கிணைப்பாளரை அணுகவும்."
            elif language == "hi":
                return "क्षमा करें, AI DR Screen ज्ञानकोष में इसके लिए पर्याप्त जानकारी उपलब्ध नहीं है। कृपया स्वास्थ्य समन्वयक से संपर्क करें।"
            return "I don't have enough information in the current AI DR Screen knowledge base to answer that accurately. Please consult your local Primary Health Center team."

        # Extract the highest scoring chunk text
        chunks = retrieved_context.split("--- Document Chunks")
        if len(chunks) > 1:
            primary_chunk = chunks[1].split("\n", 1)[-1].strip()
            # Clean up metadata tags
            cleaned = re.sub(r'\[.*? - .*?\]', '', primary_chunk).strip()
            return cleaned

        return retrieved_context.strip()

_llm_service = None

def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
