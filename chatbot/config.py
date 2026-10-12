"""
Configuration settings for AI DR Screen Chatbot Assistant.
Loads environment variables with sensible defaults for local and production deployment.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CHATBOT_DIR = BASE_DIR / "chatbot"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
VECTOR_DB_DIR = CHATBOT_DIR / "data" / "vector_store"
HEALTHCARE_DATA_PATH = CHATBOT_DIR / "data" / "healthcare_centers.json"
SYSTEM_PROMPT_PATH = CHATBOT_DIR / "prompts" / "system_prompt.txt"

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # gemini, openai, anthropic, fallback
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", "")))
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))

# Embedding and RAG
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "tfidf-cosine")  # tfidf-cosine, all-MiniLM-L6-v2, etc.
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))

# Session and Safety
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")  # en, ta, hi
