# AI DR Screen Chatbot Assistant — Setup & Installation Guide

## 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Existing virtual environment (`.venv`) with installed requirements:
  `flask`, `flask-cors`, `torch`, `torchvision`, `scikit-learn`, `numpy`, `pillow`.

## 2. Environment Configuration
Create a `.env` file in the project root (or copy from `.env.example`):
```bash
cp .env.example .env
```

Set your desired configuration variables:
```env
PORT=5000
LLM_PROVIDER=gemini       # Options: gemini, openai, anthropic, openrouter, fallback
LLM_API_KEY=your_key_here # Optional: if omitted, automatic local RAG fallback is used
LLM_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=tfidf-cosine
TOP_K_RETRIEVAL=4
SIMILARITY_THRESHOLD=0.10
MAX_HISTORY_TURNS=6
DEFAULT_LANGUAGE=en       # Options: en, ta, hi
```

## 3. Starting the Backend Server
Activate the virtual environment and start the Flask server:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python backend/app.py
```
The server will start on `http://localhost:5000` (or the configured `PORT`).

## 4. Verifying Chatbot Health
You can verify the chatbot component health via curl or browser:
```bash
curl http://localhost:5000/api/chat/health
```
Expected output:
```json
{
  "indexed_chunks": 143,
  "service": "AI DR Screen Assistant",
  "status": "healthy",
  "supported_languages": ["en", "ta", "hi"],
  "vector_store_ready": true
}
```

## 5. Opening the Frontend
Open `http://localhost:5000` in Google Chrome, Microsoft Edge, or Firefox.
Look for the circular blue-cyan floating chatbot icon in the bottom right corner of the screen.
Click the icon to launch the assistant.
