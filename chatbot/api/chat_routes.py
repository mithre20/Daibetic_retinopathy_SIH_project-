"""
Flask Blueprint for Chatbot API Endpoints
Provides POST /api/chat, GET /api/chat/health, POST /api/chat/clear,
GET /api/chat/quick-actions, and GET /api/chat/history/<session_id>.
"""

import logging
from flask import Blueprint, request, jsonify
from chatbot.services.chatbot_service import get_chatbot_service
from chatbot.context.context_manager import get_context_manager
from chatbot.rag.vector_store import get_vector_store
from chatbot.data.mongo_db import get_mongo_store

logger = logging.getLogger("aidr_chatbot")

chat_bp = Blueprint("chatbot_api", __name__, url_prefix="/api/chat")

QUICK_ACTIONS = [
    {"id": "start_screening", "icon": "👁", "label": "Start Screening", "query": "How do I start diabetic retinopathy screening?"},
    {"id": "upload_image", "icon": "📷", "label": "Upload Eye Image", "query": "How do I upload a retinal fundus scan?"},
    {"id": "upload_report", "icon": "📄", "label": "Upload Report", "query": "What medical reports can I upload for OCR analysis?"},
    {"id": "understand_result", "icon": "📊", "label": "Understand Result", "query": "What does my current screening result and DR grade mean?"},
    {"id": "explain_ai", "icon": "🔍", "label": "Explain AI Result", "query": "Explain how Grad-CAM, SHAP, and U-Net work together."},
    {"id": "find_healthcare", "icon": "🏥", "label": "Find Healthcare", "query": "Where is the nearest verified eye hospital or referral center?"},
    {"id": "navigate", "icon": "🧭", "label": "Navigate Website", "query": "How do I navigate between stations and what shortcuts exist?"},
    {"id": "how_it_works", "icon": "❓", "label": "How It Works", "query": "What is AI DR Screen and how does the 5-stage pipeline work?"},
    {"id": "privacy_safety", "icon": "🔐", "label": "Privacy & Safety", "query": "How is patient data protected and what are the clinical safety rules?"},
    {"id": "support", "icon": "📞", "label": "Contact Support", "query": "How do I troubleshoot technical issues with the screening station?"}
]

@chat_bp.route("", methods=["POST"])
@chat_bp.route("/", methods=["POST"])
def handle_chat_message():
    """
    POST /api/chat
    Processes conversational input with session context, page location, screening telemetry,
    and user identity.
    """
    data = request.get_json(silent=True) or {}

    message = data.get("message", "").strip()
    session_id = data.get("session_id")
    user_id = data.get("user_id")
    page = data.get("page", "pipeline")
    context = data.get("context", {})
    language = data.get("language", "en")

    if not message:
        return jsonify({
            "error": "Empty message provided.",
            "answer": "Please enter a question or select one of the quick actions below 😊.",
            "message": "Please enter a question or select one of the quick actions below 😊.",
            "intent": "general_info",
            "confidence": 0.99,
            "actions": [],
            "quick_replies": [a["label"] for a in QUICK_ACTIONS[:4]],
            "sources": [],
            "safety_flag": False
        }), 400

    try:
        service = get_chatbot_service()
        response = service.process_message(
            message=message,
            session_id=session_id,
            page=page,
            context=context,
            language=language,
            user_id=user_id
        )
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Chat processing exception securely logged: {str(e)}", exc_info=True)
        return jsonify({
            "answer": "I'm having trouble connecting right now. Please try again in a moment. You can still use the website's screening features.",
            "message": "I'm having trouble connecting right now. Please try again in a moment. You can still use the website's screening features.",
            "intent": "technical_support",
            "confidence": 0.90,
            "actions": [],
            "quick_replies": ["Start Screening", "Contact Support"],
            "sources": [],
            "safety_flag": False
        }), 500

@chat_bp.route("/health", methods=["GET"])
def chatbot_health():
    """
    GET /api/chat/health
    Returns chatbot component health, RAG vector store stats, MongoDB status, and service status.
    """
    store = get_vector_store()
    mongo_store = get_mongo_store()
    return jsonify({
        "status": "healthy",
        "service": "AI DR Screen Assistant",
        "indexed_chunks": len(store.chunks),
        "vector_store_ready": store._is_indexed,
        "mongodb_connected": mongo_store.is_connected,
        "supported_languages": ["en", "ta", "hi"]
    }), 200

@chat_bp.route("/clear", methods=["POST"])
def clear_session_history():
    """
    POST /api/chat/clear
    Clears memory history for the active session.
    """
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if session_id:
        get_context_manager().clear_session(session_id)
    return jsonify({"success": True, "message": "Conversation history cleared."}), 200

@chat_bp.route("/quick-actions", methods=["GET"])
def get_quick_actions():
    """
    GET /api/chat/quick-actions
    Returns list of 10 standard quick action buttons.
    """
    return jsonify({"quick_actions": QUICK_ACTIONS}), 200

@chat_bp.route("/history/<session_id>", methods=["GET"])
def get_chat_history(session_id: str):
    """
    GET /api/chat/history/<session_id>
    Retrieves stored message history from MongoDB or in-memory fallback.
    """
    mongo_store = get_mongo_store()
    if mongo_store.is_connected:
        messages = mongo_store.get_messages(session_id)
        if messages:
            return jsonify({"session_id": session_id, "messages": messages, "source": "mongodb"}), 200

    # In-memory fallback
    context_mgr = get_context_manager()
    raw_history = context_mgr.get_history(session_id)
    return jsonify({"session_id": session_id, "messages": raw_history, "source": "memory"}), 200
