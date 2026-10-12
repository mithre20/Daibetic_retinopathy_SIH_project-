"""
Session Memory and Context Manager
Manages multi-turn conversation history with session isolation,
LRU turn pruning, and session scoping to prevent memory leaks across users.
"""

from typing import Dict, List, Any, Optional
import time
from chatbot.config import MAX_HISTORY_TURNS

class ContextManager:
    def __init__(self, max_history_turns: int = MAX_HISTORY_TURNS):
        self.max_history_turns = max_history_turns
        # session_id -> list of {"role": "user"|"assistant", "content": str, "timestamp": float}
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        # session_id -> last_activity timestamp
        self.session_timestamps: Dict[str, float] = {}

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves recent turns for the specified session.
        """
        if not session_id:
            return []
        return list(self.sessions.get(session_id, []))

    def add_turn(self, session_id: str, role: str, content: str):
        """
        Records a conversational turn and enforces turn limit.
        """
        if not session_id or not content:
            return

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        self.session_timestamps[session_id] = time.time()

        # Enforce history limit
        max_messages = self.max_history_turns * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]

    def clear_session(self, session_id: str):
        """
        Clears conversation history for the given session.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.session_timestamps:
            del self.session_timestamps[session_id]

    def cleanup_idle_sessions(self, max_idle_seconds: int = 3600):
        """
        Removes expired sessions older than max_idle_seconds.
        """
        now = time.time()
        expired = [
            sid for sid, last_active in self.session_timestamps.items()
            if (now - last_active) > max_idle_seconds
        ]
        for sid in expired:
            self.clear_session(sid)

# Global singleton
_context_manager = None

def get_context_manager() -> ContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager
