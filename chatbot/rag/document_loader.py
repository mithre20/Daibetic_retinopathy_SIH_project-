"""
Document Loader for AI DR Screen Knowledge Base
Reads all markdown files recursively from the knowledge directory.
"""

from pathlib import Path
from typing import List, Dict, Any
from chatbot.config import KNOWLEDGE_DIR

def load_knowledge_documents(knowledge_dir: Path = KNOWLEDGE_DIR) -> List[Dict[str, Any]]:
    """
    Recursively scans the knowledge directory and loads all .md documents.
    Returns a list of dicts with 'file_path', 'relative_path', 'category', 'content'.
    """
    documents = []
    if not knowledge_dir.exists():
        return documents

    for md_file in knowledge_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            relative_path = md_file.relative_to(knowledge_dir)
            category = relative_path.parts[0] if len(relative_path.parts) > 1 else "general"

            documents.append({
                "file_path": str(md_file),
                "relative_path": str(relative_path).replace("\\", "/"),
                "category": category,
                "title": md_file.stem.replace("_", " ").title(),
                "content": content
            })
        except Exception as e:
            print(f"Warning: Failed to read knowledge file {md_file}: {e}")

    return documents
