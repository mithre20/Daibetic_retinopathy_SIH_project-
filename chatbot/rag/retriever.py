"""
Semantic Knowledge Retriever
Retrieves top relevant knowledge chunks and prepares formatted context blocks for the LLM.
"""

from typing import List, Dict, Any, Tuple
from chatbot.rag.vector_store import get_vector_store
from chatbot.config import TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD

class Retriever:
    def __init__(self, top_k: int = TOP_K_RETRIEVAL, threshold: float = SIMILARITY_THRESHOLD):
        self.top_k = top_k
        self.threshold = threshold
        self.vector_store = get_vector_store()

    def retrieve(self, query: str, top_k: int = None, threshold: float = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        th = threshold if threshold is not None else self.threshold

        ranked_results = self.vector_store.search(query, top_k=k, threshold=th)

        retrieved_chunks = []
        for chunk, score in ranked_results:
            item = dict(chunk)
            item["score"] = score
            retrieved_chunks.append(item)

        return retrieved_chunks

    def format_context_for_prompt(self, chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Formats retrieved chunks into a clean prompt context block and collects source file references.
        """
        if not chunks:
            return ("No specific knowledge base documents matched this query.", [])

        formatted_blocks = []
        sources = []

        for idx, c in enumerate(chunks, 1):
            src = c.get("source", "knowledge_base")
            if src not in sources:
                sources.append(src)
            formatted_blocks.append(f"--- Document Chunks [{idx}] (Source: {src}) ---\n{c['text']}")

        context_str = "\n\n".join(formatted_blocks)
        return context_str, sources

_retriever = None

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever

def retrieve_relevant_knowledge(query: str, top_k: int = None) -> Tuple[str, List[str]]:
    retriever = get_retriever()
    chunks = retriever.retrieve(query, top_k=top_k)
    return retriever.format_context_for_prompt(chunks)
