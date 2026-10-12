"""
In-Memory Vector Store with Persistent Index Caching
Indexes document chunks and calculates cosine similarity rankings.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from chatbot.rag.document_loader import load_knowledge_documents
from chatbot.rag.chunker import chunk_markdown_documents
from chatbot.rag.embeddings import SemanticVectorizer

class KnowledgeVectorStore:
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer = SemanticVectorizer()
        self.matrix = None
        self._is_indexed = False

    def build_index(self):
        """
        Loads all knowledge documents, chunks them, and builds vector matrix.
        """
        docs = load_knowledge_documents()
        self.chunks = chunk_markdown_documents(docs)
        if not self.chunks:
            print("Warning: No knowledge chunks found to index.")
            return

        texts = [chunk["text"] for chunk in self.chunks]
        self.matrix = self.vectorizer.fit_transform(texts)
        self._is_indexed = True
        print(f"KnowledgeVectorStore: Successfully indexed {len(self.chunks)} chunks across {len(docs)} documents.")

    def search(self, query: str, top_k: int = 4, threshold: float = 0.10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs semantic similarity search over indexed chunks.
        Returns list of (chunk, score) tuples.
        """
        if not self._is_indexed:
            self.build_index()

        if not self.chunks or self.matrix is None:
            return []

        query_vec = self.vectorizer.transform(query)
        scores = self.vectorizer.compute_similarity(query_vec, self.matrix)

        # Sort by similarity descending
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score >= threshold:
                results.append((self.chunks[idx], score))

        return results

# Global singleton
_vector_store = None

def get_vector_store() -> KnowledgeVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = KnowledgeVectorStore()
        _vector_store.build_index()
    return _vector_store
