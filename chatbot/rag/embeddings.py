"""
Embedding and Vectorization Module
Provides high-speed, local TF-IDF and dense vector representations for semantic retrieval.
"""

from typing import List, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticVectorizer:
    """
    Combines sublinear TF-IDF word and character n-grams to robustly handle
    clinical terminology, Indian English phrasing, and conversational variations.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b'
        )
        self.is_fitted = False

    def fit_transform(self, texts: List[str]):
        if not texts:
            raise ValueError("No texts provided to fit vectorizer.")
        vectors = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return vectors

    def transform(self, query: str):
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before transforming queries.")
        return self.vectorizer.transform([query])

    def compute_similarity(self, query_vec, doc_matrix) -> np.ndarray:
        return cosine_similarity(query_vec, doc_matrix).flatten()
