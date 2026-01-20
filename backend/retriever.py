# backend/retriever.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, documents, embeddings=None):
        """
        documents: list of text chunks or list of dicts with {'page_content', 'metadata'}
        embeddings: np.ndarray of document embeddings (optional)
        """
        self.documents = documents
        self.embeddings = embeddings

        # Extract text safely whether docs are dicts or plain strings
        self.texts = [
            doc["page_content"] if isinstance(doc, dict) and "page_content" in doc else str(doc)
            for doc in documents
        ]

        # Tokenize for BM25
        self.tokenized_docs = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def semantic_search(self, query_embedding, top_k=5):
        """Return top_k docs based on semantic similarity"""
        if self.embeddings is None:
            raise ValueError("Semantic embeddings not provided for semantic search.")
        sims = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.texts[i], float(sims[i])) for i in top_indices]

    def keyword_search(self, query, top_k=5):
        """Return top_k docs based on BM25 keyword matching"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.texts[i], float(scores[i])) for i in top_indices]

    def hybrid_search(self, query, query_embedding, top_k=5, alpha=0.5):
        """
        Combine semantic + keyword scores
        alpha controls weighting:
            1.0 = fully semantic
            0.0 = fully keyword
        """
        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # Semantic scores
        if self.embeddings is not None:
            semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        else:
            semantic_scores = np.zeros(len(self.texts))

        # Normalize scores
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        sem_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)

        # Combine
        hybrid_scores = alpha * sem_norm + (1 - alpha) * bm25_norm
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        return [(self.texts[i], float(hybrid_scores[i])) for i in top_indices]

    def retrieve(self, query, top_k=3):
        """Fallback simple keyword-based retriever"""
        query_tokens = query.lower().split()
        scores = []
        for idx, tokens in enumerate(self.tokenized_docs):
            overlap = len(set(query_tokens) & set(tokens))
            scores.append((idx, overlap))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [self.texts[i] for i, _ in scores[:top_k]]
        return top_docs
