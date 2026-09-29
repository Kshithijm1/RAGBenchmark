"""Dense retrieval and Reciprocal Rank Fusion (RRF) for hybrid retrieval."""

from __future__ import annotations

from typing import List

import numpy as np

from backends.embeddings import EmbeddingBackend


class DenseRetriever:
    def __init__(self, embedding_backend: EmbeddingBackend, chunk_texts: List[str]):
        self.backend = embedding_backend
        self.chunk_texts = chunk_texts
        self.backend.fit(chunk_texts)
        self.chunk_vecs = self.backend.embed(chunk_texts)

    def retrieve(self, query: str, top_k: int) -> List[int]:
        """Return indices into chunk_texts, best-first, by cosine similarity."""
        q_vec = self.backend.embed([query])[0]
        sims = self.chunk_vecs @ q_vec
        order = np.argsort(-sims)
        return order[:top_k].tolist()


def reciprocal_rank_fusion(rank_lists: List[List[int]], k: int = 60) -> List[int]:
    """
    Combine multiple ranked lists of item indices into a single fused ranking.

    score(item) = sum over lists containing item of  1 / (k + rank_in_list)

    rank_in_list is 1-indexed. Items not present in a list contribute 0 for
    that list. Returns all items that appear in at least one list, sorted by
    fused score descending.
    """
    scores: dict[int, float] = {}
    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda item: -scores[item])
