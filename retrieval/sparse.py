"""Sparse (lexical) retrieval via BM25."""

from __future__ import annotations

import re
from typing import List

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


class BM25Retriever:
    def __init__(self, chunk_texts: List[str]):
        self.chunk_texts = chunk_texts
        self._tokenized = [_tokenize(t) for t in chunk_texts]
        self.bm25 = BM25Okapi(self._tokenized)

    def retrieve(self, query: str, top_k: int) -> List[int]:
        """Return indices into chunk_texts, best-first."""
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return order[:top_k]
