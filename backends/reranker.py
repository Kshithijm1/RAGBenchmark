"""
Reranker backends.

Given a query and a list of candidate chunks (already retrieved by the
fusion/dense stage), return the indices of those chunks reordered by
relevance, most relevant first.

- LexicalOverlapReranker: TF-IDF cosine similarity between query and each
  candidate. Dependency-free, used for the in-sandbox demo. This is a real
  reranking signal (it genuinely reorders candidates), just much weaker than
  a trained cross-encoder.

- CrossEncoderReranker: a trained cross-encoder (e.g.
  cross-encoder/ms-marco-MiniLM-L-6-v2) that jointly encodes (query, chunk)
  pairs and outputs a relevance score. This is what the published "enhanced
  RAG" pipeline should use. Requires `sentence-transformers` + HF hub access.

Select via config.reranker_backend in {"lexical", "cross_encoder"}.
"""

from __future__ import annotations

import abc
from typing import List

import numpy as np


class RerankerBackend(abc.ABC):
    @abc.abstractmethod
    def rerank(self, query: str, candidates: List[str], top_k: int) -> List[int]:
        """Return indices into `candidates`, best-first, length <= top_k."""


class LexicalOverlapReranker(RerankerBackend):
    def rerank(self, query: str, candidates: List[str], top_k: int) -> List[int]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if not candidates:
            return []

        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            matrix = vectorizer.fit_transform([query] + candidates)
        except ValueError:
            # Degenerate corpus (e.g. all-stopword text); fall back to
            # original order.
            return list(range(min(top_k, len(candidates))))

        query_vec = matrix[0:1]
        cand_vecs = matrix[1:]
        sims = cosine_similarity(query_vec, cand_vecs)[0]
        order = np.argsort(-sims)
        return order[:top_k].tolist()


class CrossEncoderReranker(RerankerBackend):
    """
    FREE, open-weight cross-encoder reranker via sentence-transformers.

    Default is `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, runs on CPU,
    no GPU required, no API key, no per-call cost). Requires
    `sentence-transformers` (pip, free) + internet access to
    huggingface.co the first time, to download model weights (cached
    locally afterward).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _model_lazy(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: List[str], top_k: int) -> List[int]:
        if not candidates:
            return []
        model = self._model_lazy()
        pairs = [(query, c) for c in candidates]
        scores = model.predict(pairs)
        order = np.argsort(-np.asarray(scores))
        return order[:top_k].tolist()


def build_reranker_backend(name: str, **kwargs) -> RerankerBackend:
    if name == "lexical":
        return LexicalOverlapReranker(**kwargs)
    if name == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    raise ValueError(f"Unknown reranker backend: {name}")
