"""
Embedding backends.

EmbeddingBackend is the interface every dense retriever depends on.

- TfidfSvdEmbeddings: pure scikit-learn, no network/model download required.
  Used for the in-sandbox demo run. It is a *real* dense embedding (LSA-style
  latent semantic vectors), just much weaker than a modern transformer
  embedding model -- which is the honest point: the demo numbers are not the
  benchmark numbers, they only prove the pipeline plumbing is correct.

- OpenAIEmbeddings: text-embedding-3-small/large via the OpenAI API. This is
  what the published "naive RAG (OpenAI baseline)" pipeline should use.

- SentenceTransformerEmbeddings: any sentence-transformers model (e.g.
  bge-large-en-v1.5, e5-large-v2). Requires `sentence-transformers` and
  internet access to huggingface.co to download weights.

Select via config.embedding_backend in {"tfidf", "openai", "sentence_transformer"}.
"""

from __future__ import annotations

import abc
from typing import List

import numpy as np


class EmbeddingBackend(abc.ABC):
    @abc.abstractmethod
    def fit(self, corpus: List[str]) -> "EmbeddingBackend":
        """Fit on the corpus (only meaningful for the TF-IDF/SVD backend)."""

    @abc.abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) array of L2-normalized vectors."""


class TfidfSvdEmbeddings(EmbeddingBackend):
    """Local, dependency-free dense embedding via TF-IDF + Truncated SVD (LSA)."""

    def __init__(self, n_components: int = 128, random_state: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.n_components = n_components
        self.random_state = random_state
        self.svd = None

    def fit(self, corpus: List[str]) -> "TfidfSvdEmbeddings":
        from sklearn.decomposition import TruncatedSVD

        tfidf = self.vectorizer.fit_transform(corpus)
        n_comp = min(self.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_comp = max(n_comp, 2)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self.svd.fit(tfidf)
        return self

    def embed(self, texts: List[str]) -> np.ndarray:
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class OpenAIEmbeddings(EmbeddingBackend):
    """text-embedding-3-small/large. Requires `openai` package + OPENAI_API_KEY."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._client = None

    def fit(self, corpus: List[str]) -> "OpenAIEmbeddings":
        return self  # stateless

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def embed(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        import time
        client = self._client_lazy()
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            while True:
                try:
                    resp = client.embeddings.create(model=self.model, input=batch)
                    all_vecs.extend([d.embedding for d in resp.data])
                    break
                except Exception as e:
                    from openai import RateLimitError
                    if isinstance(e, RateLimitError):
                        print(f"  Embedding rate limit, waiting 60s... (batch {i//batch_size + 1})", flush=True)
                        time.sleep(60)
                    else:
                        raise
            if i + batch_size < len(texts):
                time.sleep(1)  # avoid bursting TPM limit
        vecs = np.array(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class SentenceTransformerEmbeddings(EmbeddingBackend):
    """
    FREE, open-weight embedding models via sentence-transformers.

    Default is `BAAI/bge-small-en-v1.5` (~130MB, runs comfortably on CPU,
    no GPU required) -- a strong, widely-benchmarked open embedding model
    and a realistic stand-in for "the embedding model a naive RAG tutorial
    tells you to use." Larger options (`BAAI/bge-base-en-v1.5`,
    `BAAI/bge-large-en-v1.5`, `intfloat/e5-large-v2`) trade speed for a
    small quality bump if you have the compute.

    Requires `sentence-transformers` (pip, free) + internet access to
    huggingface.co the first time, to download model weights (one-time,
    cached locally afterward -- no per-call cost).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def fit(self, corpus: List[str]) -> "SentenceTransformerEmbeddings":
        return self  # stateless

    def _model_lazy(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        model = self._model_lazy()
        vecs = model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def build_embedding_backend(name: str, **kwargs) -> EmbeddingBackend:
    if name == "tfidf":
        return TfidfSvdEmbeddings(**kwargs)
    if name == "openai":
        return OpenAIEmbeddings(**kwargs)
    if name == "sentence_transformer":
        return SentenceTransformerEmbeddings(**kwargs)
    raise ValueError(f"Unknown embedding backend: {name}")
