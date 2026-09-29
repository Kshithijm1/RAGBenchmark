"""
Central configuration for the RAG benchmark.

Everything that differs between a "local demo run" (no internet / no API keys,
runs entirely in-sandbox on TF-IDF + lexical fallbacks) and a "real benchmark
run" (OpenAI/Anthropic embeddings + LLMs, HotpotQA/BEIR data, RAGAS metrics)
is controlled from here via environment variables.

Local demo (default):
    python run_benchmark.py

Real benchmark:
    export RAG_EMBEDDING_BACKEND=openai
    export RAG_LLM_BACKEND=anthropic
    export RAG_RERANKER_BACKEND=cross_encoder
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python run_benchmark.py --data hotpotqa --n 150
"""

import os
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # --- Chunking ---
    naive_chunk_size: int = 512        # tokens (approx, whitespace-based)
    naive_chunk_overlap: int = 0
    enhanced_chunk_size: int = 256
    enhanced_chunk_overlap: int = 64

    # --- Retrieval ---
    naive_top_k: int = 3               # what naive RAG hands to the LLM
    enhanced_candidate_k: int = 30     # candidates pulled before reranking
    enhanced_final_k: int = 3          # what enhanced RAG hands to the LLM
    rrf_k: int = 60                    # Reciprocal Rank Fusion constant

    # --- Embedding ---
    embedding_dim: int = 128           # used by the TF-IDF/SVD fallback only

    # --- Backends ---
    embedding_backend: str = field(
        default_factory=lambda: os.environ.get("RAG_EMBEDDING_BACKEND", "tfidf")
    )
    llm_backend: str = field(
        default_factory=lambda: os.environ.get("RAG_LLM_BACKEND", "template")
    )
    reranker_backend: str = field(
        default_factory=lambda: os.environ.get("RAG_RERANKER_BACKEND", "lexical")
    )

    # --- Misc ---
    random_seed: int = 42


CONFIG = PipelineConfig()
