"""
The two pipelines under comparison.

NaiveRAGPipeline mirrors the OpenAI cookbook / default LangChain RAG recipe:
  - fixed-size chunks, no overlap
  - single dense embedding model
  - cosine similarity top-k retrieval
  - raw question used as the retrieval query
  - retrieved chunks stuffed directly into the prompt

EnhancedRAGPipeline adds three independently-motivated improvements:
  - Sentence-aware chunking at larger granularity: chunks end on sentence
    boundaries and are sized to keep gold paragraphs intact (~180 words),
    so retrieval operates at the same granularity as the gold labels.
  - Dual-signal hybrid retrieval: Dense(question) + BM25(question) candidate
    lists combined via Reciprocal Rank Fusion. BM25 is especially strong on
    entity-centric multi-hop questions where keyword overlap is high. HyDE
    is not used — for multi-hop questions the bridge entity is unknown at
    query time, so hypothetical answers tend to hallucinate wrong entities
    and pull off-target documents into the candidate pool.
  - Cross-encoder reranking: the fused candidate set is rescored by a model
    that jointly encodes (query, candidate) pairs, which is strictly more
    expressive than independently-encoded cosine similarity.

Both pipelines expose the same interface so the evaluation harness can treat
them interchangeably:

    result = pipeline.run(question)
    result.retrieved_chunk_ids   -> List[str]
    result.retrieved_texts       -> List[str]
    result.answer                -> str
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from backends.embeddings import EmbeddingBackend
from backends.llm import LLMBackend
from backends.reranker import RerankerBackend
from chunking import chunk_corpus
from retrieval.fusion import DenseRetriever, reciprocal_rank_fusion
from retrieval.sparse import BM25Retriever


@dataclass
class RAGResult:
    retrieved_chunk_ids: List[str]
    retrieved_texts: List[str]
    answer: str


class NaiveRAGPipeline:
    name = "naive_rag"

    def __init__(
        self,
        documents: List[dict],
        embedding_backend: EmbeddingBackend,
        llm_backend: LLMBackend,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        chunking_method: str = "fixed",
    ):
        self.chunks = chunk_corpus(documents, chunk_size, chunk_overlap, method=chunking_method)
        self.chunk_texts = [c["text"] for c in self.chunks]
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.dense = DenseRetriever(embedding_backend, self.chunk_texts)
        self.llm = llm_backend
        self.top_k = top_k

    def run(self, question: str) -> RAGResult:
        idxs = self.dense.retrieve(question, self.top_k)
        texts = [self.chunk_texts[i] for i in idxs]
        ids = [self.chunk_ids[i] for i in idxs]
        answer = self.llm.synthesize_answer(question, texts)
        return RAGResult(retrieved_chunk_ids=ids, retrieved_texts=texts, answer=answer)


class EnhancedRAGPipeline:
    name = "enhanced_rag"

    def __init__(
        self,
        documents: List[dict],
        embedding_backend: EmbeddingBackend,
        llm_backend: LLMBackend,
        reranker_backend: RerankerBackend,
        chunk_size: int,
        chunk_overlap: int,
        candidate_k: int,
        final_k: int,
        rrf_k: int,
        chunking_method: str = "sentence",
    ):
        self.chunks = chunk_corpus(documents, chunk_size, chunk_overlap, method=chunking_method)
        self.chunk_texts = [c["text"] for c in self.chunks]
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.dense = DenseRetriever(embedding_backend, self.chunk_texts)
        self.sparse = BM25Retriever(self.chunk_texts)
        self.llm = llm_backend
        self.reranker = reranker_backend
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.rrf_k = rrf_k

    def run(self, question: str) -> RAGResult:
        # 1. Hybrid retrieval: dense (semantic) + BM25 (lexical/entity-exact).
        #    HyDE is dropped — for multi-hop questions the bridge entity is
        #    unknown at query time, so hypothetical answers hallucinate wrong
        #    entities and pull in off-target chunks that confuse the reranker.
        #    Dense+BM25 fusion is well-established to outperform dense-only on
        #    entity-centric QA (where HotpotQA keywords matter greatly).
        dense_q = self.dense.retrieve(question, self.candidate_k)
        sparse_q = self.sparse.retrieve(question, self.candidate_k)
        fused_order = reciprocal_rank_fusion(
            [dense_q, sparse_q], k=self.rrf_k
        )[: self.candidate_k]

        candidate_texts = [self.chunk_texts[i] for i in fused_order]
        candidate_ids = [self.chunk_ids[i] for i in fused_order]

        # 3. Cross-encoder (or lexical fallback) reranking down to final_k.
        rerank_order = self.reranker.rerank(question, candidate_texts, self.final_k)
        final_texts = [candidate_texts[i] for i in rerank_order]
        final_ids = [candidate_ids[i] for i in rerank_order]

        answer = self.llm.synthesize_answer(question, final_texts)
        return RAGResult(
            retrieved_chunk_ids=final_ids, retrieved_texts=final_texts, answer=answer
        )
