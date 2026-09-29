"""
Evaluation metrics.

context_precision / context_recall are computed EXACTLY (no LLM needed):
each QA example has a set of gold supporting document ids. We check, for
each pipeline's retrieved chunks, which document they came from (chunk ids
are "{doc_id}::chunk{n}"), and compute:

    precision = |retrieved chunks whose doc_id is gold| / |retrieved chunks|
    recall    = |gold docs covered by >=1 retrieved chunk| / |gold docs|

These are computed identically for both pipelines regardless of chunk size,
so they're directly comparable -- this is the core, dependency-free signal
in this benchmark and is meaningful even in local/demo mode.

mean_reciprocal_rank is also computed exactly:

    MRR = 1 / rank_of_first_gold_chunk

where rank_of_first_gold_chunk is the 1-indexed position of the first
retrieved chunk whose parent document is a gold document. MRR = 0 when no
gold chunk is retrieved at all. This rewards pipelines that surface a gold
chunk near the top of their ranked list, not just anywhere in the top-k.

faithfulness_proxy / answer_relevancy_proxy are LEXICAL-OVERLAP proxies for
RAGAS's LLM-judged Faithfulness and Answer Relevancy metrics:

    faithfulness_proxy: fraction of the answer's content words that also
        appear somewhere in the retrieved context (measures whether the
        answer is "grounded" in what was retrieved)
    answer_relevancy_proxy: fraction of the question's content words that
        also appear in the answer (measures whether the answer engages
        with what was asked)

These proxies are useful for validating that the pipeline plumbing works and
produces sane, differently-scored outputs -- but they are NOT a substitute
for RAGAS's actual LLM-judged Faithfulness/Answer Relevancy in the real
benchmark. See evaluation/ragas_adapter.py for the real version.
"""

from __future__ import annotations

import math
import re
from typing import List, Set

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "does", "do", "did", "has", "have",
    "had", "be", "been", "that", "this", "it", "its", "as", "by", "with",
    "from", "than", "also",
}


def _content_words(text: str) -> Set[str]:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def chunk_id_to_doc_id(chunk_id: str) -> str:
    return chunk_id.split("::chunk")[0]


def context_precision(retrieved_chunk_ids: List[str], gold_doc_ids: Set[str]) -> float:
    if not retrieved_chunk_ids:
        return 0.0
    hits = sum(
        1 for cid in retrieved_chunk_ids if chunk_id_to_doc_id(cid) in gold_doc_ids
    )
    return hits / len(retrieved_chunk_ids)


def context_recall(retrieved_chunk_ids: List[str], gold_doc_ids: Set[str]) -> float:
    if not gold_doc_ids:
        return 1.0
    retrieved_docs = {chunk_id_to_doc_id(cid) for cid in retrieved_chunk_ids}
    covered = retrieved_docs & gold_doc_ids
    return len(covered) / len(gold_doc_ids)


def mean_reciprocal_rank(retrieved_chunk_ids: List[str], gold_doc_ids: Set[str]) -> float:
    """1 / rank of the first retrieved chunk whose parent doc is a gold doc. 0 if none."""
    for rank, cid in enumerate(retrieved_chunk_ids, 1):
        if chunk_id_to_doc_id(cid) in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_chunk_ids: List[str], gold_doc_ids: Set[str], k: int = 10) -> float:
    """
    nDCG@k with binary relevance, computed at document level.

    Matches the metric used by the BEIR benchmark and MTEB retrieval leaderboard,
    making our scores directly comparable to published numbers from OpenAI,
    Cohere, Microsoft, etc.

    Multiple chunks from the same document are deduplicated (only the
    highest-ranked occurrence counts), since the gold labels are at doc level.
    nDCG@k = DCG@k / IDCG@k, where relevance is binary (1 if gold doc, 0 otherwise).
    """
    if not gold_doc_ids:
        return 1.0

    # Build ranked unique-doc list from retrieved chunks.
    seen: set = set()
    ranked_docs: List[str] = []
    for cid in retrieved_chunk_ids:
        doc_id = chunk_id_to_doc_id(cid)
        if doc_id not in seen:
            seen.add(doc_id)
            ranked_docs.append(doc_id)

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_docs[:k], start=1)
        if doc_id in gold_doc_ids
    )

    # Ideal DCG: gold docs ranked first.
    n_relevant = min(len(gold_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_relevant + 1))

    return dcg / idcg if idcg > 0 else 0.0


def faithfulness_proxy(answer: str, context_chunks: List[str]) -> float:
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    context_words: Set[str] = set()
    for c in context_chunks:
        context_words |= _content_words(c)
    grounded = answer_words & context_words
    return len(grounded) / len(answer_words)


def answer_relevancy_proxy(question: str, answer: str) -> float:
    q_words = _content_words(question)
    if not q_words:
        return 0.0
    a_words = _content_words(answer)
    overlap = q_words & a_words
    return len(overlap) / len(q_words)


def evaluate_example(
    question: str,
    gold_doc_ids: Set[str],
    retrieved_chunk_ids: List[str],
    retrieved_texts: List[str],
    answer: str,
) -> dict:
    return {
        "ndcg_at_10": ndcg_at_k(retrieved_chunk_ids, gold_doc_ids, k=10),
        "context_precision": context_precision(retrieved_chunk_ids, gold_doc_ids),
        "context_recall": context_recall(retrieved_chunk_ids, gold_doc_ids),
        "mean_reciprocal_rank": mean_reciprocal_rank(retrieved_chunk_ids, gold_doc_ids),
        "faithfulness_proxy": faithfulness_proxy(answer, retrieved_texts),
        "answer_relevancy_proxy": answer_relevancy_proxy(question, answer),
    }
