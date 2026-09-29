"""
Chunking strategies.

Naive RAG (mimics the common default in LangChain/OpenAI cookbook examples):
fixed-size token windows, no overlap. This is known to frequently split
sentences/facts mid-thought, which is one of the reasons naive RAG loses
context precision/recall on multi-hop questions.

Enhanced RAG: sentence-boundary-aware chunks with overlap, so every chunk
contains only complete sentences and a fact split across a chunk boundary in
one chunk is still fully present in the adjacent overlapping chunk.

Token counts here are approximated by whitespace-split word counts, which
avoids a tokenizer dependency. This is a reasonable approximation for English
text (~0.75 words per GPT/Claude token means our "tokens" are slightly
larger than real tokens, but the *relative* comparison between the two
chunking strategies, which is what matters for this benchmark, is unaffected).
"""

from __future__ import annotations

import re
from typing import List

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def fixed_size_chunks(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def sentence_aware_chunks(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Chunk text by accumulating whole sentences up to chunk_size words.

    Unlike fixed_size_chunks, this never splits a sentence mid-way, so each
    chunk is always a grammatically complete unit. Overlap is approximated by
    carrying over trailing sentences from the previous chunk whose total word
    count is <= overlap.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sents: List[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())

        # Single sentence longer than chunk_size: emit as its own chunk.
        if not current_sents and sent_words >= chunk_size:
            chunks.append(sent)
            continue

        # Adding this sentence would overflow — flush current chunk first.
        if current_words + sent_words > chunk_size and current_sents:
            chunks.append(" ".join(current_sents))
            if overlap > 0:
                # Carry over trailing sentences that fit within overlap budget.
                carry: List[str] = []
                carry_words = 0
                for s in reversed(current_sents):
                    sw = len(s.split())
                    if carry_words + sw <= overlap:
                        carry.insert(0, s)
                        carry_words += sw
                    else:
                        break
                current_sents = carry
                current_words = carry_words
            else:
                current_sents = []
                current_words = 0

        current_sents.append(sent)
        current_words += sent_words

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks


def chunk_corpus(
    documents: List[dict],
    chunk_size: int,
    overlap: int = 0,
    method: str = "fixed",
) -> List[dict]:
    """
    documents: list of {"id": str, "text": str}
    method: "fixed" for word-window chunks (naive baseline),
            "sentence" for sentence-boundary-aware chunks (enhanced pipeline)
    returns: list of {"chunk_id": str, "doc_id": str, "text": str}
    """
    out = []
    for doc in documents:
        if method == "sentence":
            pieces = sentence_aware_chunks(doc["text"], chunk_size, overlap)
        else:
            pieces = fixed_size_chunks(doc["text"], chunk_size, overlap)
        for i, piece in enumerate(pieces):
            out.append(
                {
                    "chunk_id": f"{doc['id']}::chunk{i}",
                    "doc_id": doc["id"],
                    "text": piece,
                }
            )
    return out
