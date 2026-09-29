"""
Convert HotpotQA (distractor setting) into this project's corpus/QA format.

HotpotQA is FREE (Apache 2.0 / CC BY-SA, via the `datasets` library on
Hugging Face) and is the standard multi-hop QA benchmark -- each question
requires combining facts from exactly 2 of ~10 candidate paragraphs
("distractors" make up the rest), which is precisely the regime where
hybrid retrieval + HyDE + reranking are expected to help over single-pass
dense retrieval.

This script does NOT run in the sandbox (no huggingface.co access there) --
run it on your own machine, where `datasets` will download HotpotQA once
(free) and cache it locally.

Usage:
    pip install datasets
    python data/load_hotpotqa.py --n 150 --out-dir data/hotpotqa

Produces:
    <out-dir>/corpus.json   -- one entry per unique paragraph (id=title, text=...)
    <out-dir>/qa.json       -- one entry per question, with gold_doc_ids = the
                               two supporting-fact titles, and reference_answer

Then run:
    python run_benchmark.py --data-dir data/hotpotqa --n 150
"""

from __future__ import annotations

import argparse
import json
import os
import re


def _sanitize_id(title: str) -> str:
    """Doc ids must not contain '::chunk' (used as a chunk_id separator) or
    whitespace (kept simple for chunk_id readability)."""
    safe = re.sub(r"\s+", "_", title.strip())
    safe = safe.replace("::chunk", "_chunk_")
    return safe or "untitled"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150, help="number of questions")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--config", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument(
        "--out-dir", default=os.path.join(os.path.dirname(__file__), "hotpotqa")
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="shuffle seed before taking --n examples"
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading hotpot_qa/{args.config} split={args.split} (downloads once, cached after)...")
    ds = load_dataset("hotpotqa/hotpot_qa", args.config, split=args.split)
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))

    corpus: dict[str, str] = {}  # doc_id -> text
    questions = []
    skipped = 0

    for ex in ds:
        titles = ex["context"]["title"]
        sentences_per_title = ex["context"]["sentences"]

        # Register every candidate paragraph (gold + distractors) in the corpus.
        for title, sentences in zip(titles, sentences_per_title):
            doc_id = _sanitize_id(title)
            text = " ".join(s.strip() for s in sentences if s.strip())
            if doc_id not in corpus and text:
                corpus[doc_id] = text

        gold_titles = set(ex["supporting_facts"]["title"])
        gold_doc_ids = sorted({_sanitize_id(t) for t in gold_titles if t in titles})

        if len(gold_doc_ids) < 2:
            # A handful of HotpotQA examples have only 1 distinct supporting
            # document after dedup; skip these for a clean 2-hop benchmark.
            skipped += 1
            continue

        questions.append(
            {
                "id": ex["id"],
                "question": ex["question"],
                "gold_doc_ids": gold_doc_ids,
                "reference_answer": ex["answer"],
            }
        )

    os.makedirs(args.out_dir, exist_ok=True)
    corpus_list = [{"id": doc_id, "text": text} for doc_id, text in corpus.items()]

    with open(os.path.join(args.out_dir, "corpus.json"), "w") as f:
        json.dump(corpus_list, f, indent=2)
    with open(os.path.join(args.out_dir, "qa.json"), "w") as f:
        json.dump(questions, f, indent=2)

    print(
        f"Wrote {len(corpus_list)} documents and {len(questions)} questions "
        f"to {args.out_dir}/ ({skipped} examples skipped: <2 distinct gold docs)."
    )
    print(
        "Next: python run_benchmark.py --data-dir "
        f"{args.out_dir} --n {len(questions)}"
    )


if __name__ == "__main__":
    main()
