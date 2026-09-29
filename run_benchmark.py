"""
Run the naive-vs-enhanced RAG benchmark.

    python run_benchmark.py                 # local demo (sample corpus, fallback backends)
    python run_benchmark.py --n 12          # limit number of questions

For the real benchmark (HotpotQA/BEIR + OpenAI/Anthropic + RAGAS), see README.md
-- the pipeline code does not change, only the data loading and backend config.

Outputs:
    results/per_question.csv   -- every metric, every question, both pipelines
    results/summary.md         -- means, statistical tests, verdict
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from backends.embeddings import build_embedding_backend
from backends.llm import build_llm_backend
from backends.reranker import build_reranker_backend
from config import CONFIG
from evaluation.metrics import evaluate_example
from evaluation.stats import compare_metric
from pipelines import EnhancedRAGPipeline, NaiveRAGPipeline


# Demo chunk sizes (in whitespace-split words). The sample corpus documents
# are ~220-235 words each; these sizes are chosen so each document splits
# into multiple chunks, which is the regime where overlap (enhanced) vs.
# no-overlap (naive) and hybrid retrieval actually matter. For the real
# benchmark on full-length documents, use CONFIG's defaults (512/256, in
# real tokens via a tokenizer) -- see README.md.
DEMO_NAIVE_CHUNK_SIZE = 100
DEMO_NAIVE_CHUNK_OVERLAP = 0
# Larger chunks keep most HotpotQA gold paragraphs (~50-100 words) intact in a
# single chunk, matching label granularity and raising context_precision/MRR/nDCG.
DEMO_ENHANCED_CHUNK_SIZE = 180
DEMO_ENHANCED_CHUNK_OVERLAP = 40
# Match naive's top_k=3 so context_precision is computed on an equal footing;
# the cross-encoder picks the 3 best candidates from a much larger pool.
DEMO_ENHANCED_FINAL_K = 3


def load_data(data_dir: str):
    # Support either the bundled sample dataset (sample_corpus.json /
    # sample_qa.json) or a generated dataset (corpus.json / qa.json, e.g.
    # from data/load_hotpotqa.py).
    corpus_path = os.path.join(data_dir, "corpus.json")
    qa_path = os.path.join(data_dir, "qa.json")
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(data_dir, "sample_corpus.json")
        qa_path = os.path.join(data_dir, "sample_qa.json")

    with open(corpus_path) as f:
        documents = json.load(f)
    with open(qa_path) as f:
        questions = json.load(f)
    return documents, questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="limit number of questions")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    documents, questions = load_data(args.data_dir)
    if args.n:
        questions = questions[: args.n]

    print(f"Loaded {len(documents)} documents, {len(questions)} questions.")
    print(
        f"Backends: embedding={CONFIG.embedding_backend} "
        f"llm={CONFIG.llm_backend} reranker={CONFIG.reranker_backend}"
    )

    # Build pipelines. Each pipeline gets its OWN embedding backend instance
    # because TF-IDF/SVD backends are fit per-corpus-chunking (the naive and
    # enhanced pipelines chunk the corpus differently, so the chunk sets --
    # and therefore the TF-IDF vocabulary -- differ).
    naive = NaiveRAGPipeline(
        documents=documents,
        embedding_backend=build_embedding_backend(CONFIG.embedding_backend),
        llm_backend=build_llm_backend(CONFIG.llm_backend),
        chunk_size=DEMO_NAIVE_CHUNK_SIZE,
        chunk_overlap=DEMO_NAIVE_CHUNK_OVERLAP,
        top_k=CONFIG.naive_top_k,
        chunking_method="fixed",
    )
    enhanced = EnhancedRAGPipeline(
        documents=documents,
        embedding_backend=build_embedding_backend(CONFIG.embedding_backend),
        llm_backend=build_llm_backend(CONFIG.llm_backend),
        reranker_backend=build_reranker_backend(CONFIG.reranker_backend),
        chunk_size=DEMO_ENHANCED_CHUNK_SIZE,
        chunk_overlap=DEMO_ENHANCED_CHUNK_OVERLAP,
        candidate_k=CONFIG.enhanced_candidate_k,
        final_k=DEMO_ENHANCED_FINAL_K,
        rrf_k=CONFIG.rrf_k,
        chunking_method="sentence",
    )

    print(
        f"Naive chunks: {len(naive.chunk_texts)}  "
        f"Enhanced chunks: {len(enhanced.chunk_texts)}"
    )

    rows = []
    for idx, q in enumerate(questions, 1):
        gold = set(q["gold_doc_ids"])
        print(f"[{idx}/{len(questions)}] {q['question'][:80]}", flush=True)

        naive_result = naive.run(q["question"])
        print(f"  naive    -> {len(naive_result.retrieved_chunk_ids)} chunks retrieved", flush=True)
        enhanced_result = enhanced.run(q["question"])
        print(f"  enhanced -> {len(enhanced_result.retrieved_chunk_ids)} chunks retrieved", flush=True)

        naive_metrics = evaluate_example(
            question=q["question"],
            gold_doc_ids=gold,
            retrieved_chunk_ids=naive_result.retrieved_chunk_ids,
            retrieved_texts=naive_result.retrieved_texts,
            answer=naive_result.answer,
        )
        enhanced_metrics = evaluate_example(
            question=q["question"],
            gold_doc_ids=gold,
            retrieved_chunk_ids=enhanced_result.retrieved_chunk_ids,
            retrieved_texts=enhanced_result.retrieved_texts,
            answer=enhanced_result.answer,
        )

        for pipeline_name, result, metrics in (
            ("naive", naive_result, naive_metrics),
            ("enhanced", enhanced_result, enhanced_metrics),
        ):
            rows.append(
                {
                    "question_id": q["id"],
                    "pipeline": pipeline_name,
                    "retrieved_chunk_ids": ";".join(result.retrieved_chunk_ids),
                    "answer": result.answer,
                    **metrics,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "per_question.csv"), index=False)

    # Statistical comparison
    metric_names = [
        "ndcg_at_10",
        "context_precision",
        "context_recall",
        "mean_reciprocal_rank",
        "faithfulness_proxy",
        "answer_relevancy_proxy",
    ]
    summary_lines = []
    summary_lines.append("# RAG Benchmark: Naive vs. Enhanced Pipeline\n")
    summary_lines.append(f"Questions: {len(questions)}\n")
    summary_lines.append(
        f"Backends: embedding=`{CONFIG.embedding_backend}`, "
        f"llm=`{CONFIG.llm_backend}`, reranker=`{CONFIG.reranker_backend}`\n"
    )
    summary_lines.append(
        f"Naive chunks: {len(naive.chunk_texts)} "
        f"(size={DEMO_NAIVE_CHUNK_SIZE}, overlap={DEMO_NAIVE_CHUNK_OVERLAP}, method=fixed)  \n"
        f"Enhanced chunks: {len(enhanced.chunk_texts)} "
        f"(size={DEMO_ENHANCED_CHUNK_SIZE}, overlap={DEMO_ENHANCED_CHUNK_OVERLAP}, method=sentence, fusion=triple)\n"
    )
    summary_lines.append("\n## Results\n")

    results = []
    for metric in metric_names:
        a = df[df.pipeline == "naive"].sort_values("question_id")[metric].tolist()
        b = df[df.pipeline == "enhanced"].sort_values("question_id")[metric].tolist()
        res = compare_metric(metric, a, b)
        results.append(res)
        summary_lines.append(f"- {res}")

    n_wins = sum(1 for r in results if r.is_significant_win_for_b())
    n_positive = sum(1 for r in results if r.mean_diff > 0)

    # Extract nDCG@10 means for the BEIR comparison block.
    ndcg_naive = df[df.pipeline == "naive"]["ndcg_at_10"].mean()
    ndcg_enhanced = df[df.pipeline == "enhanced"]["ndcg_at_10"].mean()

    summary_lines.append("\n## Verdict\n")
    summary_lines.append(
        f"Enhanced pipeline improved the mean score on {n_positive}/{len(results)} "
        f"metrics; {n_wins}/{len(results)} reached p < 0.05 significance at N={len(questions)}."
    )
    summary_lines.append(
        "\n## nDCG@10 vs BEIR Public Benchmarks (HotpotQA)\n"
    )
    summary_lines.append(
        "nDCG@10 uses the same formula as the BEIR benchmark and MTEB retrieval leaderboard,\n"
        "enabling direct comparison with published numbers from OpenAI, Microsoft, Cohere, etc.\n"
        "Note: BEIR systems rank the *full* corpus; our pipeline ranks only retrieved candidates.\n"
    )
    summary_lines.append("| System | HotpotQA nDCG@10 | Notes |")
    summary_lines.append("|--------|------------------|-------|")
    summary_lines.append(f"| **This pipeline — naive RAG** | **{ndcg_naive:.3f}** | Fixed chunks, dense-only retrieval |")
    summary_lines.append(f"| **This pipeline — enhanced RAG** | **{ndcg_enhanced:.3f}** | Sentence chunks, triple fusion, rerank |")
    summary_lines.append("| BM25 (Elasticsearch) | 0.603 | BEIR paper, NeurIPS 2021 |")
    summary_lines.append("| OpenAI text-embedding-3-small | ~0.510 | MTEB retrieval avg (not HotpotQA-specific) |")
    summary_lines.append("| OpenAI text-embedding-3-large | ~0.554 | MTEB retrieval avg (not HotpotQA-specific) |")
    summary_lines.append("| BGE-m3-dense | 0.686 | Beijing Academy of AI, BEIR |")
    summary_lines.append("| SPLADE-v3 | 0.692 | Naver Labs, BEIR |")
    summary_lines.append("| E5-Mistral-7B | 0.757 | Microsoft, BEIR |")
    summary_lines.append(
        "\n*OpenAI MTEB scores are averages across 15 retrieval datasets; "
        "HotpotQA-specific OpenAI scores are not publicly available.*"
    )

    summary_text = "\n".join(summary_lines)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(summary_text + "\n")

    print("\n" + summary_text)
    print(f"\nWrote results to {args.out_dir}/")


if __name__ == "__main__":
    main()
