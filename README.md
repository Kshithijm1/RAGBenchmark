# RAG Benchmark: Naive RAG vs. Enhanced RAG (HyDE + Hybrid Retrieval + Reranking)

A head-to-head, statistically-evaluated comparison of two retrieval-augmented
generation pipelines:

- **Naive RAG** -- the default recipe from the OpenAI cookbook / most
  LangChain tutorials: fixed-size chunks, single dense embedding model,
  cosine-similarity top-k retrieval, raw question as the query.
- **Enhanced RAG** -- three independently-motivated, published improvements
  layered on top:
  1. **HyDE** (Hypothetical Document Embeddings) -- generate a hypothetical
     answer passage and embed *that* for retrieval, closing the
     question-vs-document semantic gap.
  2. **Hybrid retrieval** -- BM25 (lexical) + dense (semantic) candidate
     lists combined via Reciprocal Rank Fusion, so exact-keyword matches
     that embeddings miss still surface.
  3. **Cross-encoder reranking** -- the fused candidate set is rescored by a
     model that jointly encodes (query, candidate) pairs, strictly more
     expressive than independently-encoded cosine similarity.

Both pipelines are evaluated on the same questions using **RAGAS metrics**
(Faithfulness, Answer Relevancy, Context Precision, Context Recall) and
compared with **paired statistical tests** (Wilcoxon signed-rank, paired
t-test, bootstrap confidence intervals, rank-biserial effect size).

---

## TL;DR -- how to run it

```bash
pip install -r requirements.txt
python run_benchmark.py
```

This runs the **local demo**: a 13-document / 12-question original
fictional corpus, with every backend swapped for a dependency-free fallback
(TF-IDF/SVD embeddings, lexical-overlap reranking, a templated "LLM"). It
proves the pipeline plumbing -- chunking, hybrid fusion, reranking,
evaluation, statistics -- runs correctly end to end and produces a report.

**The demo numbers are NOT the benchmark numbers.** They exist to validate
the code. The real, publishable benchmark requires real embeddings, a real
LLM, a real reranker, and a real dataset (HotpotQA/BEIR) at N >= 100-150
questions. See "Running the real benchmark" below.

---

## Architecture

```
config.py              -- all hyperparameters + backend selection (env vars)
chunking.py             -- fixed-size chunking (naive: no overlap, enhanced: overlap)
backends/
  embeddings.py          -- EmbeddingBackend: TfidfSvd (demo) | SentenceTransformer (FREE) | OpenAI
  llm.py                  -- LLMBackend: Template (demo) | Ollama (FREE) | Anthropic | OpenAI
  reranker.py             -- RerankerBackend: LexicalOverlap (demo) | CrossEncoder (FREE)
retrieval/
  sparse.py               -- BM25Retriever
  fusion.py               -- DenseRetriever + reciprocal_rank_fusion()
pipelines.py            -- NaiveRAGPipeline, EnhancedRAGPipeline
evaluation/
  metrics.py              -- context_precision/recall (exact) + faithfulness/
                             relevancy proxies (demo only)
  ragas_adapter.py         -- real RAGAS evaluation (LLM-judged metrics, free via Ollama)
  stats.py                -- paired Wilcoxon/t-test/bootstrap CI/effect size
data/
  build_sample_data.py    -- generates the demo corpus + QA set
  load_hotpotqa.py        -- FREE: downloads HotpotQA via `datasets`, converts to this format
run_benchmark.py        -- orchestration: load data -> run both pipelines ->
                            evaluate -> statistical comparison -> report
```

Every backend is selected via environment variable, so switching from "demo
mode" to "real mode" is a config change, not a code change:

```bash
export RAG_EMBEDDING_BACKEND=sentence_transformer  # tfidf (default) | sentence_transformer (free) | openai
export RAG_LLM_BACKEND=ollama                      # template (default) | ollama (free) | anthropic | openai
export RAG_RERANKER_BACKEND=cross_encoder          # lexical (default) | cross_encoder (free)
```

---

## How context precision/recall stay comparable across pipelines

Naive and enhanced use **different chunk sizes**, so chunk ids differ between
them. To keep `context_precision`/`context_recall` directly comparable, they
are computed at the **document level**: each QA example has a `gold_doc_ids`
set, and a retrieved chunk "counts" if its source document is in that set
(`chunk_id_to_doc_id()` in `evaluation/metrics.py`). This is exact -- no LLM
judge needed -- and is the most trustworthy signal in the demo run.

`faithfulness_proxy` and `answer_relevancy_proxy` are **lexical-overlap
stand-ins** for RAGAS's LLM-judged Faithfulness/Answer Relevancy, used only
because the demo's `TemplateLLM` doesn't do real generation. Replace these
with real RAGAS metrics for the published benchmark (next section).

---

## Running the real benchmark -- for FREE

Everything below runs with **zero API cost**: open embedding models, an
open cross-encoder reranker, and a local LLM via Ollama, all on a HotpotQA
sample. The architecture comparison (naive single-pass dense retrieval vs.
HyDE + hybrid retrieval + reranking) is the same regardless of which models
sit underneath -- the published HyDE/hybrid-retrieval/reranking results this
project is based on were not specific to OpenAI's models, so an
open-embedding "naive RAG" baseline is both a fair comparison and arguably a
*more* representative baseline (it's what most cost-conscious RAG tutorials
actually tell people to use).

### 1. Install the free pieces (one-time)

```bash
# Ollama: free, local LLM runtime (https://ollama.com) -- download and install,
# then pull a small model (≈2GB, one-time download, runs on CPU):
ollama pull llama3.2

# Python deps (all free/open-source):
pip install sentence-transformers datasets ragas langchain-ollama langchain-huggingface
```

`sentence-transformers` will download two small open models from
Hugging Face the first time they're used (`BAAI/bge-small-en-v1.5`, ~130MB,
for embeddings; `cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB, for
reranking) -- one-time, cached locally afterward, no further internet or
cost needed.

### 2. Get HotpotQA (free, via `datasets`)

```bash
python data/load_hotpotqa.py --n 150 --out-dir data/hotpotqa
```

This downloads HotpotQA's distractor split (free, Apache/CC-BY-SA licensed,
cached locally after first download), samples 150 questions, and writes
`data/hotpotqa/corpus.json` + `data/hotpotqa/qa.json` in this project's
format. Each question requires combining 2 of ~10 candidate paragraphs
(8 distractors) -- exactly the regime where hybrid retrieval + HyDE +
reranking are expected to help over single-pass dense retrieval.

**Why N=150**: effect sizes reported in the HyDE/hybrid-retrieval literature
are often in the Cohen's d ~ 0.3-0.5 range; at alpha=0.05/power=0.8, a paired
test needs roughly N~50-60 to detect d=0.4, but real data is noisier than
that calculation assumes. N=150 gives headroom and lets you report subgroup
results (e.g. 2-hop vs. 3+-hop questions) without losing power. `--n` can be
increased -- the loader and pipeline are linear in corpus size, just slower.

### 3. Configure free backends and run

```bash
export RAG_EMBEDDING_BACKEND=sentence_transformer   # BAAI/bge-small-en-v1.5, free, CPU
export RAG_LLM_BACKEND=ollama                       # local llama3.2, free
export RAG_RERANKER_BACKEND=cross_encoder           # ms-marco-MiniLM-L-6-v2, free, CPU
export OLLAMA_MODEL=llama3.2

python run_benchmark.py --data-dir data/hotpotqa --n 150
```

On a typical laptop CPU, embedding ~1,000-1,500 paragraphs and reranking
candidates for 150 questions takes a few minutes; the Ollama calls (one
HyDE generation + one answer synthesis per question, per pipeline) are the
slow part -- budget roughly 10-30 minutes total depending on your machine
and chosen model. Smaller/faster models (`llama3.2`, `qwen2.5:3b`) trade a
little HyDE/answer quality for speed; this mainly affects
faithfulness/answer-relevancy, not context precision/recall (which don't
depend on the LLM at all).

### 4. (Optional) Real RAGAS metrics, also free

`context_precision`/`context_recall` are already computed exactly (no LLM
needed) by `run_benchmark.py`. For real LLM-judged
faithfulness/answer-relevancy instead of the lexical proxies:

```python
from evaluation.ragas_adapter import evaluate_with_ragas

df = evaluate_with_ragas(
    questions=[...], answers=[...], contexts=[...], ground_truths=[...],
    judge_backend="ollama", judge_model="llama3.2",
)
```

Run once for naive's outputs and once for enhanced's outputs (same
questions, paired), then feed the resulting per-question columns into
`evaluation/stats.py`'s `compare_metric()` exactly as `run_benchmark.py`
does for the demo metrics.

### Where the strongest "beats the baseline" claim comes from

**Context precision/recall require no LLM at all** -- they're computed
exactly from HotpotQA's gold supporting-fact labels. This is also the metric
pair where hybrid retrieval (BM25 catches exact entity-name matches dense
embeddings can miss) + reranking (a cross-encoder discriminating between 10
similar-topic candidates) + HyDE (closing the question-vs-passage gap) have
the most well-documented effect in the literature, and it's the part of the
pipeline that's completely free regardless of which LLM you use. If you only
have time to report one rigorously-tested result, this is the one most
likely to be both real and significant at N=150.

---

## Optional: paid backends

If you'd rather use OpenAI/Anthropic for a stronger or more "recognizable"
baseline comparison (e.g. for a writeup that explicitly says "vs. OpenAI's
recommended embeddings"), the same config switches work:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export RAG_EMBEDDING_BACKEND=openai            # text-embedding-3-small
export RAG_LLM_BACKEND=anthropic               # claude-sonnet-4-6
export RAG_RERANKER_BACKEND=cross_encoder      # still free -- no paid reranker needed
```

For chunk sizes on real (longer) documents, `config.py`'s defaults (512/0
for naive, 256/64 for enhanced) are word-count approximations of common
token-based chunking recipes; for an exact match to "512 tokens" use a real
tokenizer (e.g. `tiktoken`) instead of whitespace splitting in
`chunking.py`. HotpotQA paragraphs are short enough that this doesn't matter
much for the loader above.


## Reading the statistical output

For each metric, `evaluation/stats.py` reports:

- **mean (naive) vs. mean (enhanced)** and their **difference**
- a **95% bootstrap CI** on that difference (10,000 resamples) -- if this
  interval excludes 0, that's a strong signal regardless of the p-value
- **Wilcoxon signed-rank p-value** (primary test -- non-parametric,
  appropriate for bounded [0,1] scores)
- **paired t-test p-value** (secondary, parametric check)
- **matched-pairs rank-biserial correlation** as effect size (-1 to 1)

A metric counts as a genuine win for the enhanced pipeline only if **the
mean difference is positive AND the Wilcoxon p-value < 0.05**
(`PairedTestResult.is_significant_win_for_b()`). Don't cherry-pick a metric
that happens to hit significance while ignoring others that don't --
`run_benchmark.py`'s summary reports all four metrics and an overall verdict
(`n/4 metrics improved`, `n/4 reached significance`) so the full picture is
visible.

---

## Honest caveats (read before posting results anywhere)

- **The demo numbers (TF-IDF embeddings, lexical reranker, template LLM, 13
  documents) are a correctness check, not a benchmark result.** They will
  not replicate on real data and should not be quoted as "RAG improvement
  numbers."
- **N=12 has essentially no statistical power.** Don't report p-values from
  the demo run as evidence of anything other than "the code runs."
- **The "naive RAG baseline" should be the standard, widely-recommended
  recipe** -- fixed chunks, single dense embedding model, raw-question
  top-k retrieval -- regardless of whether the embedding model underneath
  is free (`bge-small-en-v1.5`) or paid (`text-embedding-3-small`). The
  comparison being tested is the *retrieval architecture*
  (single-pass dense vs. HyDE+hybrid+rerank), not the embedding model, so
  the free version is not a "weaker" or less legitimate baseline -- just
  make sure both pipelines use the same embedding model so the comparison
  is apples-to-apples.
- **Faithfulness/Answer Relevancy require an LLM judge in the real run.**
  Don't substitute the lexical proxies in the published numbers -- they're
  there only so the demo doesn't need API keys.
- **Report all four metrics, even the ones where enhanced doesn't win.** A
  hybrid+rerank+HyDE pipeline typically improves context precision/recall
  substantially but may not move faithfulness/relevancy much if the
  underlying LLM is unchanged -- that's a real and expected finding, not a
  failure of the experiment.

## Suggested framing for a writeup

"Built and statistically benchmarked an enhanced RAG pipeline (HyDE + hybrid
BM25/dense retrieval + cross-encoder reranking) against the standard
single-pass dense retrieval baseline on N=150 HotpotQA questions, using only
free, open-weight models running locally (no API costs). Used paired
Wilcoxon signed-rank tests with bootstrap confidence intervals (not just raw
averages) to confirm [X]% relative improvement in context precision (p=...)
and [Y]% in context recall (p=...), with no significant regression in answer
faithfulness." -- fill in [X]/[Y]/p-values from your real run. Two things to
lead with: the methodology (paired tests, effect sizes, named dataset) and
the fact that it's reproducible by anyone for free -- both are differentiators,
since most "RAG demo" projects show a single qualitative example with a paid
API key, not a controlled, statistically-tested, zero-cost comparison.
