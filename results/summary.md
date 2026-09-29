# RAG Benchmark: Naive vs. Enhanced Pipeline

Questions: 150

Backends: embedding=`openai`, llm=`openai`, reranker=`cross_encoder`

Naive chunks: 1988 (size=100, overlap=0, method=fixed)  
Enhanced chunks: 1570 (size=180, overlap=40, method=sentence, fusion=triple)


## Results

- ndcg_at_10               naive=0.733  enhanced=0.793  diff=+0.060  95% CI=[+0.023, +0.099]  wilcoxon p=0.0035  ttest p=0.0021  effect_r=0.401  (enhanced > naive, SIGNIFICANT)
- context_precision        naive=0.498  enhanced=0.524  diff=+0.027  95% CI=[-0.004, +0.058]  wilcoxon p=0.1893  ttest p=0.0962  effect_r=0.193  (enhanced > naive, not significant)
- context_recall           naive=0.727  enhanced=0.783  diff=+0.057  95% CI=[+0.013, +0.103]  wilcoxon p=0.0131  ttest p=0.0127  effect_r=0.378  (enhanced > naive, SIGNIFICANT)
- mean_reciprocal_rank     naive=0.869  enhanced=0.929  diff=+0.060  95% CI=[+0.021, +0.101]  wilcoxon p=0.0064  ttest p=0.0042  effect_r=0.579  (enhanced > naive, SIGNIFICANT)
- faithfulness_proxy       naive=0.686  enhanced=0.742  diff=+0.056  95% CI=[+0.005, +0.109]  wilcoxon p=0.0206  ttest p=0.0324  effect_r=0.268  (enhanced > naive, SIGNIFICANT)
- answer_relevancy_proxy   naive=0.671  enhanced=0.730  diff=+0.059  95% CI=[+0.009, +0.111]  wilcoxon p=0.0508  ttest p=0.0244  effect_r=0.281  (enhanced > naive, not significant)

## Verdict

Enhanced pipeline improved the mean score on 6/6 metrics; 4/6 reached p < 0.05 significance at N=150.

## nDCG@10 vs BEIR Public Benchmarks (HotpotQA)

nDCG@10 uses the same formula as the BEIR benchmark and MTEB retrieval leaderboard,
enabling direct comparison with published numbers from OpenAI, Microsoft, Cohere, etc.
Note: BEIR systems rank the *full* corpus; our pipeline ranks only retrieved candidates.

| System | HotpotQA nDCG@10 | Notes |
|--------|------------------|-------|
| **This pipeline — naive RAG** | **0.733** | Fixed chunks, dense-only retrieval |
| **This pipeline — enhanced RAG** | **0.793** | Sentence chunks, triple fusion, rerank |
| BM25 (Elasticsearch) | 0.603 | BEIR paper, NeurIPS 2021 |
| OpenAI text-embedding-3-small | ~0.510 | MTEB retrieval avg (not HotpotQA-specific) |
| OpenAI text-embedding-3-large | ~0.554 | MTEB retrieval avg (not HotpotQA-specific) |
| BGE-m3-dense | 0.686 | Beijing Academy of AI, BEIR |
| SPLADE-v3 | 0.692 | Naver Labs, BEIR |
| E5-Mistral-7B | 0.757 | Microsoft, BEIR |

*OpenAI MTEB scores are averages across 15 retrieval datasets; HotpotQA-specific OpenAI scores are not publicly available.*
