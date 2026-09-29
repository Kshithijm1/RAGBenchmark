# Graph Report - .  (2026-06-30)

## Corpus Check
- 27 files · ~160,585 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 346 nodes · 532 edges · 24 communities (18 shown, 6 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 32 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Retrieval & Statistics|Retrieval & Statistics]]
- [[_COMMUNITY_Embedding Backend Core|Embedding Backend Core]]
- [[_COMMUNITY_Backend Abstractions|Backend Abstractions]]
- [[_COMMUNITY_Backend Factories|Backend Factories]]
- [[_COMMUNITY_LLM Backends|LLM Backends]]
- [[_COMMUNITY_Answer Evaluation|Answer Evaluation]]
- [[_COMMUNITY_OpenAI Embeddings|OpenAI Embeddings]]
- [[_COMMUNITY_RAG Pipeline Concepts|RAG Pipeline Concepts]]
- [[_COMMUNITY_LLM Implementations|LLM Implementations]]
- [[_COMMUNITY_HotpotQA Dataset|HotpotQA Dataset]]
- [[_COMMUNITY_Sample Corpus Data|Sample Corpus Data]]
- [[_COMMUNITY_Evaluation Metrics|Evaluation Metrics]]
- [[_COMMUNITY_Benchmark Results|Benchmark Results]]
- [[_COMMUNITY_HotpotQA Loader|HotpotQA Loader]]
- [[_COMMUNITY_Sample Data Builder|Sample Data Builder]]
- [[_COMMUNITY_RAGAS Adapter|RAGAS Adapter]]
- [[_COMMUNITY_Enhanced Chunk Overlap|Enhanced Chunk Overlap]]
- [[_COMMUNITY_Naive Chunk Overlap|Naive Chunk Overlap]]
- [[_COMMUNITY_Evaluation Module|Evaluation Module]]
- [[_COMMUNITY_Retrieval Module|Retrieval Module]]

## God Nodes (most connected - your core abstractions)
1. `EmbeddingBackend` - 17 edges
2. `LLMBackend` - 16 edges
3. `DenseRetriever` - 13 edges
4. `RerankerBackend` - 12 edges
5. `BM25Retriever` - 11 edges
6. `NaiveRAGPipeline` - 10 edges
7. `EnhancedRAGPipeline` - 10 edges
8. `evaluate_example()` - 9 edges
9. `main()` - 9 edges
10. `OpenAIEmbeddings` - 8 edges

## Surprising Connections (you probably didn't know these)
- `EmbeddingBackend` --uses--> `EmbeddingBackend`  [INFERRED]
  retrieval/fusion.py → backends/embeddings.py
- `LLMBackend` --uses--> `EmbeddingBackend`  [INFERRED]
  pipelines.py → backends/embeddings.py
- `EnhancedRAGPipeline` --uses--> `EmbeddingBackend`  [INFERRED]
  pipelines.py → backends/embeddings.py
- `NaiveRAGPipeline` --uses--> `EmbeddingBackend`  [INFERRED]
  pipelines.py → backends/embeddings.py
- `EmbeddingBackend` --uses--> `EmbeddingBackend`  [INFERRED]
  pipelines.py → backends/embeddings.py

## Import Cycles
- None detected.

## Communities (24 total, 6 thin omitted)

### Community 0 - "Retrieval & Statistics"
Cohesion: 0.05
Nodes (54): BEIR Benchmark, BM25Okapi (rank_bm25), BM25Retriever, Bootstrap Confidence Interval, _bootstrap_ci_mean_diff(), chunk_corpus(), compare_metric(), CONFIG (+46 more)

### Community 1 - "Embedding Backend Core"
Cohesion: 0.10
Nodes (30): EmbeddingBackend, Embedding backends.  EmbeddingBackend is the interface every dense retriever dep, Fit on the corpus (only meaningful for the TF-IDF/SVD backend)., LLMBackend, Return indices into `candidates`, best-first, length <= top_k., RerankerBackend, chunk_corpus(), fixed_size_chunks() (+22 more)

### Community 2 - "Backend Abstractions"
Cohesion: 0.06
Nodes (36): CrossEncoderReranker, EmbeddingBackend, LexicalOverlapReranker, OpenAIEmbeddings, PipelineConfig, RerankerBackend, SentenceTransformerEmbeddings, TfidfSvdEmbeddings (+28 more)

### Community 3 - "Backend Factories"
Cohesion: 0.11
Nodes (18): build_embedding_backend(), build_reranker_backend(), CrossEncoderReranker, LexicalOverlapReranker, Reranker backends.  Given a query and a list of candidate chunks (already retrie, FREE, open-weight cross-encoder reranker via sentence-transformers.      Default, PipelineConfig, Central configuration for the RAG benchmark.  Everything that differs between a (+10 more)

### Community 4 - "LLM Backends"
Cohesion: 0.10
Nodes (14): AnthropicLLM, build_llm_backend(), _content_words(), OllamaLLM, OpenAILLM, LLM backends, used for two things in this benchmark:  1. HyDE query transformati, Requires `anthropic` package + ANTHROPIC_API_KEY., Requires `openai` package + OPENAI_API_KEY. (+6 more)

### Community 5 - "Answer Evaluation"
Cohesion: 0.13
Nodes (25): Answer Relevancy Metric, answer_relevancy_proxy(), Anthropic Judge Backend, chunk_id_to_doc_id(), claude-sonnet-4-6 Model, _content_words(), context_precision(), Context Precision Metric (+17 more)

### Community 6 - "OpenAI Embeddings"
Cohesion: 0.12
Nodes (8): OpenAIEmbeddings, ndarray, FREE, open-weight embedding models via sentence-transformers.      Default is `B, Return an (n_texts, dim) array of L2-normalized vectors., Local, dependency-free dense embedding via TF-IDF + Truncated SVD (LSA)., text-embedding-3-small/large. Requires `openai` package + OPENAI_API_KEY., SentenceTransformerEmbeddings, TfidfSvdEmbeddings

### Community 7 - "RAG Pipeline Concepts"
Cohesion: 0.12
Nodes (20): Enhanced Chunking (Sentence-Aware with Overlap), Enhanced RAG Pipeline, Hybrid Retrieval (Dense + Lexical), HyDE (Hypothetical Document Embeddings), Naive Chunking (Fixed-Size, No Overlap), Naive RAG Pipeline, Reciprocal Rank Fusion (RRF), rrf_k (60) - Reciprocal Rank Fusion (+12 more)

### Community 8 - "LLM Implementations"
Cohesion: 0.16
Nodes (14): AnthropicLLM, LLMBackend, OllamaLLM, OpenAILLM, TemplateLLM, Answer Synthesis, Anthropic API (claude-sonnet-4-6), Ollama Local LLM Server (+6 more)

### Community 9 - "HotpotQA Dataset"
Cohesion: 0.16
Nodes (14): gold_doc_ids (Relevance Labels), HuggingFace datasets Library, Multi-Hop QA (2-document questions), QUESTIONS (12 multi-hop questions), HotpotQA Dataset (distractor setting), data/hotpotqa/corpus.json, data/hotpotqa/qa.json, sample_corpus.json (+6 more)

### Community 10 - "Sample Corpus Data"
Cohesion: 0.30
Nodes (14): DOCUMENTS (13 fictional docs), Aurora-Nimbus Acquisition (document), Aurora Dynamics (document), Aurora-Solace Partnership (document), Global Tech Council (document), Mira Chen (document), Nimbus Systems (document), Priya Nair (document) (+6 more)

### Community 11 - "Evaluation Metrics"
Cohesion: 0.31
Nodes (12): answer_relevancy_proxy(), chunk_id_to_doc_id(), _content_words(), context_precision(), context_recall(), evaluate_example(), faithfulness_proxy(), mean_reciprocal_rank() (+4 more)

### Community 12 - "Benchmark Results"
Cohesion: 0.28
Nodes (9): Benchmark Verdict: 2/6 metrics improved, 0/6 significant, Lexical Reranker Backend, nDCG@10 Metric, OpenAI Embedding Backend, OpenAI LLM Backend, Enhanced nDCG@10=0.664 (N=150), Naive nDCG@10=0.733 (N=150), results/run_log.txt (+1 more)

### Community 13 - "HotpotQA Loader"
Cohesion: 0.50
Nodes (4): main(), Convert HotpotQA (distractor setting) into this project's corpus/QA format.  Hot, Doc ids must not contain '::chunk' (used as a chunk_id separator) or     whitesp, _sanitize_id()

## Knowledge Gaps
- **1 isolated node(s):** `PipelineConfig`
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EmbeddingBackend` connect `Embedding Backend Core` to `Backend Factories`, `OpenAI Embeddings`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Why does `LLMBackend` connect `Embedding Backend Core` to `LLM Backends`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Why does `RerankerBackend` connect `Embedding Backend Core` to `Backend Factories`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `EmbeddingBackend` (e.g. with `LLMBackend` and `EnhancedRAGPipeline`) actually correct?**
  _`EmbeddingBackend` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `LLMBackend` (e.g. with `LLMBackend` and `EnhancedRAGPipeline`) actually correct?**
  _`LLMBackend` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `DenseRetriever` (e.g. with `LLMBackend` and `EnhancedRAGPipeline`) actually correct?**
  _`DenseRetriever` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Embedding backends.  EmbeddingBackend is the interface every dense retriever dep`, `Fit on the corpus (only meaningful for the TF-IDF/SVD backend).`, `Return an (n_texts, dim) array of L2-normalized vectors.` to the rest of the system?**
  _37 weakly-connected nodes found - possible documentation gaps or missing edges._