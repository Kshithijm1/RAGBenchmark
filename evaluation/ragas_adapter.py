"""
Adapter for the real benchmark: RAGAS's LLM-judged metrics.

This module is NOT used in the in-sandbox demo (which uses the lexical
proxies in metrics.py). It's here so the real benchmark run is a small
config change, not a rewrite.

FREE option (default, recommended): judge_backend="ollama" uses a local
Ollama model as both the LLM judge and the embedding model for RAGAS's
metrics. Zero API cost, no API key, runs entirely on your machine.

    pip install ragas langchain-ollama langchain-huggingface datasets
    ollama pull llama3.2          # or any model you prefer

    from evaluation.ragas_adapter import evaluate_with_ragas
    df = evaluate_with_ragas(
        questions=[...], answers=[...], contexts=[...], ground_truths=[...],
        judge_backend="ollama", judge_model="llama3.2",
    )

Paid options: judge_backend="openai" (gpt-4o, requires OPENAI_API_KEY) or
judge_backend="anthropic" (claude-sonnet-4-6, requires
ANTHROPIC_API_KEY + langchain-anthropic) if you want a stronger judge model
and don't mind the (typically small -- judging is cheap relative to
generation) API cost.
"""

from __future__ import annotations

from typing import List, Optional


def evaluate_with_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
    judge_backend: str = "ollama",
    judge_model: Optional[str] = None,
):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    if judge_backend == "ollama":
        # Fully free, fully local: Ollama for the judge LLM, a small free
        # HuggingFace sentence-transformer for the embeddings RAGAS itself
        # needs internally (separate from the embeddings used by the RAG
        # pipelines under test).
        from langchain_ollama import ChatOllama
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        llm = LangchainLLMWrapper(ChatOllama(model=judge_model or "llama3.2"))
        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        )
        result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)

    elif judge_backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from ragas.llms import LangchainLLMWrapper

        llm = LangchainLLMWrapper(
            ChatAnthropic(model=judge_model or "claude-sonnet-4-6")
        )
        result = evaluate(dataset, metrics=metrics, llm=llm)

    elif judge_backend == "openai":
        # RAGAS defaults to OpenAI (env-configured) if no llm= is passed.
        result = evaluate(dataset, metrics=metrics)

    else:
        raise ValueError(f"Unknown judge_backend: {judge_backend}")

    return result.to_pandas()
