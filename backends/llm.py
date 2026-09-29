"""
LLM backends, used for two things in this benchmark:

1. HyDE query transformation (enhanced pipeline only): "write a hypothetical
   passage that would answer this question", which is then embedded for
   retrieval instead of the raw question.

2. Answer synthesis: given retrieved chunks + question, generate the final
   answer that gets scored by the evaluation metrics.

- TemplateLLM: a dependency-free, deterministic fallback used for the
  in-sandbox demo. It does NOT do real language generation -- it is a stand-in
  so the pipeline is fully runnable end-to-end without API keys. The HyDE
  "hypothetical document" it produces is just the question's content words
  echoed into a templated sentence, and its "answer" is the most relevant
  sentence extracted from the retrieved context via lexical overlap. This
  is clearly NOT what should produce your published numbers.

- AnthropicLLM: real generation via the Anthropic API (claude-sonnet-4-6).
- OpenAILLM: real generation via the OpenAI API (for the GPT-4 comparison
  arm, or for RAGAS's judge model).
- OllamaLLM: FREE, fully local generation via Ollama (https://ollama.com).
  No API key, no per-token cost, no internet required after the model is
  pulled. Requires the Ollama server running locally and a model pulled
  (e.g. `ollama pull llama3.2`). This is what makes the real benchmark
  runnable at zero marginal cost -- see README.md "Free real benchmark".

Select via config.llm_backend in {"template", "anthropic", "openai", "ollama"}.
"""

from __future__ import annotations

import abc
import re


class LLMBackend(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        ...

    def hyde_document(self, question: str) -> str:
        """Generate a hypothetical document that would answer `question`."""
        prompt = (
            "Write a short, factual passage (3-5 sentences) that directly "
            f"answers the following question, as if it were an excerpt from "
            f"a reference document. Question: {question}\n\nPassage:"
        )
        return self.generate(prompt, max_tokens=200)

    def synthesize_answer(self, question: str, context_chunks: list[str]) -> str:
        """Generate a final answer given retrieved context."""
        context = "\n\n".join(
            f"[{i+1}] {c}" for i, c in enumerate(context_chunks)
        )
        prompt = (
            "Answer the question using ONLY the provided context. "
            "Be concise (1-2 sentences). If the context does not contain "
            "the answer, say so explicitly.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        return self.generate(prompt, max_tokens=200)


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "does", "do", "did", "has", "have",
    "had", "be", "been", "that", "this", "it", "its", "as", "by", "with",
    "from", "than", "also", "also,",
}


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if s]


class TemplateLLM(LLMBackend):
    """Dependency-free deterministic stand-in. See module docstring."""

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # Not used directly -- hyde_document/synthesize_answer are overridden
        # below with deterministic lexical logic instead of free generation.
        return ""

    def hyde_document(self, question: str) -> str:
        words = _content_words(question)
        # A templated "hypothetical answer passage": just restates the
        # content words of the question in declarative form. This gives the
        # dense retriever a document-shaped query instead of a question-
        # shaped one, which is the structural point of HyDE -- even though
        # the *content* here is trivial compared to a real LLM's output.
        return (
            "This passage discusses " + ", ".join(words) + ". "
            + " ".join(words).capitalize() + " is described in detail below, "
            "including relevant facts, figures, and relationships."
        )

    def synthesize_answer(self, question: str, context_chunks: list[str]) -> str:
        q_words = set(_content_words(question))
        best_sent, best_score = "", -1
        for chunk in context_chunks:
            for sent in _split_sentences(chunk):
                s_words = set(_content_words(sent))
                if not s_words:
                    continue
                overlap = len(q_words & s_words) / max(len(q_words), 1)
                if overlap > best_score:
                    best_score, best_sent = overlap, sent
        return best_sent or "I cannot answer based on the provided context."


class AnthropicLLM(LLMBackend):
    """Requires `anthropic` package + ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import time
        from anthropic import RateLimitError
        client = self._client_lazy()
        while True:
            try:
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(block.text for block in resp.content if block.type == "text")
            except RateLimitError:
                print("  Anthropic rate limit, waiting 60s...", flush=True)
                time.sleep(60)


class OpenAILLM(LLMBackend):
    """Requires `openai` package + OPENAI_API_KEY."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import time
        from openai import RateLimitError, APIConnectionError
        client = self._client_lazy()
        time.sleep(0.3)  # stay well under 500 RPM
        while True:
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content or ""
            except RateLimitError:
                print("  LLM rate limit, waiting 60s...", flush=True)
                time.sleep(60)
            except APIConnectionError:
                print("  LLM connection error, retrying in 10s...", flush=True)
                time.sleep(10)


class OllamaLLM(LLMBackend):
    """
    FREE, fully local generation via Ollama.

    Install Ollama (https://ollama.com, free, runs on Mac/Windows/Linux),
    pull a small model once (e.g. `ollama pull llama3.2`), and make sure the
    Ollama server is running (it runs automatically after install, or start
    with `ollama serve`). No API key, no per-token cost, no rate limits, and
    no internet required once the model is downloaded.

    Uses only the standard library (urllib) to talk to Ollama's local HTTP
    API, so it adds zero extra pip dependencies.

    Model selection: pass `model=` explicitly, or set the OLLAMA_MODEL
    environment variable. Small, fast, good-enough-for-HyDE models include
    `llama3.2` (3B), `qwen2.5:3b`, and `phi3.5`. Larger models (e.g.
    `llama3.1:8b`) will give better HyDE documents and answer synthesis at
    the cost of slower local inference.
    """

    def __init__(self, model: str | None = None, host: str = "http://localhost:11434"):
        import os

        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.host = host

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import json
        import urllib.request

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is the Ollama server "
                f"running (`ollama serve`) and is the model pulled "
                f"(`ollama pull {self.model}`)? Original error: {e}"
            )
        return data.get("response", "")


def build_llm_backend(name: str, **kwargs) -> LLMBackend:
    import os
    if name == "template":
        return TemplateLLM()
    if name == "anthropic":
        model = os.environ.get("RAG_LLM_MODEL", "claude-sonnet-4-6")
        return AnthropicLLM(model=model, **kwargs)
    if name == "openai":
        model = os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini")
        return OpenAILLM(model=model, **kwargs)
    if name == "ollama":
        return OllamaLLM(**kwargs)
    raise ValueError(f"Unknown LLM backend: {name}")
