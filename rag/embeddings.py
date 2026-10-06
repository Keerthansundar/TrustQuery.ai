"""Local embeddings through Ollama (nomic-embed-text)."""
from typing import Any, Optional

import ollama

from config import get_settings

BATCH = 32


class OllamaEmbedder:
    def __init__(self, client: Any = None, model: Optional[str] = None):
        s = get_settings()
        self.client = client or ollama.Client(host=s.ollama_host)
        self.model = model or s.embed_model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            out.extend(self.client.embed(model=self.model,
                       input=texts[i:i + BATCH]).embeddings)
        return [list(v) for v in out]

    # nomic-embed-text is trained with task prefixes; using them noticeably improves retrieval.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"search_query: {text}"])[0]
