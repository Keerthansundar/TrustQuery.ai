"""Two-stage retrieval: vector search (top-K) -> cross-encoder rerank -> top-N, with a confidence signal."""
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from rag.chunker import Chunk


@dataclass
class Hit:
    chunk: Chunk
    similarity: float                    # cosine similarity from the vector search
    rerank_score: Optional[float] = None


@dataclass
class RetrievalResult:
    hits: list[Hit]
    top_similarity: float = 0.0
    # Phase 4 uses this to abstain instead of guessing
    low_confidence: bool = True
    reranked: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(self, embedder: Any, store: Any, reranker: Any, *, top_k_retrieve: int = 8,
                 top_k_final: int = 4, min_similarity: float = 0.35):
        self.embedder, self.store, self.reranker = embedder, store, reranker
        self.k_retrieve, self.k_final, self.min_similarity = top_k_retrieve, top_k_final, min_similarity

    def retrieve(self, question: str) -> RetrievalResult:
        t0 = time.perf_counter()
        raw = self.store.query(
            self.embedder.embed_query(question), self.k_retrieve)
        t1 = time.perf_counter()
        hits = [Hit(c, sim) for c, sim in raw]
        scores = self.reranker.score(
            question, [h.chunk.text for h in hits]) if hits else None
        if scores is not None:
            for h, sc in zip(hits, scores):
                h.rerank_score = sc
            hits.sort(key=lambda h: h.rerank_score, reverse=True)
        # confidence = best vector match
        top_sim = max((h.similarity for h in hits), default=0.0)
        return RetrievalResult(
            hits=hits[: self.k_final], top_similarity=top_sim,
            low_confidence=top_sim < self.min_similarity, reranked=scores is not None,
            timings_ms={"retrieve_ms": (t1 - t0) * 1000, "rerank_ms": (time.perf_counter() - t1) * 1000})
