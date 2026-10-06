"""Context builder (Phase 3: RAG).

Always included : the live DB schema (tiny, authoritative: exact table/column names).
Retrieved (RAG) : only the business-rule chunks relevant to the question
                  (vector search -> cross-encoder rerank -> top-N).
Fallback        : if the RAG stack is unavailable (Ollama embed model missing, Chroma error, ...),
                  all docs are used, exactly like Phase 2, so the app keeps working.
"""
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from config import get_settings
from database.schema import get_data_window, get_schema_text
from rag.embeddings import OllamaEmbedder
from rag.ingest import ingest
from rag.reranker import get_reranker
from rag.retriever import RetrievalResult, Retriever
from rag.vector_store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class ContextBundle:
    text: str
    sources: list[str] = field(default_factory=list)
    data_window: str = ""
    mode: str = "rag"
    low_confidence: bool = False               # Phase 4 uses this to abstain
    retrieval: Optional[RetrievalResult] = None


@lru_cache(maxsize=1)
def _get_retriever() -> Retriever:
    s = get_settings()
    embedder = OllamaEmbedder()
    store = VectorStore(s.chroma_dir, s.rag_collection)
    # no-op when the index is already up to date
    n = ingest(store, embedder, s)
    if n:
        log.info("RAG index (re)built: %d chunks", n)
    return Retriever(embedder, store, get_reranker(s.rerank_enabled, s.rerank_model),
                     top_k_retrieve=s.rag_top_k_retrieve, top_k_final=s.rag_top_k_final,
                     min_similarity=s.rag_min_similarity)


def reset_retriever_cache() -> None:
    clear = getattr(_get_retriever, "cache_clear", None)
    if clear:
        clear()


def _data_window() -> str:
    window = get_data_window()
    return f"orders from {window[0]} to {window[1]}" if window else "unknown"


def _full_docs_context(reason: str) -> ContextBundle:
    s = get_settings()
    parts, sources = [f"## Live database schema\n{get_schema_text()}"], [
        "Live database schema"]
    for doc in sorted(s.docs_dir.glob("*.md")):
        parts.append(doc.read_text(encoding="utf-8").strip())
        sources.append(f"{doc.name} (all docs - RAG unavailable)")
    return ContextBundle("\n\n".join(parts), sources, _data_window(), mode=f"full-docs ({reason})")


def build_context(question: str) -> ContextBundle:
    try:
        result = _get_retriever().retrieve(question)
    except Exception as exc:
        log.warning("RAG failed (%s); falling back to full docs.", exc)
        # retry initialisation next time (e.g. after `ollama pull`)
        reset_retriever_cache()
        return _full_docs_context(type(exc).__name__)

    parts = [f"## Live database schema\n{get_schema_text()}"]
    sources = ["Live database schema"]
    if result.hits:
        parts.append("## Retrieved business rules (most relevant first)")
    for h in result.hits:
        parts.append(h.chunk.text)
        score = h.rerank_score if h.rerank_score is not None else h.similarity
        sources.append(
            f"{h.chunk.source} > {h.chunk.section}  (score {score:.2f})")
    return ContextBundle("\n\n".join(parts), sources, _data_window(), mode="rag",
                         low_confidence=result.low_confidence, retrieval=result)
