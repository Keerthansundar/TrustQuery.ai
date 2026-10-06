"""Central configuration. Every value can be overridden via environment / .env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Ollama) ---
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 4096

    # --- Paths ---
    db_path: Path = ROOT / "data" / "business.db"
    docs_dir: Path = ROOT / "data" / "business_docs"

    # --- RAG (Phase 3) ---
    embed_model: str = "nomic-embed-text"
    chroma_dir: Path = ROOT / "data" / "chroma"
    rag_collection: str = "trustquery_docs"
    chunk_max_chars: int = 450
    chunk_overlap_chars: int = 80
    rag_top_k_retrieve: int = 8        # first stage: vector search
    rag_top_k_final: int = 4           # after re-ranking: chunks sent to the LLM
    # below this the context is "low confidence" (calibrated in Phase 5)
    rag_min_similarity: float = 0.35
    rerank_enabled: bool = True
    # or "BAAI/bge-reranker-base" (bigger)
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- SQL safety / limits ---
    max_rows: int = 200            # hard cap on rows fetched from SQLite
    llm_max_rows: int = 25         # rows shown to the LLM in the authoritative block
    query_timeout_s: float = 5.0

    # --- Reliability ---
    max_sql_attempts: int = 2      # first try + 1 retry on recoverable SQL error
    max_answer_repairs: int = 1    # LLM re-writes allowed when numbers fail verification
    history_turns: int = 3         # conversation memory window


@lru_cache
def get_settings() -> Settings:
    return Settings()
