"""Thin ChromaDB wrapper (persistent, cosine similarity, fingerprint-based staleness check)."""
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag.chunker import Chunk


class VectorStore:
    def __init__(self, path: Path, collection: str, client: Any = None):
        self._client = client or chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        self._name = collection
        self._col = self._client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    @property
    def fingerprint(self) -> Optional[str]:
        return (self._col.metadata or {}).get("fingerprint")

    def count(self) -> int:
        return self._col.count()

    def reset(self, fingerprint: str) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._client.create_collection(
            self._name, metadata={"hnsw:space": "cosine", "fingerprint": fingerprint})

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._col.add(
            ids=[c.id for c in chunks], embeddings=embeddings, documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "section": c.section, "index": c.index} for c in chunks])

    def query(self, embedding: list[float], k: int) -> list[tuple[Chunk, float]]:
        k = min(k, self.count())
        if k == 0:
            return []
        res = self._col.query(query_embeddings=[embedding], n_results=k)
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append((Chunk(cid, doc, meta["source"], meta["section"], meta["index"]), 1.0 - float(dist)))
        return hits