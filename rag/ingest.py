"""Index the business docs: load -> chunk -> embed -> store.   Run manually: python -m rag.ingest"""
import hashlib
from pathlib import Path

from config import Settings, get_settings
from rag.chunker import Chunk, chunk_markdown
from rag.embeddings import OllamaEmbedder
from rag.vector_store import VectorStore


def load_chunks(s: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in sorted(Path(s.docs_dir).glob("*.md")):
        chunks += chunk_markdown(doc.read_text(encoding="utf-8"), doc.name,
                                 s.chunk_max_chars, s.chunk_overlap_chars)
    return chunks


def fingerprint(chunks: list[Chunk], s: Settings) -> str:
    """Changes whenever the docs, chunking settings or embedding model change -> index auto-rebuilds."""
    h = hashlib.sha256(f"{s.embed_model}|{s.chunk_max_chars}|{s.chunk_overlap_chars}".encode())
    for c in chunks:
        h.update(c.id.encode() + c.text.encode())
    return h.hexdigest()[:16]


def ingest(store: VectorStore, embedder: OllamaEmbedder, s: Settings, force: bool = False) -> int:
    chunks = load_chunks(s)
    fp = fingerprint(chunks, s)
    if not force and store.fingerprint == fp and store.count() == len(chunks):
        return 0                                   # index is up to date
    store.reset(fp)
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return len(chunks)


if __name__ == "__main__":
    settings = get_settings()
    n = ingest(VectorStore(settings.chroma_dir, settings.rag_collection), OllamaEmbedder(), settings, force=True)
    print(f"Indexed {n} chunks from {settings.docs_dir} into {settings.chroma_dir}")