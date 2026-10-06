"""Markdown-header-aware chunking: split on headings first, then pack lines up to a size limit."""
import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str      # heading path + body: what gets embedded, reranked and shown to the LLM
    source: str    # file name
    section: str   # heading path, e.g. "Revenue Rules > Dates"
    index: int


def _hard_split(line: str, max_chars: int, overlap: int) -> list[str]:
    step = max(1, max_chars - overlap)
    return [line[i:i + max_chars] for i in range(0, len(line), step)]


def _pack(body: str, max_chars: int, overlap: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    pieces, current, size = [], [], 0
    for raw in body.splitlines():
        for line in (_hard_split(raw, max_chars, overlap) if len(raw) > max_chars else [raw]):
            if current and size + len(line) + 1 > max_chars:
                pieces.append("\n".join(current))
                carry = [current[-1]
                         ] if overlap and len(current[-1]) <= overlap else []
                current, size = carry, sum(len(x) + 1 for x in carry)
            current.append(line)
            size += len(line) + 1
    if current:
        pieces.append("\n".join(current))
    return pieces


def chunk_markdown(text: str, source: str, max_chars: int = 700, overlap_chars: int = 80) -> list[Chunk]:
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    path, lines = "", []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            body = "\n".join(lines).strip()
            if body:
                sections.append((path, body))
            lines = []
            level, title = len(m.group(1)), m.group(2).strip()
            stack = [(lv, t)
                     for lv, t in stack if lv < level] + [(level, title)]
            path = " > ".join(t for _, t in stack)
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append((path, "\n".join(lines).strip()))

    chunks: list[Chunk] = []
    for path, body in sections:
        for piece in _pack(body, max_chars, overlap_chars):
            n = len(chunks)
            chunks.append(
                Chunk(f"{source}::{n}", f"{path}\n{piece}" if path else piece, source, path, n))
    return chunks
