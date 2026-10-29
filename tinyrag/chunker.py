"""Split documents into overlapping word chunks for retrieval."""

from __future__ import annotations

import os


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    """Sliding window over words. Overlap keeps context across boundaries."""
    words = text.split()
    if not words:
        return []
    if size <= 0:
        raise ValueError("chunk size must be positive")
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def chunk_file(path: str, size: int = 400, overlap: int = 80) -> tuple[str, list[str]]:
    """Read a .txt/.md file and chunk it. Returns (doc_name, chunks)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    doc_name = os.path.basename(path)
    return doc_name, chunk_text(text, size=size, overlap=overlap)


def chunk_folder(folder: str, size: int = 400, overlap: int = 80) -> list[dict]:
    """Chunk every .txt/.md file in a folder. Returns chunk records."""
    records = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".txt", ".md")):
            continue
        doc_name, chunks = chunk_file(os.path.join(folder, name), size, overlap)
        for i, text in enumerate(chunks):
            records.append({"doc": doc_name, "chunk_id": i, "text": text})
    return records
