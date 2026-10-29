"""Extractive answering: pick the best sentence from the top chunks.

No generation, no LLM — the answer is always a real sentence from your
documents, quoted with its source. If nothing matches well, say so.
"""

from __future__ import annotations

import re

from .vectorizer import tokenize

_SENT_RE = re.compile(r"[^.!?]+[.!?]")
_MIN_SCORE = 0.05


def split_sentences(text: str) -> list[str]:
    """Split per line first (so titles don't glue to the next sentence),
    then on sentence punctuation. Fragments without punctuation survive."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [s.strip() for s in _SENT_RE.findall(line) if s.strip()]
        consumed = sum(len(p) for p in parts)
        if consumed < len(line):  # leftover fragment (e.g. a title)
            rest = _SENT_RE.sub("", line).strip()
            if rest:
                parts.insert(0, rest)
        out.extend(parts)
    return out


def _miss() -> dict:
    return {
        "text": "I couldn't find anything relevant in your documents.",
        "doc": None,
        "chunk_id": None,
        "confidence": 0.0,
    }


def answer(
    query_terms: set[str],
    idf: dict[str, float],
    ranked: list[tuple[int, float]],
    chunks: list[dict],
    min_score: float = _MIN_SCORE,
) -> dict:
    """Build an answer dict: text, source doc, chunk id, confidence.

    Sentences are scored by summed IDF weight of matched query terms, so a
    rare term (backpropagation) beats two common ones. Confidence = the
    retrieval score of the top chunk (0..1).
    """
    if not ranked or ranked[0][1] < min_score:
        return _miss()
    best_sent, best_weight, best_idx = "", -1.0, ranked[0][0]
    for idx, _ in ranked:
        for sent in split_sentences(chunks[idx]["text"]):
            weight = sum(idf.get(t, 0.0) for t in set(tokenize(sent))
                         if t in query_terms)
            if weight > best_weight:
                best_weight, best_sent, best_idx = weight, sent, idx
    if best_weight <= 0:
        return _miss()
    return {
        "text": best_sent,
        "doc": chunks[best_idx]["doc"],
        "chunk_id": chunks[best_idx]["chunk_id"],
        "confidence": round(ranked[0][1], 2),
    }
