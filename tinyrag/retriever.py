"""Rank chunks by cosine similarity to the query vector."""

from __future__ import annotations

from .vectorizer import cosine


def retrieve(
    query_vec: dict[str, float],
    chunk_vecs: list[dict[str, float]],
    k: int = 3,
) -> list[tuple[int, float]]:
    """Return top-k (chunk_index, score), best first."""
    scored = [(i, cosine(query_vec, vec)) for i, vec in enumerate(chunk_vecs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(k, 1)]
