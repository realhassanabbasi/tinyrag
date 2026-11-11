"""Search the index: find what the corpus actually says about a term.

`ask` answers one question with one sentence; `stats` counts what is
indexed. Neither shows you *where* a word lives. Search fills that gap:
it runs the same retriever `ask` uses and lists the matching chunks —
doc, chunk, score, excerpt — so you can see what the index knows before
you ask, and debug a bad answer by checking what the retriever actually
returned. The rows come from `evidence.evidence_payload`, the same
function behind `ask --sources`, so a search result always agrees with
what `ask` would cite for the same words. Zero-score chunks are never
matches: they matched nothing.
"""

from __future__ import annotations

from .evidence import evidence_payload
from .retriever import retrieve
from .vectorizer import tokenize, vectorize

DEFAULT_K = 5


def search_payload(index: dict, query: str, k: int = DEFAULT_K) -> dict:
    """Top-`k` matching chunks for `query`, best first.

    Payload: query, matches (evidence rows), total, searched_chunks.
    A query with no overlap scores 0 everywhere and returns no matches
    — an honest empty result, not a forced guess.
    """
    chunks = index.get("chunks", [])
    q_vec = vectorize(tokenize(query), index.get("idf", {}))
    ranked = retrieve(q_vec, index.get("vectors", []), k=k)
    matches = evidence_payload(ranked, chunks, n=k)
    return {
        "query": query,
        "matches": matches,
        "total": len(matches),
        "searched_chunks": len(chunks),
    }


def format_search(payload: dict) -> str:
    """Plain-text view of a search payload."""
    if not payload["matches"]:
        return (
            f"No matches for \"{payload['query']}\" — nothing in the "
            "index scored above 0. Try different words or index more "
            "documents."
        )
    lines = [
        f"{payload['total']} match"
        f"{'es' if payload['total'] != 1 else ''} for "
        f"\"{payload['query']}\" "
        f"({payload['searched_chunks']} chunk"
        f"{'s' if payload['searched_chunks'] != 1 else ''} searched):"
    ]
    for row in payload["matches"]:
        lines.append(
            f"  {row['rank']}. {row['doc']} (chunk {row['chunk_id']}, "
            f"{row['score']:.2f}) — \"{row['excerpt']}\""
        )
    return "\n".join(lines)
