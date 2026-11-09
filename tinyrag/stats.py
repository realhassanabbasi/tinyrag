"""Index stats: what is actually in the index, at a glance.

Everything here is read straight from the saved index — no re-indexing,
no re-scoring. Document counts come from the chunk records, vocabulary
size from the stored IDF table, and the most common terms are counted
from the chunk texts themselves. A term is counted once per chunk it
appears in, so a word repeated fifty times in one chunk still counts
once — what matters is how widely a term shows up, not how loud it is.
"""

from __future__ import annotations

from .vectorizer import tokenize

TOP_TERMS = 10


def stats_payload(index: dict) -> dict:
    """Summarize an index: docs, chunks, vocabulary, per-doc rows, top terms.

    Works on any index the tool built (v1.0.0 onward — only chunks, idf
    and vectors are needed; meta is ignored). An empty index reports
    zeros with has_data False instead of pretending to be useful.
    """
    chunks = index.get("chunks", [])
    idf = index.get("idf", {})

    docs: dict[str, dict[str, int]] = {}
    for record in chunks:
        row = docs.setdefault(record["doc"], {"chunks": 0, "words": 0})
        row["chunks"] += 1
        row["words"] += len(record["text"].split())

    total_words = sum(row["words"] for row in docs.values())
    avg_chunk_words = (
        round(total_words / len(chunks), 1) if chunks else 0.0
    )

    # How many chunks each term appears in (document frequency, by chunk).
    term_chunks: dict[str, int] = {}
    for record in chunks:
        for term in set(tokenize(record["text"])):
            term_chunks[term] = term_chunks.get(term, 0) + 1
    top_terms = [
        {"term": term, "chunks": n}
        for term, n in sorted(
            term_chunks.items(), key=lambda kv: (-kv[1], kv[0])
        )[:TOP_TERMS]
    ]

    per_doc = [
        {"doc": name, "chunks": row["chunks"], "words": row["words"]}
        for name, row in sorted(docs.items())
    ]
    largest = max(per_doc, key=lambda r: (r["chunks"], r["words"]), default=None)

    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "terms": len(idf),
        "total_words": total_words,
        "avg_chunk_words": avg_chunk_words,
        "docs": per_doc,
        "largest_doc": largest,
        "top_terms": top_terms,
        "has_data": bool(chunks),
    }


def format_stats(payload: dict) -> str:
    """Plain-text view of a stats payload."""
    if not payload["has_data"]:
        return "index is empty — run `tinyrag index <folder>` first"
    lines = [
        f"Documents: {payload['documents']}",
        f"Chunks: {payload['chunks']}",
        f"Vocabulary: {payload['terms']} terms",
        (f"Words: {payload['total_words']:,} "
         f"(avg {payload['avg_chunk_words']} per chunk)"),
    ]
    largest = payload["largest_doc"]
    if largest is not None:
        lines.append(
            f"Largest document: {largest['doc']} "
            f"({largest['chunks']} chunk"
            f"{'s' if largest['chunks'] != 1 else ''}, "
            f"{largest['words']:,} words)"
        )
    lines.append("")
    lines.append("Per document:")
    for row in payload["docs"]:
        lines.append(
            f"  {row['doc']} — {row['chunks']} chunk"
            f"{'s' if row['chunks'] != 1 else ''}, "
            f"{row['words']:,} words"
        )
    if payload["top_terms"]:
        lines.append("")
        lines.append("Most common terms (by chunks they appear in):")
        for row in payload["top_terms"]:
            lines.append(
                f"  {row['term']} — in {row['chunks']} chunk"
                f"{'s' if row['chunks'] != 1 else ''}"
            )
    return "\n".join(lines)
