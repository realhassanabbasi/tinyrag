"""Show the evidence behind an answer: the retrieved chunks, ranked.

`ask` answers with one sentence, but the retriever scores several chunks
and the rest were thrown away. These helpers turn the ranked list into a
small, honest evidence table — doc, chunk, score, excerpt — so anyone can
check the answer against what the index actually returned. Zero-score
chunks are never evidence: they matched nothing.
"""

from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def _excerpt(text: str, chars: int = 200) -> str:
    """One line of the chunk, cut at `chars` with an ellipsis when cut."""
    flat = _WS_RE.sub(" ", text).strip()
    if len(flat) <= chars:
        return flat
    return flat[:chars].rstrip() + "…"


def evidence_payload(
    ranked: list[tuple[int, float]],
    chunks: list[dict],
    n: int = 3,
    excerpt_chars: int = 200,
) -> list[dict]:
    """Top-`n` retrieved chunks as evidence rows.

    Each row: rank (1-based), doc, chunk_id, score (rounded to 2dp),
    excerpt. Chunks scoring 0.0 are dropped — they matched nothing.
    """
    rows = []
    for rank, (idx, score) in enumerate(ranked[: max(n, 0)], start=1):
        if score <= 0.0:
            continue
        chunk = chunks[idx]
        rows.append({
            "rank": rank,
            "doc": chunk["doc"],
            "chunk_id": chunk["chunk_id"],
            "score": round(score, 2),
            "excerpt": _excerpt(chunk["text"], excerpt_chars),
        })
    return rows


def format_evidence(rows: list[dict]) -> str:
    """Text view: one numbered line per evidence row."""
    lines = ["Evidence:"]
    for row in rows:
        lines.append(
            f"  {row['rank']}. {row['doc']} (chunk {row['chunk_id']}, "
            f"{row['score']:.2f}) — \"{row['excerpt']}\""
        )
    return "\n".join(lines)
