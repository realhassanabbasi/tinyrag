"""Merge two indexes into one searchable index.

`compare` answers one question set on two indexes side by side — it
never lets you ASK one question across both. Merge fills that gap:
point it at two indexes (notes + docs, v1 + v2, two teammates'
folders) and it builds a single index holding both sets of chunks,
with IDF rebuilt over the merged corpus, so the merged index answers
exactly like an index built from both folders at once.

Dedupe is honest and fail-closed:
- exact duplicates (same doc, chunk id and text — e.g. the same folder
  indexed twice) are kept once and counted as duplicates skipped.
- conflicts (same doc + chunk id, different text — the same doc name
  with different content in each index) keep the FIRST index's chunk;
  the second's is skipped and counted, and the doc names are listed
  in the report, never silently dropped.
"""

from __future__ import annotations

from .store import build_index, save_index
from .vectorizer import build_idf, tokenize, vectorize

DEFAULT_OUT = "merged.json"


def merge_indexes(index_a: dict, index_b: dict, out_path: str) -> dict:
    """Combine two indexes, save the merged index, return a report.

    Chunks are deduplicated by (doc, chunk_id): the first index wins
    every conflict, exact duplicates are kept once. IDF and vectors
    are rebuilt over the merged corpus, so retrieval on the merged
    index behaves like a fresh index over both folders. File
    signatures from both indexes' meta are merged the same way
    (first index wins) so `refresh` keeps working on the docs it
    knows about.

    Report: per-side doc/chunk counts, merged totals, duplicates and
    conflicts skipped, the conflicted doc names, and the output path.
    """
    chunks_a = index_a.get("chunks", [])
    chunks_b = index_b.get("chunks", [])
    seen: dict[tuple[str, int], str] = {}
    records: list[dict] = []
    duplicates = 0
    conflicts = 0
    conflict_docs: list[str] = []
    for record in chunks_a + chunks_b:
        key = (record["doc"], record["chunk_id"])
        if key not in seen:
            seen[key] = record["text"]
            records.append(record)
        elif seen[key] == record["text"]:
            duplicates += 1
        else:
            conflicts += 1
            if record["doc"] not in conflict_docs:
                conflict_docs.append(record["doc"])
    records.sort(key=lambda r: (r["doc"], r["chunk_id"]))

    docs_tokens = [tokenize(r["text"]) for r in records]
    idf = build_idf(docs_tokens)
    vectors = [vectorize(t, idf) for t in docs_tokens]
    meta_a = index_a.get("meta", {}).get("docs", {})
    meta_b = index_b.get("meta", {}).get("docs", {})
    meta = dict(meta_b)
    meta.update(meta_a)
    save_index(
        build_index(records, idf, vectors, meta={"docs": meta}),
        out_path,
    )
    return {
        "docs_a": len({r["doc"] for r in chunks_a}),
        "chunks_a": len(chunks_a),
        "docs_b": len({r["doc"] for r in chunks_b}),
        "chunks_b": len(chunks_b),
        "docs": len({r["doc"] for r in records}),
        "chunks": len(records),
        "duplicates_skipped": duplicates,
        "conflicts_skipped": conflicts,
        "conflict_docs": conflict_docs,
        "out": out_path,
    }


def format_merge(report: dict) -> str:
    """One honest line per merge class, plus the totals."""
    lines = [
        f"merged {report['docs_a']} + {report['docs_b']} documents, "
        f"{report['chunks_a']} + {report['chunks_b']} chunks "
        f"→ {report['docs']} documents, {report['chunks']} chunks"
    ]
    if report["duplicates_skipped"]:
        lines.append(
            f"duplicates skipped: {report['duplicates_skipped']} "
            "(same chunk in both indexes, kept once)"
        )
    if report["conflicts_skipped"]:
        lines.append(
            f"conflicts: {report['conflicts_skipped']} — first index kept "
            f"for: {', '.join(report['conflict_docs'])}"
        )
    lines.append(f"wrote merged index to {report['out']}")
    return "\n".join(lines)
