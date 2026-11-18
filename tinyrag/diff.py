"""Diff two indexes: what content changed between them, doc by doc.

`compare` answers one question set on two indexes — it shows answers
moving, never the content that moved. `merge` combines two indexes
but only reports its conflicts after the fact. Diff fills the gap
before either: give it two index files and it tells you, per document,
what is only in A, only in B, identical on both sides, or changed
(chunk counts plus how many chunks were added, removed, or rewritten).

The natural use is before a merge — a changed doc is exactly the doc
`merge` will resolve first-wins — and between two snapshots of the
same folder, where it answers "what did my docs actually change?".

Diff reads only the saved chunk records: no re-indexing, no re-scoring,
no IDF work. Two indexes are identical when every shared document has
the same chunk ids with the same text.
"""

from __future__ import annotations


def _doc_map(index: dict) -> dict[str, dict[int, str]]:
    """Chunk text per document: {doc: {chunk_id: text}}."""
    docs: dict[str, dict[int, str]] = {}
    for record in index.get("chunks", []):
        docs.setdefault(record["doc"], {})[record["chunk_id"]] = record["text"]
    return docs


def diff_indexes(index_a: dict, index_b: dict) -> dict:
    """Compare two indexes' contents. Returns a payload.

    Payload: per-side doc/chunk counts, shared-doc count, the sorted
    doc lists only in A / only in B / identical, and one row per
    changed doc (chunk counts on each side plus the added/removed/
    changed chunk ids — chunk counts are just len() of those lists).
    """
    docs_a = _doc_map(index_a)
    docs_b = _doc_map(index_b)
    names_a = set(docs_a)
    names_b = set(docs_b)

    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)
    identical = sorted(
        d for d in names_a & names_b if docs_a[d] == docs_b[d]
    )
    changed = []
    for doc in sorted(names_a & names_b):
        if doc in identical:
            continue
        chunks_a = docs_a[doc]
        chunks_b = docs_b[doc]
        ids_a = set(chunks_a)
        ids_b = set(chunks_b)
        changed.append({
            "doc": doc,
            "chunks_a": len(chunks_a),
            "chunks_b": len(chunks_b),
            "added": sorted(ids_b - ids_a),
            "removed": sorted(ids_a - ids_b),
            "changed": sorted(
                i for i in ids_a & ids_b if chunks_a[i] != chunks_b[i]
            ),
        })

    return {
        "docs_a": len(names_a),
        "chunks_a": sum(len(c) for c in docs_a.values()),
        "docs_b": len(names_b),
        "chunks_b": sum(len(c) for c in docs_b.values()),
        "shared": len(names_a & names_b),
        "only_a": only_a,
        "only_b": only_b,
        "identical": identical,
        "changed": changed,
    }


def format_diff(payload: dict) -> str:
    """Plain-text view of a diff payload."""
    lines = [
        (f"A: {payload['docs_a']} document"
         f"{'s' if payload['docs_a'] != 1 else ''}, "
         f"{payload['chunks_a']} chunk"
         f"{'s' if payload['chunks_a'] != 1 else ''}"),
        (f"B: {payload['docs_b']} document"
         f"{'s' if payload['docs_b'] != 1 else ''}, "
         f"{payload['chunks_b']} chunk"
         f"{'s' if payload['chunks_b'] != 1 else ''}"),
    ]
    for label, key in (("Only in A", "only_a"), ("Only in B", "only_b"),
                       ("Identical", "identical")):
        names = payload[key]
        lines.append(f"{label}: {len(names)}"
                     + (f" ({', '.join(names)})" if names else ""))
    changed = payload["changed"]
    lines.append(f"Changed: {len(changed)}")
    for row in changed:
        n_added = len(row["added"])
        n_removed = len(row["removed"])
        n_changed = len(row["changed"])
        lines.append(
            f"  {row['doc']} — {row['chunks_a']} chunk"
            f"{'s' if row['chunks_a'] != 1 else ''} → "
            f"{row['chunks_b']} chunk"
            f"{'s' if row['chunks_b'] != 1 else ''} "
            f"({n_added} added, {n_removed} removed, "
            f"{n_changed} changed)"
        )
    if not payload["only_a"] and not payload["only_b"] and not changed:
        lines.append("no differences — the indexes hold the same documents")
    return "\n".join(lines)
