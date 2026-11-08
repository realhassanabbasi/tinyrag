"""Refresh an existing index: re-chunk only the files that changed.

An index built by `tinyrag index` is a snapshot. Documents grow, get
edited, get deleted — re-indexing the whole folder every time works but
wastes the work already done. Refresh compares each file's signature
(modification time + size) against the signatures stored in the index
and only re-chunks what actually changed:

- added:   file in the folder, not in the index — chunked fresh
- updated: signature changed since the last index — re-chunked
- removed: in the index, gone from the folder — dropped
- unchanged: signature matches — old chunks kept as-is

IDF is corpus-wide, so vectors are rebuilt over the full record set
every refresh — that part is cheap; the saved work is the re-reading
and re-chunking of unchanged files. A v1.1.0 index has no stored
signatures, so the first refresh after upgrading rebuilds everything
once (reported honestly as a legacy rebuild) and stores signatures
from then on.
"""

from __future__ import annotations

import os

from .chunker import chunk_file
from .store import build_index, load_index, save_index
from .vectorizer import build_idf, tokenize, vectorize


def file_signature(path: str) -> dict[str, int]:
    """Cheap change fingerprint: modification time + size."""
    st = os.stat(path)
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def scan_folder(folder: str) -> dict[str, dict[str, int]]:
    """Signatures for every .txt/.md file in a folder, by doc name."""
    out: dict[str, dict[str, int]] = {}
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".txt", ".md")):
            continue
        out[name] = file_signature(os.path.join(folder, name))
    return out


def refresh_index(folder: str, index_path: str) -> dict:
    """Refresh the index at index_path from folder. Returns a report.

    Raises FileNotFoundError when the index does not exist yet —
    run `tinyrag index <folder>` first.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    old = load_index(index_path)
    old_docs = {r["doc"] for r in old["chunks"]}
    old_sigs = old.get("meta", {}).get("docs", {})
    legacy = not old_sigs  # v1.1.0 indexes stored no signatures
    new_sigs = scan_folder(folder)

    added = sorted(set(new_sigs) - old_docs)
    removed = sorted(old_docs - set(new_sigs))
    updated = sorted(
        d for d in new_sigs
        if d in old_docs and (legacy or old_sigs.get(d) != new_sigs[d])
    )
    unchanged = sorted(
        d for d in new_sigs
        if d in old_docs and d not in updated
    )

    records = [r for r in old["chunks"] if r["doc"] in unchanged]
    for doc in added + updated:
        _, chunks = chunk_file(os.path.join(folder, doc))
        for i, text in enumerate(chunks):
            records.append({"doc": doc, "chunk_id": i, "text": text})
    records.sort(key=lambda r: (r["doc"], r["chunk_id"]))

    docs_tokens = [tokenize(r["text"]) for r in records]
    idf = build_idf(docs_tokens)
    vectors = [vectorize(t, idf) for t in docs_tokens]
    save_index(
        build_index(records, idf, vectors, meta={"docs": new_sigs}),
        index_path,
    )
    return {
        "added": added,
        "updated": updated,
        "removed": removed,
        "unchanged": unchanged,
        "docs": len(new_sigs),
        "chunks": len(records),
        "legacy": legacy,
    }


def format_refresh(report: dict) -> str:
    """One honest line per change class, plus the totals."""
    lines = []
    if report["legacy"]:
        lines.append("note: old index had no file signatures — rebuilt all docs once")
    for label, key in (("Added", "added"), ("Updated", "updated"),
                       ("Removed", "removed"), ("Unchanged", "unchanged")):
        names = report[key]
        lines.append(f"{label}: {len(names)}" + (f" ({', '.join(names)})" if names else ""))
    lines.append(f"index now: {report['docs']} documents, {report['chunks']} chunks")
    return "\n".join(lines)
