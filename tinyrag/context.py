"""Context budget: fit the retrieved chunks into a character budget.

Real RAG pipelines hand the model a context window, not the whole index.
TinyRAG's answerer reads the top-k chunks in full — with big documents
that context is unbounded. `fit_context` caps it: walk the ranked chunks
best-first, keep whole chunks while they fit, trim a too-big chunk down to
its best sentences (scored the same way the answerer scores them), and
skip a chunk entirely when not even one sentence fits. Everything that was
cut is reported, never hidden.
"""

from __future__ import annotations

from .answerer import split_sentences
from .vectorizer import tokenize


def score_sentence(sent: str, query_terms: set[str],
                   idf: dict[str, float]) -> float:
    """IDF weight of the query terms a sentence contains.

    Same weighting the answerer uses, so the sentences kept here are the
    ones the answerer would have picked from anyway.
    """
    return sum(idf.get(t, 0.0) for t in set(tokenize(sent))
               if t in query_terms)


def _trim_to_budget(text: str, budget: int, query_terms: set[str],
                    idf: dict[str, float]) -> str:
    """Keep the best sentences of `text` that fit in `budget` chars.

    Sentences are picked best-first by query-term weight, then reassembled
    in their original order so the trimmed chunk still reads in order.
    Returns "" when not even the best sentence fits.
    """
    sentences = split_sentences(text)
    scored = sorted(
        ((score_sentence(s, query_terms, idf), i, s)
         for i, s in enumerate(sentences)),
        key=lambda x: (-x[0], x[1]),
    )
    picked: list[tuple[int, str]] = []
    used = 0
    for weight, i, sent in scored:
        if weight <= 0:
            continue  # a sentence with no query terms adds no evidence
        sep = 1 if picked else 0  # one space joins sentences
        if used + sep + len(sent) <= budget:
            picked.append((i, sent))
            used += sep + len(sent)
    picked.sort(key=lambda x: x[0])
    return " ".join(s for _, s in picked)


def fit_context(
    ranked: list[tuple[int, float]],
    chunks: list[dict],
    max_chars: int,
    query_terms: set[str],
    idf: dict[str, float],
) -> dict:
    """Fit the ranked chunks into `max_chars` characters.

    Returns a report dict: budget, chars_in (kept), chars_out (ranked
    total), kept / trimmed / dropped counts, truncated flag, and `entries`
    — one per kept chunk, in rank order, each with the (possibly trimmed)
    chunk dict, its retrieval score, and a trimmed flag. Zero-score ranked
    entries are ignored: they matched nothing, so they are not context.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    entries: list[dict] = []
    chars_in = 0
    chars_out = 0
    n_trimmed = 0
    n_dropped = 0
    for idx, score in ranked:
        if score <= 0.0:
            continue
        chunk = chunks[idx]
        text = chunk["text"]
        chars_out += len(text)
        room = max_chars - chars_in
        if len(text) <= room:
            entries.append({"chunk": chunk, "score": score,
                            "trimmed": False})
            chars_in += len(text)
            continue
        trimmed = _trim_to_budget(text, room, query_terms, idf)
        if trimmed:
            fitted = dict(chunk)
            fitted["text"] = trimmed
            entries.append({"chunk": fitted, "score": score,
                            "trimmed": True})
            chars_in += len(trimmed)
            n_trimmed += 1
        else:
            n_dropped += 1
    return {
        "budget": max_chars,
        "chars_in": chars_in,
        "chars_out": chars_out,
        "kept": len(entries),
        "trimmed": n_trimmed,
        "dropped": n_dropped,
        "truncated": n_trimmed > 0 or n_dropped > 0,
        "entries": entries,
    }


def context_payload(fit: dict) -> dict:
    """JSON envelope for a fit: budget, counts, and the fitted chunks."""
    return {
        "budget": fit["budget"],
        "chars_in": fit["chars_in"],
        "chars_out": fit["chars_out"],
        "kept": fit["kept"],
        "trimmed": fit["trimmed"],
        "dropped": fit["dropped"],
        "truncated": fit["truncated"],
        "chunks": [
            {
                "doc": e["chunk"]["doc"],
                "chunk_id": e["chunk"]["chunk_id"],
                "score": round(e["score"], 2),
                "trimmed": e["trimmed"],
                "chars": len(e["chunk"]["text"]),
                "text": e["chunk"]["text"],
            }
            for e in fit["entries"]
        ],
    }


def format_context(fit: dict) -> str:
    """Text view: the exact context the answer was drawn from."""
    lines = [
        f"Context: {fit['chars_in']} / {fit['budget']} chars "
        f"({fit['kept']} kept, {fit['trimmed']} trimmed, "
        f"{fit['dropped']} dropped)"
    ]
    for entry in fit["entries"]:
        chunk = entry["chunk"]
        tag = " (trimmed)" if entry["trimmed"] else ""
        lines.append(f"  [{chunk['doc']} chunk {chunk['chunk_id']}, "
                     f"{entry['score']:.2f}]{tag}")
        lines.append(f"    {chunk['text']}")
    if not fit["entries"]:
        lines.append("  (nothing fit in the budget)")
    return "\n".join(lines)
