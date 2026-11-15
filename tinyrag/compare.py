"""Compare two indexes on the same question set — what did the docs change?

`check` re-answers a saved question set on ONE index and flags answers
that moved to a different document. It never shows the old answer next
to the new one, and it cannot compare two indexes at all. Compare fills
that gap: point it at an old index and a new one (before and after a
docs update, or two versions of the same notes) and it answers every
question on both, then reports what changed — same, changed text, moved
to another document, newly answered, lost, or still unanswered.

Each row is the answer `ask` would print for the same question on that
index: same retriever, same answerer, same confidence — nothing
re-implemented, so a compare row can never disagree with a one-off `ask`
on either side. A question with no match on a side keeps the honest
miss text, never a guess.
"""

from __future__ import annotations

from .ask_batch import ask_one

DEFAULT_K = 3

# Status labels, in the order the summary counts them.
STATUSES = ("same", "changed", "moved", "new-answer", "lost-answer",
            "still-missed")


def _status(a: dict, b: dict) -> str:
    """Classify one question's pair of answers."""
    if not a["answered"] and not b["answered"]:
        return "still-missed"
    if not a["answered"]:
        return "new-answer"
    if not b["answered"]:
        return "lost-answer"
    if a["doc"] != b["doc"]:
        return "moved"
    if a["answer"] != b["answer"]:
        return "changed"
    return "same"


def compare_payload(index_a: dict, index_b: dict, questions: list[dict],
                    k: int = DEFAULT_K) -> dict:
    """Answer every question on both indexes, in order.

    Payload: questions (compare rows), total, and one count per status.
    `questions` is the list `check.load_questions` returns, so a `check`
    file doubles as a compare file — no second format to learn.
    """
    rows = []
    for q in questions:
        a = ask_one(index_a, q["question"], k=k)
        b = ask_one(index_b, q["question"], k=k)
        rows.append({
            "question": q["question"],
            "status": _status(a, b),
            "answer_a": a["answer"],
            "answer_b": b["answer"],
            "doc_a": a["doc"],
            "doc_b": b["doc"],
            "confidence_a": a["confidence"],
            "confidence_b": b["confidence"],
        })
    payload = {"questions": rows, "total": len(rows)}
    for status in STATUSES:
        payload[status.replace("-", "_")] = sum(
            1 for row in rows if row["status"] == status)
    return payload


def _summary(payload: dict) -> str:
    parts = []
    for status in STATUSES:
        n = payload[status.replace("-", "_")]
        if n:
            parts.append(f"{n} {status}")
    detail = ", ".join(parts) if parts else "no questions"
    return (f"{payload['total']} question"
            f"{'s' if payload['total'] != 1 else ''} · {detail}")


def format_compare(payload: dict) -> str:
    """Plain-text view: one numbered Q block per question, A then B."""
    if not payload["questions"]:
        return "No questions to compare — the question set is empty."
    lines = [_summary(payload) + ":", ""]
    for i, row in enumerate(payload["questions"], start=1):
        lines.append(f"{i}. {row['question']} [{row['status']}]")
        lines.append(f"   A: {row['answer_a']}")
        if row["doc_a"]:
            lines.append(f"      ↳ {row['doc_a']} "
                         f"(confidence {row['confidence_a']})")
        lines.append(f"   B: {row['answer_b']}")
        if row["doc_b"]:
            lines.append(f"      ↳ {row['doc_b']} "
                         f"(confidence {row['confidence_b']})")
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def compare_markdown(payload: dict) -> str:
    """Markdown report: one section per question, both answers side by side.

    Misses get an honest "No answer found" line instead of a section
    left blank — the report never hides a gap.
    """
    lines = ["# TinyRAG index comparison", "", _summary(payload) + ".", ""]
    for i, row in enumerate(payload["questions"], start=1):
        lines.append(f"## {i}. {row['question']} ({row['status']})")
        lines.append("")
        lines.append(f"**Before:** {row['answer_a']}")
        if row["doc_a"]:
            lines.append(f"*Source: {row['doc_a']} "
                         f"(confidence {row['confidence_a']})*")
        else:
            lines.append("*No answer found in the first index.*")
        lines.append("")
        lines.append(f"**After:** {row['answer_b']}")
        if row["doc_b"]:
            lines.append(f"*Source: {row['doc_b']} "
                         f"(confidence {row['confidence_b']})*")
        else:
            lines.append("*No answer found in the second index.*")
        lines.append("")
    return "\n".join(lines)
