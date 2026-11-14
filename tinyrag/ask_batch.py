"""Answer a whole question set in one run — and keep the answers.

`ask` answers one question at a time; `check` re-answers a saved set
only to say whether anything moved. Neither gives you the answers
themselves as a document. Ask-batch fills that gap: it takes the same
question-set file `check` reads (``{"questions": [{"question": ...}]}``),
runs every question through the exact pipeline `ask` runs, and returns
the answers — on screen, as JSON for scripts, or as a Markdown report
with `-o` (an FAQ page you can hand to someone, generated from your
own documents, with a source line under every answer).

Each row is the answer `ask` would print for the same question: same
retriever, same answerer, same confidence — nothing re-implemented,
so a batch answer can never disagree with a one-off `ask`. A question
with no match gets the honest miss text, never a guess.
"""

from __future__ import annotations

from .answerer import answer
from .retriever import retrieve
from .vectorizer import tokenize, vectorize

DEFAULT_K = 3


def ask_one(index: dict, question: str, k: int = DEFAULT_K) -> dict:
    """One question through the exact pipeline `ask` runs.

    Returns an answer row: question, answer, doc, chunk_id,
    confidence, answered. A miss keeps the honest miss text from
    the answerer, with doc None and answered False.
    """
    q_terms = tokenize(question)
    q_vec = vectorize(q_terms, index.get("idf", {}))
    ranked = retrieve(q_vec, index.get("vectors", []), k=k)
    result = answer(set(q_terms), index.get("idf", {}), ranked,
                    index.get("chunks", []))
    return {
        "question": question,
        "answer": result["text"],
        "doc": result["doc"],
        "chunk_id": result["chunk_id"],
        "confidence": result["confidence"],
        "answered": result["doc"] is not None,
    }


def ask_batch_payload(index: dict, questions: list[dict],
                      k: int = DEFAULT_K) -> dict:
    """Answer every question in the set, in order.

    Payload: questions (answer rows), total, answered, missed.
    `questions` is the list `check.load_questions` returns, so a
    `check` file doubles as a batch file — no second format to learn.
    """
    rows = [ask_one(index, q["question"], k=k) for q in questions]
    answered = sum(1 for row in rows if row["answered"])
    return {
        "questions": rows,
        "total": len(rows),
        "answered": answered,
        "missed": len(rows) - answered,
    }


def _summary(payload: dict) -> str:
    return (f"{payload['total']} question"
            f"{'s' if payload['total'] != 1 else ''} · "
            f"{payload['answered']} answered, "
            f"{payload['missed']} missed")


def format_ask_batch(payload: dict) -> str:
    """Plain-text view: one numbered Q/A block per question."""
    if not payload["questions"]:
        return "No questions to answer — the question set is empty."
    lines = [_summary(payload) + ":", ""]
    for i, row in enumerate(payload["questions"], start=1):
        lines.append(f"{i}. {row['question']}")
        lines.append(f"   A: {row['answer']}")
        if row["doc"]:
            lines.append(f"   ↳ {row['doc']} (chunk {row['chunk_id']}, "
                         f"confidence {row['confidence']})")
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def ask_batch_markdown(payload: dict) -> str:
    """Markdown report: one section per question, source under each answer.

    Misses get an honest \"No answer found\" line instead of a section
    left blank — the report never hides a gap.
    """
    lines = ["# TinyRAG answers", "", _summary(payload) + ".", ""]
    for i, row in enumerate(payload["questions"], start=1):
        lines.append(f"## {i}. {row['question']}")
        lines.append("")
        lines.append(row["answer"])
        lines.append("")
        if row["doc"]:
            lines.append(f"*Source: {row['doc']} (chunk {row['chunk_id']}, "
                         f"confidence {row['confidence']})*")
        else:
            lines.append("*No answer found in the indexed documents.*")
        lines.append("")
    return "\n".join(lines)
