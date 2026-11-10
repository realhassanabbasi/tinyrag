"""Regression check: re-answer a saved question set, flag what moved.

A question set is a JSON file:
    {"questions": [{"question": "what is backpropagation?",
                    "expect": "deep-learning.txt"}]}

``expect`` is the document the question should be answered from.
``tinyrag check FILE --record`` fills in each ``expect`` from the current
index (the baseline); a plain ``tinyrag check FILE`` re-answers every
question and reports steady / changed / unrecorded per question. Run it
after ``refresh``: if a document edit moved an answer to a different
document, the check says so instead of silently changing what users get.

Nothing here re-scores beyond what ``ask`` already does — the answerer
is the single source of truth, so a check result always agrees with
what ``ask`` would print for the same question.
"""

from __future__ import annotations

import json
import os

from .answerer import answer
from .retriever import retrieve
from .vectorizer import tokenize, vectorize


def _top_answer(index: dict, question: str) -> dict:
    """The same pipeline ``ask`` runs: tokenize, vectorize, retrieve, answer."""
    q_vec = vectorize(tokenize(question), index["idf"])
    ranked = retrieve(q_vec, index["vectors"], k=3)
    return answer(set(tokenize(question)), index["idf"], ranked, index["chunks"])


def load_questions(path: str) -> list[dict]:
    """Read a question set. Refuses bad files outright — no half-checks.

    Accepts {"questions": [...]} or a bare [...] list. Every entry must
    be an object with a non-empty "question" string; "expect" is optional
    (None means unrecorded). Raises FileNotFoundError for a missing file
    and ValueError for malformed JSON or a bad shape.
    """
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if isinstance(data, dict):
        entries = data.get("questions")
    else:
        entries = data
    if not isinstance(entries, list):
        raise ValueError(
            f"{path} is not a question set — expected "
            '{"questions": [{"question": "...", "expect": "..."}]}'
        )
    questions = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("question"):
            raise ValueError(
                f"{path} entry {i} is not a question — expected "
                '{"question": "...", "expect": "..."}'
            )
        questions.append({
            "question": str(entry["question"]),
            "expect": entry.get("expect"),
        })
    return questions


def save_questions(path: str, questions: list[dict]) -> None:
    """Write a question set back to disk (the --record round-trip)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f, indent=2)
        f.write("\n")


def record_expectations(index: dict, questions: list[dict]) -> list[dict]:
    """Fill in each question's expect from the current index.

    Returns a new list — the input is untouched. A question with no
    matching answer records expect None (an honest miss, not a guess).
    """
    recorded = []
    for q in questions:
        result = _top_answer(index, q["question"])
        recorded.append({"question": q["question"], "expect": result["doc"]})
    return recorded


def check_questions(index: dict, questions: list[dict]) -> list[dict]:
    """Re-answer every question; verdict per question.

    steady: the top document still matches the recorded expectation.
    changed: the expectation exists but the top document moved — or
    there is no answer at all now (a miss is a change, not a pass).
    unrecorded: no expectation was recorded, so there is nothing to
    compare against; never counted as a failure.
    """
    results = []
    for q in questions:
        result = _top_answer(index, q["question"])
        now = result["doc"]
        expect = q.get("expect")
        if expect is None:
            verdict = "unrecorded"
        elif now == expect:
            verdict = "steady"
        else:
            verdict = "changed"
        results.append({
            "question": q["question"],
            "expect": expect,
            "now": now,
            "confidence": result["confidence"],
            "verdict": verdict,
        })
    return results


def check_payload(index: dict, questions: list[dict]) -> dict:
    """Machine-readable check: results plus steady/changed/unrecorded counts."""
    results = check_questions(index, questions)
    return {
        "results": results,
        "checked": len(results),
        "steady": sum(1 for r in results if r["verdict"] == "steady"),
        "changed": sum(1 for r in results if r["verdict"] == "changed"),
        "unrecorded": sum(1 for r in results if r["verdict"] == "unrecorded"),
    }


def format_check(payload: dict) -> str:
    """Plain-text view of a check payload."""
    if not payload["results"]:
        return ("no questions in the set — add some, then run "
                "`tinyrag check FILE --record` to record expectations")
    lines = []
    for row in payload["results"]:
        question = row["question"]
        if row["verdict"] == "steady":
            lines.append(
                f"  ✓ {question} — {row['now']} ({row['confidence']})"
            )
        elif row["verdict"] == "changed":
            now = row["now"] if row["now"] is not None else "no answer"
            lines.append(
                f"  ✗ {question} — expected {row['expect']}, "
                f"now {now} ({row['confidence']})"
            )
        else:
            now = row["now"] if row["now"] is not None else "no answer"
            lines.append(
                f"  ? {question} — no expectation recorded "
                f"(top answer: {now}, {row['confidence']})"
            )
    summary = (f"{payload['checked']} checked: "
               f"{payload['steady']} steady, {payload['changed']} changed")
    if payload["unrecorded"]:
        summary += f", {payload['unrecorded']} unrecorded"
    lines.append("")
    lines.append(summary)
    return "\n".join(lines)
