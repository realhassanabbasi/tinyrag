"""Command line: index a folder, ask questions, run the demo."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .answerer import answer
from .ask_batch import (ask_batch_markdown, ask_batch_payload,
                        format_ask_batch)
from .check import (check_payload, format_check, load_questions,
                    record_expectations, save_questions)
from .chunker import chunk_folder
from .evidence import evidence_payload, format_evidence
from .refresh import format_refresh, refresh_index, scan_folder
from .retriever import retrieve
from .search import format_search, search_payload
from .stats import format_stats, stats_payload
from .store import build_index, load_index, save_index
from .vectorizer import build_idf, tokenize, vectorize

DEFAULT_INDEX = os.path.join(".tinyrag", "index.json")


def cmd_index(folder: str, index_path: str = DEFAULT_INDEX) -> None:
    records = chunk_folder(folder)
    if not records:
        print(f"no .txt/.md files found in {folder}")
        return
    docs_tokens = [tokenize(r["text"]) for r in records]
    idf = build_idf(docs_tokens)
    vectors = [vectorize(t, idf) for t in docs_tokens]
    save_index(build_index(records, idf, vectors,
                           meta={"docs": scan_folder(folder)}), index_path)
    n_docs = len({r["doc"] for r in records})
    print(f"indexed {n_docs} documents → {len(records)} chunks → {index_path}")


def cmd_refresh(folder: str, index_path: str = DEFAULT_INDEX,
                as_json: bool = False) -> None:
    if not os.path.isdir(folder):
        print(f"no such folder: {folder}")
        return
    try:
        report = refresh_index(folder, index_path)
    except FileNotFoundError:
        print("no index found — run `tinyrag index <folder>` first")
        return
    if as_json:
        print(json.dumps(report, indent=2))
        return
    print(format_refresh(report))


def cmd_stats(index_path: str = DEFAULT_INDEX, as_json: bool = False) -> None:
    if not os.path.exists(index_path):
        print("no index found — run `tinyrag index <folder>` first")
        return
    payload = stats_payload(load_index(index_path))
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    print(format_stats(payload))


def cmd_check(questions_path: str, index_path: str = DEFAULT_INDEX,
              as_json: bool = False, record: bool = False,
              fail_on_change: bool = False) -> None:
    if not os.path.exists(questions_path):
        print(f"no such file: {questions_path}")
        sys.exit(1)
    try:
        questions = load_questions(questions_path)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)
    if not os.path.exists(index_path):
        print("no index found — run `tinyrag index <folder>` first")
        return
    index = load_index(index_path)
    if record:
        recorded = record_expectations(index, questions)
        save_questions(questions_path, recorded)
        n = sum(1 for q in recorded if q["expect"] is not None)
        print(f"recorded {n} expectation(s) in {questions_path}")
        for q in recorded:
            expect = q["expect"] if q["expect"] is not None else "no answer"
            print(f"  {q['question']} → {expect}")
        return
    payload = check_payload(index, questions)
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_check(payload))
    if fail_on_change and payload["changed"] > 0:
        sys.exit(1)


def cmd_ask_batch(questions_path: str, index_path: str = DEFAULT_INDEX,
                  k: int = 3, out_path: str | None = None,
                  as_json: bool = False) -> None:
    if not os.path.exists(questions_path):
        print(f"no such file: {questions_path}")
        sys.exit(1)
    try:
        questions = load_questions(questions_path)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)
    if not os.path.exists(index_path):
        print("no index found — run `tinyrag index <folder>` first")
        return
    payload = ask_batch_payload(load_index(index_path), questions, k=k)
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(ask_batch_markdown(payload))
        print(f"wrote {payload['answered']} answer"
              f"{'s' if payload['answered'] != 1 else ''} "
              f"({payload['missed']} missed) to {out_path}")
        return
    print(format_ask_batch(payload))


def cmd_search(query: str, index_path: str = DEFAULT_INDEX, k: int = 5,
               as_json: bool = False) -> None:
    if not os.path.exists(index_path):
        print("no index found — run `tinyrag index <folder>` first")
        return
    payload = search_payload(load_index(index_path), query, k=k)
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    print(format_search(payload))


def cmd_ask(question: str, index_path: str = DEFAULT_INDEX, k: int = 3,
            sources: int = 0, as_json: bool = False) -> None:
    if not os.path.exists(index_path):
        print("no index found — run `tinyrag index <folder>` first")
        return
    index = load_index(index_path)
    q_terms = tokenize(question)
    q_vec = vectorize(q_terms, index["idf"])
    ranked = retrieve(q_vec, index["vectors"], k=k)
    result = answer(set(q_terms), index["idf"], ranked, index["chunks"])
    rows = evidence_payload(ranked, index["chunks"], n=k)
    if as_json:
        payload = {
            "question": question,
            "answer": result["text"],
            "doc": result["doc"],
            "chunk_id": result["chunk_id"],
            "confidence": result["confidence"],
            "sources": rows,
        }
        print(json.dumps(payload, indent=2))
        return
    print(f"\nQ: {question}\n")
    print(f"A: {result['text']}")
    if result["doc"]:
        print(f"\n   ↳ {result['doc']} (chunk {result['chunk_id']}, "
              f"confidence {result['confidence']})")
        if sources > 0 and rows:
            print(f"\n{format_evidence(rows[:sources])}")
    else:
        print("\n   (tip: index more documents or rephrase the question)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tinyrag",
                                description="Ask questions over your own documents.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="index a folder of .txt/.md files")
    pi.add_argument("folder", help="folder with documents")
    pi.add_argument("--out", default=DEFAULT_INDEX, help="index file path")

    pa = sub.add_parser("ask", help="ask a question over the index")
    pa.add_argument("question", help="your question in quotes")
    pa.add_argument("--index", default=DEFAULT_INDEX, help="index file path")
    pa.add_argument("-k", type=int, default=3, help="chunks to consider")
    pa.add_argument("--sources", type=int, default=0,
                    help="show the top N retrieved chunks as evidence")
    pa.add_argument("--json", action="store_true",
                    help="print the answer + sources as JSON")

    pr = sub.add_parser("refresh", help="re-index only files that changed")
    pr.add_argument("folder", help="folder with documents")
    pr.add_argument("--index", default=DEFAULT_INDEX, help="index file path")
    pr.add_argument("--json", action="store_true",
                    help="print the refresh report as JSON")

    ps = sub.add_parser("stats", help="show what is in the index")
    ps.add_argument("--index", default=DEFAULT_INDEX, help="index file path")
    ps.add_argument("--json", action="store_true",
                    help="print the stats as JSON")

    ps = sub.add_parser("search", help="find the chunks that mention a "
                        "word or phrase")
    ps.add_argument("query", help="words to look for, in quotes")
    ps.add_argument("--index", default=DEFAULT_INDEX, help="index file path")
    ps.add_argument("-k", type=int, default=5,
                    help="matches to show (default 5)")
    ps.add_argument("--json", action="store_true",
                    help="print the matches as JSON")

    pc = sub.add_parser("ask-batch", help="answer a whole question set "
                        "in one run, optionally as a Markdown report")
    pc.add_argument("questions", help="JSON question-set file "
                    "(same format as check)")
    pc.add_argument("--index", default=DEFAULT_INDEX, help="index file path")
    pc.add_argument("-k", type=int, default=3,
                    help="chunks to consider per question")
    pc.add_argument("-o", default=None, metavar="FILE",
                    help="write the answers as a Markdown report")
    pc.add_argument("--json", action="store_true",
                    help="print the answers as JSON")

    pc = sub.add_parser("check", help="re-answer a saved question set, "
                        "flag answers that moved")
    pc.add_argument("questions", help="JSON question-set file")
    pc.add_argument("--index", default=DEFAULT_INDEX, help="index file path")
    pc.add_argument("--json", action="store_true",
                    help="print the check report as JSON")
    pc.add_argument("--record", action="store_true",
                    help="record the current top document as the "
                    "expectation for each question")
    pc.add_argument("--fail-on-change", action="store_true",
                    help="exit 1 if any answer changed (for CI)")

    args = p.parse_args(argv)
    if args.cmd == "index":
        cmd_index(args.folder, args.out)
    elif args.cmd == "search":
        cmd_search(args.query, args.index, args.k, args.json)
    elif args.cmd == "ask":
        cmd_ask(args.question, args.index, args.k, args.sources, args.json)
    elif args.cmd == "refresh":
        cmd_refresh(args.folder, args.index, args.json)
    elif args.cmd == "stats":
        cmd_stats(args.index, args.json)
    elif args.cmd == "check":
        cmd_check(args.questions, args.index, args.json, args.record,
                  args.fail_on_change)
    elif args.cmd == "ask-batch":
        cmd_ask_batch(args.questions, args.index, args.k, args.o,
                      args.json)


if __name__ == "__main__":
    main()
