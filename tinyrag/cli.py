"""Command line: index a folder, ask questions, run the demo."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .answerer import answer
from .chunker import chunk_folder
from .retriever import retrieve
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
    save_index(build_index(records, idf, vectors), index_path)
    n_docs = len({r["doc"] for r in records})
    print(f"indexed {n_docs} documents → {len(records)} chunks → {index_path}")


def cmd_ask(question: str, index_path: str = DEFAULT_INDEX, k: int = 3) -> None:
    if not os.path.exists(index_path):
        print("no index found — run `tinyrag index <folder>` first")
        return
    index = load_index(index_path)
    q_terms = tokenize(question)
    q_vec = vectorize(q_terms, index["idf"])
    ranked = retrieve(q_vec, index["vectors"], k=k)
    result = answer(set(q_terms), index["idf"], ranked, index["chunks"])
    print(f"\nQ: {question}\n")
    print(f"A: {result['text']}")
    if result["doc"]:
        print(f"\n   ↳ {result['doc']} (chunk {result['chunk_id']}, "
              f"confidence {result['confidence']})")
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

    args = p.parse_args(argv)
    if args.cmd == "index":
        cmd_index(args.folder, args.out)
    elif args.cmd == "ask":
        cmd_ask(args.question, args.index, args.k)


if __name__ == "__main__":
    main()
