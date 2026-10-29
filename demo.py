"""Zero-arg end-to-end demo: index the sample docs, ask real questions."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinyrag.answerer import answer
from tinyrag.chunker import chunk_folder
from tinyrag.retriever import retrieve
from tinyrag.vectorizer import build_idf, tokenize, vectorize

QUESTIONS = [
    "What is backpropagation?",
    "How does RAG reduce hallucinations?",
    "Which Python library is used for numerical computing?",
    "What is the capital of France?",  # not in the docs — honest miss
]

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "tinyrag", "data", "sample_docs")


def main() -> None:
    print("=" * 64)
    print("  tinyrag demo — your documents, your answers, no API key")
    print("=" * 64)
    records = chunk_folder(DOCS)
    toks = [tokenize(r["text"]) for r in records]
    idf = build_idf(toks)
    vecs = [vectorize(t, idf) for t in toks]
    print(f"\nindexed {len({r['doc'] for r in records})} docs "
          f"→ {len(records)} chunks\n")
    for q in QUESTIONS:
        qv = vectorize(tokenize(q), idf)
        ranked = retrieve(qv, vecs, k=3)
        res = answer(set(tokenize(q)), idf, ranked, records)
        print(f">>> {q}")
        print(f"    {res['text']}")
        if res["doc"]:
            print(f"    ↳ {res['doc']} (confidence {res['confidence']})\n")
        else:
            print("    ↳ no match — answered honestly\n")


if __name__ == "__main__":
    main()
