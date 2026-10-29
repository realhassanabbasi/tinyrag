# 🔍 TinyRAG

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen?style=flat-square)
![Deps](https://img.shields.io/badge/dependencies-zero-blue?style=flat-square)

**Ask questions over your own documents. No API key, no internet, no dependencies.**

TinyRAG is a minimal Retrieval-Augmented Generation pipeline built **from scratch** in pure Python: chunking, TF-IDF vectorization, cosine retrieval, and extractive answering — no sklearn, no numpy, no LLM. Index a folder of `.txt`/`.md` files, ask questions, get answers quoted from your documents with source citations. When nothing matches, it says so honestly instead of hallucinating.

## ✨ Features

- **Zero-dependency RAG** — the whole pipeline (tokenizer, TF-IDF, retriever, answerer) written by hand
- **Extractive answers** — every answer is a real sentence from your documents, never generated text
- **Source citations** — each answer names the document, chunk, and retrieval confidence
- **Honest misses** — below-threshold queries get "I couldn't find anything" instead of a guess
- **Persistent index** — build once, save to JSON, ask anytime without re-indexing
- **IDF-weighted sentence picking** — rare terms (backpropagation) outrank common ones

## 🚀 Quickstart

```bash
pip install git+https://github.com/realhassanabbasi/tinyrag.git

# index your documents
tinyrag index ./my-notes

# ask questions
tinyrag ask "what is backpropagation"
```

Or run the instant demo (no install needed):

```bash
git clone https://github.com/realhassanabbasi/tinyrag.git
cd tinyrag
python demo.py
```

## 💡 Example

```
>>> tinyrag ask "how does RAG reduce hallucinations?"

A: RAG reduces hallucinations because answers are grounded in real documents.

   ↳ rag-explained.txt (chunk 0, confidence 0.29)

>>> tinyrag ask "what is the capital of france?"

A: I couldn't find anything relevant in your documents.

   (tip: index more documents or rephrase the question)
```

## 🏗️ Architecture

```
                    ┌──────────┐
  ./my-notes ──────▶│ chunker  │  sliding window, 400 words / 80 overlap
                    └────┬─────┘
                         ▼
                    ┌──────────┐
                    │vectorizer│  TF-IDF from scratch, L2-normalized,
                    └────┬─────┘  english stopwords removed
                         ▼
  .tinyrag/index.json (chunks + idf + vectors, persisted)
                         │
  "your question" ──▶ tokenize ──▶ vectorize ──▶ ┌───────────┐
                                                 │ retriever │ cosine top-k
                                                 └─────┬─────┘
                                                       ▼
                                                 ┌───────────┐
                                                 │ answerer  │ best sentence by
                                                 └───────────┘ IDF-weighted term
                                                               overlap + citation
```

**Design decisions:**
- **TF-IDF instead of embeddings** — runs anywhere, no model downloads, no API; the honest trade-off is keyword matching instead of semantic understanding.
- **Extractive instead of generative** — without an LLM there is nothing to generate with; quoting real sentences guarantees zero hallucination by construction.
- **IDF-weighted sentence scoring** — plain term-overlap lets two common words beat one rare, decisive term; weighting by IDF fixes that.
- **Stopword removal** — "what is the" matches every document; dropping ~100 common words is standard IR practice and cut false positives to zero in testing.

## ⚠️ Honest limitations

- **English-only tokenizer** (`[a-z]+`) — other languages won't tokenize.
- **Keyword matching, not semantics** — "car" won't match "automobile"; no synonyms, no stemming.
- **Single-term queries on long docs score low** — TF-IDF cosine dilutes one rare term across a long document (measured: 0.09 on the demo corpus vs 0.29–0.30 for multi-term queries).
- **Naive sentence splitter** — abbreviations like "e.g." can split sentences early.
- **Small-scale design** — whole index lives in memory as JSON; fine for notes and docs, not for millions of pages.

## 🧪 Tests

```bash
python -m unittest discover -s tests   # 15 tests, all passing
```

Covers: chunking boundaries and overlap, TF-IDF normalization and IDF weighting, cosine sanity, retrieval ranking, IDF-weighted answer picking, honest-miss behavior, index save/load roundtrip, and a full end-to-end run over the sample docs.

## 📁 Project structure

```
tinyrag/
├── tinyrag/
│   ├── cli.py          # index / ask commands
│   ├── chunker.py      # sliding-window chunking
│   ├── vectorizer.py   # tokenizer + TF-IDF + cosine, from scratch
│   ├── retriever.py    # top-k cosine ranking
│   ├── answerer.py     # extractive answering with citations
│   ├── store.py        # index save/load (JSON)
│   └── data/sample_docs/  # 3 demo documents
├── tests/test_tinyrag.py
├── demo.py             # zero-arg end-to-end demo
└── pyproject.toml
```

## 👤 Author

**Hassan Abbasi** — AI Engineer building AI-powered products with excellent user experiences.
[GitHub](https://github.com/realhassanabbasi)
