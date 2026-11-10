# 🔍 TinyRAG

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Tests](https://img.shields.io/badge/tests-50%20passing-brightgreen?style=flat-square)
![Deps](https://img.shields.io/badge/dependencies-zero-blue?style=flat-square)

**Ask questions over your own documents. No API key, no internet, no dependencies.**

TinyRAG is a minimal Retrieval-Augmented Generation pipeline built **from scratch** in pure Python: chunking, TF-IDF vectorization, cosine retrieval, and extractive answering — no sklearn, no numpy, no LLM. Index a folder of `.txt`/`.md` files, ask questions, get answers quoted from your documents with source citations. When nothing matches, it says so honestly instead of hallucinating.

## ✨ Features

- **Zero-dependency RAG** — the whole pipeline (tokenizer, TF-IDF, retriever, answerer) written by hand
- **Extractive answers** — every answer is a real sentence from your documents, never generated text
- **Source citations** — each answer names the document, chunk, and retrieval confidence
- **Honest misses** — below-threshold queries get "I couldn't find anything" instead of a guess
- **Persistent index** — build once, save to JSON, ask anytime without re-indexing
- **Evidence view** — `ask --sources N` shows the ranked chunks behind an answer (doc, chunk, score, excerpt); `--json` prints the answer plus sources for scripts
- **Incremental refresh** — `refresh` re-indexes only added/edited files, drops deleted ones, and leaves unchanged files alone (signature = mtime + size)
- **Index stats** — `stats` shows documents, chunks, vocabulary size, per-document word counts, and the most common terms, read straight from the saved index; `--json` prints the same numbers for scripts
- **Regression check** — `check` re-answers a saved question set after a refresh and flags answers that moved to a different document (or vanished); `--record` builds the baseline, `--fail-on-change` exits 1 for CI
- **IDF-weighted sentence picking** — rare terms (backpropagation) outrank common ones

## 🚀 Quickstart

```bash
pip install git+https://github.com/realhassanabbasi/tinyrag.git

# index your documents
tinyrag index ./my-notes

# ask questions
tinyrag ask "what is backpropagation"

# add a document later, then refresh instead of re-indexing everything
tinyrag refresh ./my-notes

# see what is actually in the index
tinyrag stats

# guard the answers: record a baseline, then check after every refresh
tinyrag check questions.json --record
tinyrag check questions.json
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

## 🔎 Show the evidence

An answer is one sentence, but the retriever scores several chunks — the
rest used to be thrown away. `--sources N` prints the ranked chunks behind
the answer so you can check it against what the index actually returned.
Chunks that scored 0.0 are never shown: they matched nothing.

```bash
tinyrag ask "python machine learning" --sources 3
```

```
Q: python machine learning

A: Python for Machine Learning Python is the most popular language for machine learning.

   ↳ python-ml.txt (chunk 0, confidence 0.41)

Evidence:
  1. python-ml.txt (chunk 0, 0.41) — "Python for Machine Learning Python is the most popular language for machine learning. Its ecosystem includes NumPy for numerical computing, pandas for data manipulation, and scikit-learn for classical…"
  2. neural-networks.txt (chunk 0, 0.11) — "Neural Networks Basics A neural network is a machine learning model inspired by the human brain. It consists of layers of interconnected nodes called neurons. Each connection has a weight that adjusts…"
```

For scripts, `--json` prints the answer plus the same sources list as pure
JSON (`question`, `answer`, `doc`, `chunk_id`, `confidence`, `sources`).
On an honest miss there is no Evidence block — nothing matched, so there
is nothing to show.

## 🔄 Refresh the index

An index is a snapshot — documents grow, get edited, get deleted.
`refresh` compares each file's signature (modification time + size)
against the signatures stored in the index and only re-chunks what
actually changed. Unchanged files keep their old chunks; deleted files
drop out. IDF is corpus-wide, so vectors are rebuilt over the full
record set every refresh — the saved work is the re-reading and
re-chunking, not the math.

```bash
tinyrag refresh ./my-notes
```

```
Added: 1 (deep-learning.txt)
Updated: 0
Removed: 0
Unchanged: 2 (neural-networks.txt, python-ml.txt)
index now: 3 documents, 3 chunks
```

The new document is searchable immediately — no full re-index needed:

```
>>> tinyrag ask "what is backpropagation"

A: Backpropagation computes gradients by applying the chain rule through every layer.

   ↳ deep-learning.txt (chunk 0, confidence 0.2)
```

A second refresh with nothing changed is a no-op that says so
(`Unchanged: 3`, `Added: 0`). An index built by v1.1.0 has no stored
signatures, so the first refresh after upgrading rebuilds all docs once
and says so honestly (`note: old index had no file signatures`). For
scripts, `--json` prints the report (`added`, `updated`, `removed`,
`unchanged`, `docs`, `chunks`, `legacy`) as pure JSON.

## 📊 Index stats

After indexing, `stats` answers the obvious question: what is actually
in this index? Everything is read from the saved index file — nothing
is re-indexed or re-scored. Document counts come from the chunk
records, vocabulary size from the stored IDF table, and the most common
terms are counted from the chunk texts (a term counts once per chunk it
appears in, so a word repeated fifty times in one chunk still counts
once).

```bash
tinyrag stats
```

```
Documents: 3
Chunks: 3
Vocabulary: 229 terms
Words: 438 (avg 146.0 per chunk)
Largest document: rag-explained.txt (1 chunk, 167 words)

Per document:
  neural-networks.txt — 1 chunk, 124 words
  python-ml.txt — 1 chunk, 147 words
  rag-explained.txt — 1 chunk, 167 words

Most common terms (by chunks they appear in):
  data — in 3 chunks
  model — in 3 chunks
  accuracy — in 2 chunks
  also — in 2 chunks
  answer — in 2 chunks
  api — in 2 chunks
  deep — in 2 chunks
  during — in 2 chunks
  language — in 2 chunks
  learn — in 2 chunks
```

The output above is the real output on the bundled sample docs. For
scripts, `--json` prints the same payload (`documents`, `chunks`,
`terms`, `total_words`, `avg_chunk_words`, `docs`, `largest_doc`,
`top_terms`, `has_data`) as pure JSON. With no index file it says
`no index found — run tinyrag index <folder> first`; an empty index
says so too instead of printing zeros as if they meant something.

## ✅ Regression check

`stats` shows what is in the index; `check` shows whether the index
still answers the same way. A question set is a JSON file of questions
plus the document each one should be answered from:

```json
{
  "questions": [
    {"question": "what is backpropagation?", "expect": null},
    {"question": "what is python used for in machine learning?", "expect": null},
    {"question": "how does RAG reduce hallucinations?", "expect": null}
  ]
}
```

`--record` fills in each `expect` from the current index — the
baseline. A plain `check` re-answers every question against the
current index and reports steady / changed / unrecorded per question:

```bash
tinyrag check questions.json --record
```

```
recorded 3 expectation(s) in questions.json
  what is backpropagation? → neural-networks.txt
  what is python used for in machine learning? → python-ml.txt
  how does RAG reduce hallucinations? → rag-explained.txt
```

```bash
tinyrag check questions.json
```

```
  ✓ what is backpropagation? — neural-networks.txt (0.09)
  ✓ what is python used for in machine learning? — python-ml.txt (0.41)
  ✓ how does RAG reduce hallucinations? — rag-explained.txt (0.29)

3 checked: 3 steady, 0 changed
```

The point is what happens after a `refresh`. Edit a document, refresh,
check again — an answer that moved (or vanished) is flagged, not
silently changed:

```
  ✗ what is alpha beta — expected a.txt, now no answer (0.0)
  ✓ what is delta epsilon — b.txt (0.82)

2 checked: 1 steady, 1 changed
```

A question with no recorded expectation reports `?` and is never
counted as a failure — the check only judges what was actually
recorded. For scripts, `--json` prints the report (`results`,
`checked`, `steady`, `changed`, `unrecorded`) as pure JSON, and
`--fail-on-change` exits 1 when anything changed, so the check plugs
straight into CI. A missing question file or malformed JSON is refused
outright (exit 1) instead of running a half-check.

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
python -m unittest discover -s tests   # 50 tests, all passing
```

Covers: chunking boundaries and overlap, TF-IDF normalization and IDF weighting, cosine sanity, retrieval ranking, IDF-weighted answer picking, honest-miss behavior, index save/load roundtrip, evidence rows (shape, zero-score drop, excerpt truncation, --sources/--json CLI, miss shows no evidence), refresh (no-change no-op, edited/added/deleted files, legacy-index rebuild, missing index/folder, --json report), stats (aggregates, per-doc rows, top terms by chunk count, empty index, largest doc, text/--json CLI, missing index), check (record fills expectations, steady on no change, changed after edit+refresh incl. honest miss, unrecorded never a failure, malformed JSON refused, text/--json CLI, --fail-on-change exit codes, missing file exits 1), and a full end-to-end run over the sample docs.

## 📁 Project structure

```
tinyrag/
├── tinyrag/
│   ├── cli.py          # index / ask / refresh / stats / check commands
│   ├── chunker.py      # sliding-window chunking
│   ├── vectorizer.py   # tokenizer + TF-IDF + cosine, from scratch
│   ├── retriever.py    # top-k cosine ranking
│   ├── answerer.py     # extractive answering with citations
│   ├── evidence.py     # evidence rows behind an answer (--sources/--json)
│   ├── refresh.py      # incremental re-index (added/updated/removed/unchanged)
│   ├── stats.py        # index stats (docs/chunks/vocab/top terms)
│   ├── check.py        # regression check (saved question set, --record/--fail-on-change)
│   ├── store.py        # index save/load (JSON)
│   └── data/sample_docs/  # 3 demo documents
├── tests/test_tinyrag.py
├── demo.py             # zero-arg end-to-end demo
└── pyproject.toml
```

## 👤 Author

**Hassan Abbasi** — AI Engineer building AI-powered products with excellent user experiences.
[GitHub](https://github.com/realhassanabbasi)
