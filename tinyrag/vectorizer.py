"""TF-IDF vectorizer written from scratch. No sklearn, no numpy."""

from __future__ import annotations

import math
import re

_TOKEN_RE = re.compile(r"[a-z]+")

# Common english words carry no retrieval signal; dropping them is
# standard IR practice and stops "what is the ..." matching everything.
STOPWORDS = frozenset("""
a an the and or but if then else when what which who whom this that these those
is are was were be been being have has had do does did will would could should
may might must shall can of in on at to for with by from as into over after
it its they them their we you your he she him her our us my me not no yes
how why where when there here so than too very just about into out up down
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase english word tokens, stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def build_idf(docs_tokens: list[list[str]]) -> dict[str, float]:
    """IDF with smoothing so unseen-in-one-doc terms don't explode."""
    n = len(docs_tokens)
    df: dict[str, int] = {}
    for tokens in docs_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    return {term: math.log((1 + n) / (1 + c)) + 1.0 for term, c in df.items()}


def vectorize(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """TF-IDF vector, L2-normalized. Sparse dict: term -> weight."""
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    vec = {t: (c * idf[t]) for t, c in tf.items() if t in idf}
    norm = math.sqrt(sum(w * w for w in vec.values()))
    if norm == 0:
        return {}
    return {t: w / norm for t, w in vec.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity of two sparse vectors (both L2-normalized)."""
    if not a or not b:
        return 0.0
    small, big = (a, b) if len(a) < len(b) else (b, a)
    return sum(w * big.get(t, 0.0) for t, w in small.items())
