"""Tests for tinyrag. Run: python -m unittest discover -s tests"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinyrag import answerer, chunker, retriever, store, vectorizer


class TestChunker(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(chunker.chunk_text(""), [])

    def test_short_text_single_chunk(self):
        self.assertEqual(chunker.chunk_text("a b c", size=10), ["a b c"])

    def test_overlap_keeps_context(self):
        chunks = chunker.chunk_text("w1 w2 w3 w4 w5 w6", size=4, overlap=2)
        self.assertEqual(chunks[0], "w1 w2 w3 w4")
        self.assertEqual(chunks[1], "w3 w4 w5 w6")  # overlap of 2 words

    def test_no_overlap(self):
        chunks = chunker.chunk_text("w1 w2 w3 w4", size=2, overlap=0)
        self.assertEqual(chunks, ["w1 w2", "w3 w4"])

    def test_bad_size_raises(self):
        with self.assertRaises(ValueError):
            chunker.chunk_text("hello", size=0)


class TestVectorizer(unittest.TestCase):
    def test_tokenize_lowercases(self):
        self.assertEqual(vectorizer.tokenize("Hello WORLD!"), ["hello", "world"])

    def test_idf_common_term_lower(self):
        idf = vectorizer.build_idf([["a", "b"], ["a", "c"], ["a", "d"]])
        self.assertLess(idf["a"], idf["b"])  # 'a' in every doc → less weight

    def test_vector_is_normalized(self):
        idf = vectorizer.build_idf([["x", "y"], ["x", "z"]])
        vec = vectorizer.vectorize(["x", "x", "y"], idf)
        norm = sum(w * w for w in vec.values()) ** 0.5
        self.assertAlmostEqual(norm, 1.0)

    def test_cosine_identical_is_one(self):
        idf = vectorizer.build_idf([["x", "y"]])
        vec = vectorizer.vectorize(["x", "y"], idf)
        self.assertAlmostEqual(vectorizer.cosine(vec, vec), 1.0)

    def test_cosine_disjoint_is_zero(self):
        idf = vectorizer.build_idf([["x"], ["y"]])
        a = vectorizer.vectorize(["x"], idf)
        b = vectorizer.vectorize(["y"], idf)
        self.assertEqual(vectorizer.cosine(a, b), 0.0)


class TestRetriever(unittest.TestCase):
    def test_best_match_first(self):
        docs = [["cat", "sat"], ["dog", "barked"], ["cat", "sat", "mat"]]
        idf = vectorizer.build_idf(docs)
        vecs = [vectorizer.vectorize(d, idf) for d in docs]
        q = vectorizer.vectorize(["cat", "mat"], idf)
        top = retriever.retrieve(q, vecs, k=2)
        self.assertEqual(top[0][0], 2)  # doc with both terms wins
        self.assertEqual(len(top), 2)


class TestAnswerer(unittest.TestCase):
    def _chunks(self):
        return [
            {"doc": "a.txt", "chunk_id": 0,
             "text": "Cats sleep all day. Dogs bark at night."},
            {"doc": "b.txt", "chunk_id": 0,
             "text": "Python is a programming language. It is very popular."},
        ]

    def test_picks_sentence_with_query_terms(self):
        chunks = self._chunks()
        docs = [vectorizer.tokenize(c["text"]) for c in chunks]
        idf = vectorizer.build_idf(docs)
        vecs = [vectorizer.vectorize(d, idf) for d in docs]
        q = vectorizer.vectorize(vectorizer.tokenize("what do dogs do"), idf)
        ranked = retriever.retrieve(q, vecs, k=2)
        res = answerer.answer(set(vectorizer.tokenize("what do dogs do")),
                              idf, ranked, chunks)
        self.assertIn("bark", res["text"])
        self.assertEqual(res["doc"], "a.txt")

    def test_no_match_says_so(self):
        chunks = self._chunks()
        docs = [vectorizer.tokenize(c["text"]) for c in chunks]
        idf = vectorizer.build_idf(docs)
        res = answerer.answer({"zzzqqq"}, idf, [(0, 0.01)], chunks)
        self.assertIsNone(res["doc"])
        self.assertEqual(res["confidence"], 0.0)


class TestStore(unittest.TestCase):
    def test_roundtrip(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "idx.json")
            index = {"chunks": [{"doc": "a", "text": "hi"}],
                     "idf": {"hi": 1.0}, "vectors": [{"hi": 1.0}]}
            store.save_index(index, path)
            self.assertEqual(store.load_index(path), index)
        finally:
            shutil.rmtree(tmp)


class TestEndToEnd(unittest.TestCase):
    def test_index_and_ask_sample_docs(self):
        base = os.path.join(os.path.dirname(__file__), "..", "tinyrag",
                            "data", "sample_docs")
        records = chunker.chunk_folder(base)
        self.assertGreaterEqual(len(records), 3)
        toks = [vectorizer.tokenize(r["text"]) for r in records]
        idf = vectorizer.build_idf(toks)
        vecs = [vectorizer.vectorize(t, idf) for t in toks]
        q = "what is backpropagation"
        qv = vectorizer.vectorize(vectorizer.tokenize(q), idf)
        ranked = retriever.retrieve(qv, vecs, k=3)
        res = answerer.answer(set(vectorizer.tokenize(q)), idf, ranked, records)
        self.assertEqual(res["doc"], "neural-networks.txt")
        self.assertIn("backprop", res["text"].lower())


if __name__ == "__main__":
    unittest.main()
