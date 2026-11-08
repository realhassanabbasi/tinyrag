"""Tests for tinyrag. Run: python -m unittest discover -s tests"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinyrag import __version__, answerer, chunker, evidence, retriever, store, vectorizer
from tinyrag.cli import main


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


class TestEvidence(unittest.TestCase):
    def _chunks(self):
        return [
            {"doc": "a.txt", "chunk_id": 0,
             "text": "Cats sleep all day. Dogs bark at night."},
            {"doc": "b.txt", "chunk_id": 2,
             "text": "Python is a programming language. It is very popular."},
        ]

    def test_payload_shape(self):
        rows = evidence.evidence_payload([(0, 0.29), (1, 0.10)], self._chunks())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {"rank": 1, "doc": "a.txt", "chunk_id": 0,
                                   "score": 0.29,
                                   "excerpt": "Cats sleep all day. Dogs bark at night."})
        self.assertEqual(rows[1]["rank"], 2)
        self.assertEqual(rows[1]["doc"], "b.txt")
        self.assertEqual(rows[1]["chunk_id"], 2)

    def test_zero_score_dropped(self):
        rows = evidence.evidence_payload([(0, 0.29), (1, 0.0)], self._chunks())
        self.assertEqual(len(rows), 1)  # a zero-score chunk matched nothing

    def test_excerpt_truncates(self):
        long_text = "word " * 100
        chunks = [{"doc": "a.txt", "chunk_id": 0, "text": long_text}]
        rows = evidence.evidence_payload([(0, 0.5)], chunks, excerpt_chars=20)
        self.assertTrue(rows[0]["excerpt"].endswith("…"))
        self.assertLessEqual(len(rows[0]["excerpt"]), 21)

    def test_n_caps_rows(self):
        rows = evidence.evidence_payload([(0, 0.29), (1, 0.10)],
                                         self._chunks(), n=1)
        self.assertEqual(len(rows), 1)

    def test_format_evidence(self):
        rows = evidence.evidence_payload([(0, 0.29)], self._chunks())
        text = evidence.format_evidence(rows)
        self.assertIn("Evidence:", text)
        self.assertIn("1. a.txt (chunk 0, 0.29)", text)
        self.assertIn("Cats sleep all day", text)


class TestAskSources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.idx = os.path.join(self.tmp, "index.json")
        base = os.path.join(os.path.dirname(__file__), "..", "tinyrag",
                            "data", "sample_docs")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["index", base, "--out", self.idx])

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _ask(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["ask", *argv, "--index", self.idx])
        return buf.getvalue()

    def test_sources_shows_evidence(self):
        out = self._ask("what is backpropagation", "--sources", "2")
        self.assertIn("Evidence:", out)
        self.assertIn("neural-networks.txt", out)
        self.assertIn("1. ", out)

    def test_default_ask_hides_evidence(self):
        out = self._ask("what is backpropagation")
        self.assertNotIn("Evidence:", out)

    def test_json_carries_sources(self):
        out = self._ask("what is backpropagation", "--json")
        payload = json.loads(out)  # pure JSON, no stray text
        self.assertEqual(payload["doc"], "neural-networks.txt")
        self.assertIn("backprop", payload["answer"].lower())
        self.assertGreaterEqual(len(payload["sources"]), 1)
        self.assertEqual(payload["sources"][0]["rank"], 1)
        self.assertIn("excerpt", payload["sources"][0])

    def test_miss_has_no_evidence(self):
        out = self._ask("what is the capital of france", "--sources", "3")
        self.assertNotIn("Evidence:", out)
        self.assertIn("couldn't find anything", out)

    def test_version_is_1_2_0(self):
        self.assertEqual(__version__, "1.2.0")


class TestRefresh(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.docs = os.path.join(self.tmp, "docs")
        os.makedirs(self.docs)
        self.idx = os.path.join(self.tmp, "index.json")
        self._write("a.txt", "alpha beta gamma " * 30)
        self._write("b.txt", "delta epsilon zeta " * 30)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["index", self.docs, "--out", self.idx])

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write(self, name, text):
        with open(os.path.join(self.docs, name), "w", encoding="utf-8") as f:
            f.write(text)

    def _refresh(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["refresh", self.docs, "--index", self.idx, *argv])
        return buf.getvalue()

    def _ask(self, question):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["ask", question, "--index", self.idx])
        return buf.getvalue()

    def test_no_change_refresh_keeps_everything(self):
        out = self._refresh()
        self.assertIn("Unchanged: 2 (a.txt, b.txt)", out)
        self.assertIn("Added: 0", out)
        self.assertIn("index now: 2 documents, 2 chunks", out)
        self.assertIn("alpha", self._ask("what is alpha beta"))

    def test_edited_file_is_updated_and_searchable(self):
        self._write("a.txt", "quokka wombat koala " * 30)
        out = self._refresh()
        self.assertIn("Updated: 1 (a.txt)", out)
        self.assertIn("Unchanged: 1 (b.txt)", out)
        self.assertIn("quokka", self._ask("what is a quokka wombat"))

    def test_added_file_is_indexed(self):
        self._write("c.txt", "mango papaya guava " * 30)
        out = self._refresh()
        self.assertIn("Added: 1 (c.txt)", out)
        self.assertIn("mango", self._ask("what is mango papaya"))

    def test_deleted_file_is_removed(self):
        os.remove(os.path.join(self.docs, "b.txt"))
        out = self._refresh()
        self.assertIn("Removed: 1 (b.txt)", out)
        self.assertIn("index now: 1 documents, 1 chunks", out)
        self.assertIn("couldn't find anything",
                      self._ask("what is delta epsilon"))

    def test_legacy_index_rebuilds_once(self):
        index = store.load_index(self.idx)
        index.pop("meta", None)  # simulate a v1.1.0 index
        store.save_index(index, self.idx)
        out = self._refresh()
        self.assertIn("no file signatures", out)
        self.assertIn("Updated: 2 (a.txt, b.txt)", out)
        out2 = self._refresh()  # second refresh is a normal no-op
        self.assertNotIn("no file signatures", out2)
        self.assertIn("Unchanged: 2", out2)

    def test_missing_index_says_index_first(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["refresh", self.docs, "--index",
                  os.path.join(self.tmp, "nope.json")])
        self.assertIn("run `tinyrag index", buf.getvalue())

    def test_missing_folder_says_so(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["refresh", os.path.join(self.tmp, "gone"),
                  "--index", self.idx])
        self.assertIn("no such folder", buf.getvalue())

    def test_json_report_schema(self):
        out = self._refresh("--json")
        report = json.loads(out)  # pure JSON, no stray text
        self.assertEqual(report["added"], [])
        self.assertEqual(report["unchanged"], ["a.txt", "b.txt"])
        self.assertEqual(report["docs"], 2)
        self.assertEqual(report["chunks"], 2)
        self.assertFalse(report["legacy"])


if __name__ == "__main__":
    unittest.main()
