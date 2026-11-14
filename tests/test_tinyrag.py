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

from tinyrag import __version__, answerer, ask_batch, check, chunker, evidence, retriever, search, stats, store, vectorizer
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

    def test_version_is_1_6_0(self):
        self.assertEqual(__version__, "1.6.0")


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


class TestStats(unittest.TestCase):
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

    def _stats(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["stats", "--index", self.idx, *argv])
        return buf.getvalue()

    def _payload(self):
        return stats.stats_payload(store.load_index(self.idx))

    def test_payload_aggregates(self):
        payload = self._payload()
        self.assertEqual(payload["documents"], 2)
        self.assertEqual(payload["chunks"], 2)
        self.assertEqual(payload["terms"], 6)  # alpha..zeta, stopwords none
        self.assertEqual(payload["total_words"], 180)
        self.assertEqual(payload["avg_chunk_words"], 90.0)
        self.assertTrue(payload["has_data"])

    def test_per_doc_rows(self):
        payload = self._payload()
        self.assertEqual(payload["docs"], [
            {"doc": "a.txt", "chunks": 1, "words": 90},
            {"doc": "b.txt", "chunks": 1, "words": 90},
        ])
        self.assertEqual(payload["largest_doc"]["chunks"], 1)

    def test_top_terms_by_chunk_count(self):
        chunks = [
            {"doc": "a.txt", "chunk_id": 0, "text": "alpha beta"},
            {"doc": "b.txt", "chunk_id": 0, "text": "alpha gamma"},
            {"doc": "c.txt", "chunk_id": 0, "text": "delta"},
        ]
        idf = vectorizer.build_idf([vectorizer.tokenize(c["text"])
                                    for c in chunks])
        payload = stats.stats_payload({"chunks": chunks, "idf": idf,
                                       "vectors": []})
        self.assertEqual(payload["top_terms"][0],
                         {"term": "alpha", "chunks": 2})
        self.assertEqual(len(payload["top_terms"]), 4)

    def test_empty_index_payload(self):
        payload = stats.stats_payload({"chunks": [], "idf": {}, "vectors": []})
        self.assertFalse(payload["has_data"])
        self.assertEqual(payload["documents"], 0)
        self.assertEqual(payload["chunks"], 0)
        self.assertEqual(payload["terms"], 0)
        self.assertEqual(payload["docs"], [])
        self.assertEqual(payload["top_terms"], [])
        self.assertIsNone(payload["largest_doc"])
        self.assertIn("run `tinyrag index",
                      stats.format_stats(payload))

    def test_cli_text_output(self):
        out = self._stats()
        self.assertIn("Documents: 2", out)
        self.assertIn("Chunks: 2", out)
        self.assertIn("Vocabulary: 6 terms", out)
        self.assertIn("Words: 180 (avg 90.0 per chunk)", out)
        self.assertIn("a.txt — 1 chunk, 90 words", out)
        self.assertIn("Most common terms", out)
        self.assertIn("alpha — in 1 chunk", out)

    def test_cli_json_schema(self):
        payload = json.loads(self._stats("--json"))  # pure JSON, no stray text
        self.assertEqual(payload["documents"], 2)
        self.assertEqual(payload["chunks"], 2)
        self.assertEqual(payload["terms"], 6)
        self.assertEqual(payload["total_words"], 180)
        self.assertEqual(payload["avg_chunk_words"], 90.0)
        self.assertTrue(payload["has_data"])
        self.assertEqual(len(payload["docs"]), 2)
        self.assertEqual(len(payload["top_terms"]), 6)

    def test_cli_missing_index_says_index_first(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["stats", "--index", os.path.join(self.tmp, "nope.json")])
        self.assertIn("run `tinyrag index", buf.getvalue())

    def test_largest_doc_picks_most_chunks(self):
        chunks = [
            {"doc": "a.txt", "chunk_id": 0, "text": "alpha beta"},
            {"doc": "a.txt", "chunk_id": 1, "text": "gamma delta"},
            {"doc": "b.txt", "chunk_id": 0, "text": "epsilon zeta"},
        ]
        idf = vectorizer.build_idf([vectorizer.tokenize(c["text"])
                                    for c in chunks])
        payload = stats.stats_payload({"chunks": chunks, "idf": idf,
                                       "vectors": []})
        self.assertEqual(payload["largest_doc"],
                         {"doc": "a.txt", "chunks": 2, "words": 4})


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.docs = os.path.join(self.tmp, "docs")
        os.makedirs(self.docs)
        self.idx = os.path.join(self.tmp, "index.json")
        self.qfile = os.path.join(self.tmp, "questions.json")
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

    def _questions(self):
        return [
            {"question": "what is alpha beta", "expect": None},
            {"question": "what is delta epsilon", "expect": None},
        ]

    def _check(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["check", self.qfile, "--index", self.idx, *argv])
        return buf.getvalue()

    def _record(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["check", self.qfile, "--index", self.idx, "--record"])
        return buf.getvalue()

    def _save_questions(self, questions):
        check.save_questions(self.qfile, questions)

    def test_record_fills_expectations_from_index(self):
        self._save_questions(self._questions())
        out = self._record()
        self.assertIn("recorded 2 expectation(s)", out)
        recorded = check.load_questions(self.qfile)
        self.assertEqual(recorded[0]["expect"], "a.txt")
        self.assertEqual(recorded[1]["expect"], "b.txt")

    def test_check_steady_when_nothing_changed(self):
        self._save_questions(self._questions())
        self._record()
        payload = check.check_payload(store.load_index(self.idx),
                                      check.load_questions(self.qfile))
        self.assertEqual(payload["checked"], 2)
        self.assertEqual(payload["steady"], 2)
        self.assertEqual(payload["changed"], 0)
        self.assertEqual(payload["unrecorded"], 0)

    def test_check_flags_changed_after_edit_and_refresh(self):
        self._save_questions(self._questions())
        self._record()
        self._write("a.txt", "quokka wombat koala " * 30)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["refresh", self.docs, "--index", self.idx])
        payload = check.check_payload(store.load_index(self.idx),
                                      check.load_questions(self.qfile))
        self.assertEqual(payload["changed"], 1)
        changed = [r for r in payload["results"]
                   if r["verdict"] == "changed"][0]
        self.assertEqual(changed["question"], "what is alpha beta")
        self.assertEqual(changed["expect"], "a.txt")
        self.assertIsNone(changed["now"])  # alpha is gone: an honest miss
        self.assertEqual(payload["steady"], 1)

    def test_unrecorded_question_is_not_a_failure(self):
        self._save_questions(self._questions())
        payload = check.check_payload(store.load_index(self.idx),
                                      check.load_questions(self.qfile))
        self.assertEqual(payload["unrecorded"], 2)
        self.assertEqual(payload["changed"], 0)
        out = self._check()
        self.assertIn("no expectation recorded", out)
        self.assertIn("2 unrecorded", out)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            check.load_questions(os.path.join(self.tmp, "gone.json"))

    def test_malformed_json_raises(self):
        bad = os.path.join(self.tmp, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("{not json")
        with self.assertRaises(ValueError):
            check.load_questions(bad)
        with open(bad, "w", encoding="utf-8") as f:
            json.dump({"questions": [{"nope": 1}]}, f)
        with self.assertRaises(ValueError):
            check.load_questions(bad)

    def test_cli_text_and_json(self):
        self._save_questions(self._questions())
        self._record()
        out = self._check()
        self.assertIn("2 checked: 2 steady, 0 changed", out)
        self.assertIn("✓ what is alpha beta", out)
        payload = json.loads(self._check("--json"))  # pure JSON
        self.assertEqual(payload["checked"], 2)
        self.assertEqual(payload["steady"], 2)
        self.assertEqual(payload["results"][0]["verdict"], "steady")
        self.assertEqual(payload["results"][0]["expect"], "a.txt")
        self.assertEqual(payload["results"][0]["now"], "a.txt")

    def test_fail_on_change_exit_codes(self):
        self._save_questions(self._questions())
        self._record()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):  # steady: exit 0, no raise
            main(["check", self.qfile, "--index", self.idx,
                  "--fail-on-change"])
        self._write("a.txt", "quokka wombat koala " * 30)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["refresh", self.docs, "--index", self.idx])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as cm:
                main(["check", self.qfile, "--index", self.idx,
                      "--fail-on-change"])
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("1 changed", buf.getvalue())

    def test_cli_missing_file_exits_1(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as cm:
                main(["check", os.path.join(self.tmp, "gone.json"),
                      "--index", self.idx])
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("no such file", buf.getvalue())


class TestSearch(unittest.TestCase):
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

    def _search(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["search", "--index", self.idx, *argv])
        return buf.getvalue()

    def _payload(self, query, k=5):
        return search.search_payload(store.load_index(self.idx), query, k=k)

    def test_matches_known_term(self):
        payload = self._payload("alpha beta")
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["matches"][0]["doc"], "a.txt")
        self.assertEqual(payload["matches"][0]["chunk_id"], 0)
        self.assertGreater(payload["matches"][0]["score"], 0.0)
        self.assertEqual(payload["searched_chunks"], 2)

    def test_best_first_across_docs(self):
        payload = self._payload("alpha delta gamma")
        self.assertEqual(payload["total"], 2)
        scores = [row["score"] for row in payload["matches"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(payload["matches"][0]["doc"], "a.txt")
        self.assertEqual(payload["matches"][1]["doc"], "b.txt")

    def test_no_overlap_returns_empty(self):
        payload = self._payload("quokka wombat")
        self.assertEqual(payload["total"], 0)
        self.assertEqual(payload["matches"], [])
        self.assertIn("No matches for", search.format_search(payload))

    def test_k_limits_rows(self):
        payload = self._payload("alpha delta", k=1)
        self.assertEqual(payload["total"], 1)
        self.assertEqual(len(payload["matches"]), 1)

    def test_agrees_with_ask_evidence(self):
        # Same query, same retriever: search's top row must equal the
        # top evidence row `ask --sources` would print.
        index = store.load_index(self.idx)
        query = "alpha beta"
        q_vec = vectorizer.vectorize(vectorizer.tokenize(query),
                                     index["idf"])
        ranked = retriever.retrieve(q_vec, index["vectors"], k=3)
        rows = evidence.evidence_payload(ranked, index["chunks"], n=3)
        payload = self._payload(query, k=3)
        self.assertEqual(payload["matches"][0]["doc"], rows[0]["doc"])
        self.assertEqual(payload["matches"][0]["chunk_id"],
                         rows[0]["chunk_id"])
        self.assertEqual(payload["matches"][0]["score"], rows[0]["score"])

    def test_cli_text_output(self):
        out = self._search("alpha beta")
        self.assertIn('1 match for "alpha beta"', out)
        self.assertIn("a.txt (chunk 0,", out)
        self.assertIn("alpha beta gamma", out)

    def test_cli_json_schema(self):
        payload = json.loads(self._search("alpha delta", "--json"))
        self.assertEqual(payload["query"], "alpha delta")
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["searched_chunks"], 2)
        self.assertEqual(len(payload["matches"]), 2)
        self.assertEqual(
            sorted(payload["matches"][0]),
            ["chunk_id", "doc", "excerpt", "rank", "score"])

    def test_cli_no_match_message(self):
        out = self._search("quokka wombat")
        self.assertIn('No matches for "quokka wombat"', out)
        self.assertIn("Try different words", out)

    def test_cli_missing_index_says_index_first(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["search", "alpha",
                  "--index", os.path.join(self.tmp, "nope.json")])
        self.assertIn("run `tinyrag index", buf.getvalue())


class TestAskBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.docs = os.path.join(self.tmp, "docs")
        os.makedirs(self.docs)
        self.idx = os.path.join(self.tmp, "index.json")
        self.qs = os.path.join(self.tmp, "questions.json")
        self._write("a.txt", "alpha beta gamma " * 30)
        self._write("b.txt", "delta epsilon zeta " * 30)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["index", self.docs, "--out", self.idx])
        self._write_questions([
            {"question": "what is alpha beta"},
            {"question": "what is delta epsilon"},
            {"question": "quokka wombat"},
        ])

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write(self, name, text):
        with open(os.path.join(self.docs, name), "w", encoding="utf-8") as f:
            f.write(text)

    def _write_questions(self, entries):
        with open(self.qs, "w", encoding="utf-8") as f:
            json.dump({"questions": entries}, f)

    def _payload(self):
        questions = check.load_questions(self.qs)
        return ask_batch.ask_batch_payload(store.load_index(self.idx),
                                           questions)

    def _run(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["ask-batch", "--index", self.idx, *argv])
        return buf.getvalue()

    def test_answers_known_question(self):
        row = self._payload()["questions"][0]
        self.assertTrue(row["answered"])
        self.assertEqual(row["doc"], "a.txt")
        self.assertEqual(row["chunk_id"], 0)
        self.assertGreater(row["confidence"], 0.0)
        self.assertIn("alpha beta gamma", row["answer"])

    def test_miss_is_honest(self):
        row = self._payload()["questions"][2]
        self.assertFalse(row["answered"])
        self.assertIsNone(row["doc"])
        self.assertEqual(row["confidence"], 0.0)
        self.assertEqual(row["answer"],
                         "I couldn't find anything relevant in your "
                         "documents.")

    def test_counts_math(self):
        payload = self._payload()
        self.assertEqual(payload["total"], 3)
        self.assertEqual(payload["answered"], 2)
        self.assertEqual(payload["missed"], 1)

    def test_agrees_with_ask(self):
        # Same question, same pipeline: the batch row must equal what
        # the answerer gives for a one-off ask.
        index = store.load_index(self.idx)
        question = "what is alpha beta"
        q_vec = vectorizer.vectorize(vectorizer.tokenize(question),
                                     index["idf"])
        ranked = retriever.retrieve(q_vec, index["vectors"], k=3)
        single = answerer.answer(set(vectorizer.tokenize(question)),
                                 index["idf"], ranked, index["chunks"])
        row = self._payload()["questions"][0]
        self.assertEqual(row["answer"], single["text"])
        self.assertEqual(row["doc"], single["doc"])
        self.assertEqual(row["confidence"], single["confidence"])

    def test_markdown_report(self):
        md = ask_batch.ask_batch_markdown(self._payload())
        self.assertIn("# TinyRAG answers", md)
        self.assertIn("## 1. what is alpha beta", md)
        self.assertIn("alpha beta gamma", md)
        self.assertIn("*Source: a.txt (chunk 0,", md)
        self.assertIn("## 3. quokka wombat", md)
        self.assertIn("*No answer found in the indexed documents.*", md)

    def test_cli_text_output(self):
        out = self._run(self.qs)
        self.assertIn("3 questions · 2 answered, 1 missed:", out)
        self.assertIn("1. what is alpha beta", out)
        self.assertIn("alpha beta gamma", out)
        self.assertIn("↳ a.txt (chunk 0,", out)
        self.assertIn("I couldn't find anything relevant", out)

    def test_cli_json_schema(self):
        payload = json.loads(self._run(self.qs, "--json"))
        self.assertEqual(sorted(payload),
                         ["answered", "missed", "questions", "total"])
        self.assertEqual(payload["total"], 3)
        self.assertEqual(payload["answered"], 2)
        row = payload["questions"][0]
        self.assertEqual(sorted(row),
                         ["answer", "answered", "chunk_id", "confidence",
                          "doc", "question"])

    def test_cli_writes_markdown_file(self):
        report = os.path.join(self.tmp, "answers.md")
        out = self._run(self.qs, "-o", report)
        self.assertIn(f"wrote 2 answers (1 missed) to {report}", out)
        with open(report, encoding="utf-8") as f:
            md = f.read()
        self.assertIn("# TinyRAG answers", md)
        self.assertIn("alpha beta gamma", md)

    def test_cli_missing_file_exits_1(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run(os.path.join(self.tmp, "nope.json"))
        self.assertEqual(ctx.exception.code, 1)

    def test_cli_malformed_file_exits_1(self):
        bad = os.path.join(self.tmp, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write('{"questions": ["not an object"]}')
        with self.assertRaises(SystemExit) as ctx:
            self._run(bad)
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
