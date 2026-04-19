"""
test_indexer.py - Unit and integration tests for indexer.py.

Coverage targets:
- Tokenisation (case folding, punctuation, stop-words)
- add_page / basic indexing
- TF-IDF computation correctness
- Boolean AND search
- Phrase (positional) search
- Spelling suggestions (Damerau-Levenshtein)
- Serialisation round-trip (save/load)
- print_postings formatting
- Edge cases: empty index, unknown term, single-term query
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from indexer import (
    Indexer,
    IndexStats,
    PostingEntry,
    _damerau_levenshtein,
    tokenise,
    tokenise_with_positions,
)


# ---------------------------------------------------------------------------
# Tokeniser tests
# ---------------------------------------------------------------------------

class TestTokenise(unittest.TestCase):
    def test_lowercase_normalisation(self) -> None:
        self.assertEqual(tokenise("Hello WORLD"), tokenise("hello world"))

    def test_punctuation_removed(self) -> None:
        tokens = tokenise("Hello, world!")
        self.assertNotIn("hello,", tokens)
        self.assertIn("hello", tokens)

    def test_stop_words_removed_by_default(self) -> None:
        tokens = tokenise("the quick brown fox")
        self.assertNotIn("the", tokens)
        self.assertIn("quick", tokens)

    def test_stop_words_retained_when_disabled(self) -> None:
        tokens = tokenise("the quick brown fox", remove_stopwords=False)
        self.assertIn("the", tokens)

    def test_empty_string(self) -> None:
        self.assertEqual(tokenise(""), [])

    def test_numbers_kept(self) -> None:
        tokens = tokenise("page 42 of the book")
        self.assertIn("42", tokens)

    def test_order_preserved(self) -> None:
        tokens = tokenise("alpha beta gamma", remove_stopwords=False)
        self.assertEqual(tokens, ["alpha", "beta", "gamma"])

    def test_tokenise_with_positions_keeps_stop_words(self) -> None:
        tokens = tokenise_with_positions("be or not to be")
        self.assertIn("be", tokens)
        self.assertIn("or", tokens)

    def test_extra_whitespace_handled(self) -> None:
        tokens = tokenise("  too   many   spaces  ")
        self.assertGreater(len(tokens), 0)
        self.assertNotIn("", tokens)


# ---------------------------------------------------------------------------
# Indexer — building
# ---------------------------------------------------------------------------

class TestIndexerAddPage(unittest.TestCase):
    def _make_indexer(self) -> Indexer:
        ix = Indexer()
        ix.add_page("https://example.com/p1", "love is a beautiful thing", "P1")
        ix.add_page("https://example.com/p2", "beautiful world of love and life", "P2")
        ix.compute_tf_idf()
        return ix

    def test_term_indexed(self) -> None:
        ix = self._make_indexer()
        postings = ix.get_postings("love")
        urls = [p.url for p in postings]
        self.assertIn("https://example.com/p1", urls)
        self.assertIn("https://example.com/p2", urls)

    def test_frequency_correct(self) -> None:
        ix = Indexer()
        ix.add_page("https://example.com/", "love love love", "T")
        ix.compute_tf_idf()
        postings = ix.get_postings("love")
        self.assertEqual(postings[0].frequency, 3)

    def test_positions_recorded(self) -> None:
        ix = Indexer()
        ix.add_page("https://example.com/", "the world is beautiful world", "T")
        ix.compute_tf_idf()
        postings = ix.get_postings("world")
        # "world" appears at position 1 and 4 in the full token stream
        self.assertEqual(len(postings[0].positions), 2)

    def test_stop_words_not_indexed(self) -> None:
        ix = Indexer()
        ix.add_page("https://example.com/", "the and is be", "T")
        ix.compute_tf_idf()
        postings = ix.get_postings("the")
        self.assertEqual(postings, [])

    def test_num_docs_incremented(self) -> None:
        ix = Indexer()
        self.assertEqual(ix.stats.num_documents, 0)
        ix.add_page("https://a.com/", "hello world", "A")
        self.assertEqual(ix.stats.num_documents, 1)

    def test_empty_text_does_not_crash(self) -> None:
        ix = Indexer()
        ix.add_page("https://example.com/", "", "Empty")
        ix.compute_tf_idf()
        self.assertEqual(ix.stats.num_documents, 1)

    def test_title_stored(self) -> None:
        ix = Indexer()
        ix.add_page("https://example.com/", "text", "My Title")
        ix.compute_tf_idf()
        postings = ix.get_postings("text")
        self.assertEqual(postings[0].title, "My Title")
    
    def test_index_stats_dataclass_fields(self):
        stats = IndexStats(num_documents=5, num_terms=100,
                        total_tokens=500, build_duration_seconds=1.5)
        self.assertEqual(stats.build_duration_seconds, 1.5)

    def test_load_invalid_json_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("not valid json{{{")
            name = f.name
        with self.assertRaises(Exception):
            Indexer.load(name)    


# ---------------------------------------------------------------------------
# TF-IDF computation
# ---------------------------------------------------------------------------

class TestTfIdf(unittest.TestCase):
    def test_tf_idf_positive(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "love and life", "A")
        ix.compute_tf_idf()
        postings = ix.get_postings("love")
        self.assertGreater(postings[0].tf_idf, 0)

    def test_higher_frequency_higher_score(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "love love love", "A")
        ix.add_page("https://b.com/", "love", "B")
        ix.compute_tf_idf()
        postings = ix.get_postings("love")
        scores = {p.url: p.tf_idf for p in postings}
        self.assertGreater(scores["https://a.com/"], scores["https://b.com/"])

    def test_rare_term_higher_idf(self) -> None:
        """A term appearing in fewer docs should have a higher IDF."""
        ix = Indexer()
        # "common" in 3 docs, "rare" in 1 doc
        ix.add_page("https://a.com/", "common rare", "A")
        ix.add_page("https://b.com/", "common word", "B")
        ix.add_page("https://c.com/", "common phrase", "C")
        ix.compute_tf_idf()
        rare_score = ix.get_postings("rare")[0].tf_idf
        common_score_a = next(
            p.tf_idf for p in ix.get_postings("common") if p.url == "https://a.com/"
        )
        self.assertGreater(rare_score, common_score_a)

    def test_results_sorted_by_score(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "love love love", "A")
        ix.add_page("https://b.com/", "love", "B")
        ix.compute_tf_idf()
        postings = ix.get_postings("love")
        scores = [p.tf_idf for p in postings]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_compute_tf_idf_idempotent(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "love", "A")
        ix.compute_tf_idf()
        score1 = ix.get_postings("love")[0].tf_idf
        ix.compute_tf_idf()
        score2 = ix.get_postings("love")[0].tf_idf
        self.assertAlmostEqual(score1, score2, places=8)


# ---------------------------------------------------------------------------
# Boolean AND search
# ---------------------------------------------------------------------------

class TestBooleanSearch(unittest.TestCase):
    def _make_indexer(self) -> Indexer:
        ix = Indexer()
        ix.add_page("https://a.com/", "good friends make life better", "A")
        ix.add_page("https://b.com/", "good food is essential", "B")
        ix.add_page("https://c.com/", "friends are important", "C")
        ix.compute_tf_idf()
        return ix

    def test_both_terms_required(self) -> None:
        ix = self._make_indexer()
        results = ix.find("good friends")
        urls = [r.url for r in results]
        self.assertIn("https://a.com/", urls)
        self.assertNotIn("https://b.com/", urls)
        self.assertNotIn("https://c.com/", urls)

    def test_single_term_search(self) -> None:
        ix = self._make_indexer()
        results = ix.find("good")
        urls = [r.url for r in results]
        self.assertIn("https://a.com/", urls)
        self.assertIn("https://b.com/", urls)

    def test_unknown_term_returns_empty(self) -> None:
        ix = self._make_indexer()
        results = ix.find("xyzzy")
        self.assertEqual(results, [])

    def test_one_unknown_term_in_multi_returns_empty(self) -> None:
        ix = self._make_indexer()
        results = ix.find("good xyzzy")
        self.assertEqual(results, [])

    def test_empty_query_returns_empty(self) -> None:
        ix = self._make_indexer()
        results = ix.find("")
        self.assertEqual(results, [])

    def test_results_sorted_by_score_descending(self) -> None:
        ix = self._make_indexer()
        results = ix.find("good")
        scores = [r.tf_idf for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ---------------------------------------------------------------------------
# Phrase search
# ---------------------------------------------------------------------------

class TestPhraseSearch(unittest.TestCase):
    def _make_indexer(self) -> Indexer:
        ix = Indexer()
        ix.add_page("https://a.com/", "the world is a beautiful place", "A")
        ix.add_page("https://b.com/", "beautiful world of wonders", "B")
        ix.add_page("https://c.com/", "world is not always beautiful", "C")
        ix.compute_tf_idf()
        return ix

    def test_consecutive_phrase_matched(self) -> None:
        ix = self._make_indexer()
        results = ix.find("beautiful place", phrase=True)
        urls = [r.url for r in results]
        self.assertIn("https://a.com/", urls)

    def test_non_consecutive_phrase_not_matched(self) -> None:
        """'beautiful world' appears in B consecutively but not in C."""
        ix = self._make_indexer()
        results = ix.find("beautiful world", phrase=True)
        urls = [r.url for r in results]
        self.assertIn("https://b.com/", urls)
        self.assertNotIn("https://c.com/", urls)

    def test_phrase_not_matched_in_wrong_order(self) -> None:
        ix = self._make_indexer()
        results = ix.find("place beautiful", phrase=True)
        self.assertEqual(results, [])

    def test_phrase_single_term_fallback(self) -> None:
        """Single-term phrase search equals normal get_postings."""
        ix = self._make_indexer()
        r_phrase = ix.find("world", phrase=True)
        r_normal = ix.find("world", phrase=False)
        self.assertEqual(
            {r.url for r in r_phrase},
            {r.url for r in r_normal},
        )


# ---------------------------------------------------------------------------
# Spelling suggestions
# ---------------------------------------------------------------------------

class TestSuggest(unittest.TestCase):
    def _make_indexer(self) -> Indexer:
        ix = Indexer()
        ix.add_page("https://a.com/", "love life beauty truth wisdom courage", "A")
        ix.compute_tf_idf()
        return ix

    def test_exact_match_returns_itself(self) -> None:
        ix = self._make_indexer()
        self.assertEqual(ix.suggest("love"), ["love"])

    def test_one_char_typo_suggested(self) -> None:
        ix = self._make_indexer()
        suggestions = ix.suggest("luve")  # 'love' with u→o
        self.assertIn("love", suggestions)

    def test_transposition_suggested(self) -> None:
        ix = self._make_indexer()
        suggestions = ix.suggest("lvoe")  # transposition of 'love'
        self.assertIn("love", suggestions)

    def test_no_suggestions_for_very_different_word(self) -> None:
        ix = self._make_indexer()
        suggestions = ix.suggest("xyzqwerty")
        self.assertEqual(suggestions, [])

    def test_suggestions_capped_at_max(self) -> None:
        ix = self._make_indexer()
        suggestions = ix.suggest("luve", max_suggestions=2)
        self.assertLessEqual(len(suggestions), 2)


# ---------------------------------------------------------------------------
# Damerau-Levenshtein distance
# ---------------------------------------------------------------------------

class TestDamerauLevenshtein(unittest.TestCase):
    def test_identical_strings(self) -> None:
        self.assertEqual(_damerau_levenshtein("abc", "abc"), 0)

    def test_single_insertion(self) -> None:
        self.assertEqual(_damerau_levenshtein("abc", "abcd"), 1)

    def test_single_deletion(self) -> None:
        self.assertEqual(_damerau_levenshtein("abcd", "abc"), 1)

    def test_single_substitution(self) -> None:
        self.assertEqual(_damerau_levenshtein("abc", "axc"), 1)

    def test_transposition(self) -> None:
        self.assertEqual(_damerau_levenshtein("ab", "ba"), 1)

    def test_empty_strings(self) -> None:
        self.assertEqual(_damerau_levenshtein("", ""), 0)

    def test_one_empty(self) -> None:
        self.assertEqual(_damerau_levenshtein("abc", ""), 3)

    def test_large_distance_capped(self) -> None:
        # Very different strings should return ≥3
        result = _damerau_levenshtein("abcdef", "xyz")
        self.assertGreaterEqual(result, 3)


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

class TestSerialisation(unittest.TestCase):
    def _make_indexer(self) -> Indexer:
        ix = Indexer()
        ix.add_page("https://a.com/", "love and beauty", "Page A")
        ix.add_page("https://b.com/", "love is eternal truth", "Page B")
        ix.compute_tf_idf()
        return ix

    def test_save_creates_file(self) -> None:
        ix = self._make_indexer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            ix.save(path)
            self.assertTrue(path.exists())

    def test_load_restores_terms(self) -> None:
        ix = self._make_indexer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            ix.save(path)
            ix2 = Indexer.load(path)
            postings = ix2.get_postings("love")
            urls = {p.url for p in postings}
            self.assertIn("https://a.com/", urls)
            self.assertIn("https://b.com/", urls)

    def test_load_restores_tf_idf(self) -> None:
        ix = self._make_indexer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            ix.save(path)
            ix2 = Indexer.load(path)
            score_orig = ix.get_postings("love")[0].tf_idf
            score_loaded = ix2.get_postings("love")[0].tf_idf
            self.assertAlmostEqual(score_orig, score_loaded, places=4)

    def test_load_restores_num_docs(self) -> None:
        ix = self._make_indexer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            ix.save(path)
            ix2 = Indexer.load(path)
            self.assertEqual(ix2.stats.num_documents, 2)

    def test_load_file_not_found_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Indexer.load("/nonexistent/path/index.json")

    def test_positions_survive_round_trip(self) -> None:
        ix = self._make_indexer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            ix.save(path)
            ix2 = Indexer.load(path)
            postings_orig = ix.get_postings("love")
            postings_loaded = ix2.get_postings("love")
            positions_orig = {p.url: p.positions for p in postings_orig}
            positions_loaded = {p.url: p.positions for p in postings_loaded}
            self.assertEqual(positions_orig, positions_loaded)


# ---------------------------------------------------------------------------
# print_postings
# ---------------------------------------------------------------------------

class TestPrintPostings(unittest.TestCase):
    def test_unknown_term_returns_not_found(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "hello world", "A")
        ix.compute_tf_idf()
        result = ix.print_postings("xyzzy")
        self.assertIn("not found", result.lower())

    def test_known_term_shows_url(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "hello world", "A")
        ix.compute_tf_idf()
        result = ix.print_postings("hello")
        self.assertIn("https://a.com/", result)

    def test_known_term_shows_frequency(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "hello hello hello", "A")
        ix.compute_tf_idf()
        result = ix.print_postings("hello")
        self.assertIn("3", result)

    def test_case_insensitive_lookup(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "hello world", "A")
        ix.compute_tf_idf()
        result_lower = ix.print_postings("hello")
        result_upper = ix.print_postings("HELLO")
        self.assertEqual(result_lower, result_upper)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestIndexStats(unittest.TestCase):
    def test_stats_reflect_added_pages(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "hello world foo bar", "A")
        ix.add_page("https://b.com/", "foo baz qux", "B")
        stats = ix.stats
        self.assertEqual(stats.num_documents, 2)
        self.assertGreater(stats.num_terms, 0)
        self.assertGreater(stats.total_tokens, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
