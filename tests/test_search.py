"""
test_search.py - Unit tests for search.py (SearchEngine, SearchResult, SearchResponse).

Coverage targets:
- Keyword search results and ranking
- Phrase auto-detection via quoted strings
- Spelling suggestions on zero results
- Response formatting (result count, score display, suggestions)
- Benchmark command (timing structure)
- Edge cases: empty query, single result, all stop-words
"""

from __future__ import annotations

import unittest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from indexer import Indexer
from search import SearchEngine, SearchResponse, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine() -> SearchEngine:
    """Build a small test index and return a SearchEngine over it."""
    ix = Indexer()
    ix.add_page(
        "https://a.com/",
        "love is a wonderful thing and life is full of love",
        "Page A — Love",
    )
    ix.add_page(
        "https://b.com/",
        "good friends make life better and friendship is beautiful",
        "Page B — Friends",
    )
    ix.add_page(
        "https://c.com/",
        "the world is a beautiful place full of wonder and magic",
        "Page C — World",
    )
    ix.add_page(
        "https://d.com/",
        "love and friendship are the foundations of a good life",
        "Page D — Love & Friends",
    )
    ix.compute_tf_idf()
    return SearchEngine(ix)


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------

class TestKeywordSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = _make_engine()

    def test_single_term_returns_results(self) -> None:
        response = self.engine.search("love")
        self.assertGreater(len(response.results), 0)

    def test_single_term_urls_correct(self) -> None:
        response = self.engine.search("love")
        urls = {r.url for r in response.results}
        self.assertIn("https://a.com/", urls)
        self.assertIn("https://d.com/", urls)

    def test_multi_term_boolean_and(self) -> None:
        """Only pages with BOTH terms should appear."""
        response = self.engine.search("love friendship")
        urls = {r.url for r in response.results}
        self.assertIn("https://d.com/", urls)
        # Page A has love but not friendship
        self.assertNotIn("https://a.com/", urls)

    def test_unknown_term_returns_empty(self) -> None:
        response = self.engine.search("xyzzy")
        self.assertEqual(len(response.results), 0)

    def test_empty_query_returns_empty(self) -> None:
        response = self.engine.search("")
        self.assertEqual(len(response.results), 0)

    def test_results_ranked_by_score(self) -> None:
        response = self.engine.search("love")
        scores = [r.score for r in response.results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ranks_are_sequential(self) -> None:
        response = self.engine.search("love")
        for i, result in enumerate(response.results):
            self.assertEqual(result.rank, i + 1)

    def test_phrase_mode_false_for_unquoted(self) -> None:
        response = self.engine.search("love life")
        self.assertFalse(response.phrase_mode)

    def test_duration_ms_positive(self) -> None:
        response = self.engine.search("love")
        self.assertGreater(response.duration_ms, 0)

    def test_stop_words_only_query_returns_empty(self) -> None:
        # "the and is" → all stop words → no results
        response = self.engine.search("the and is")
        self.assertEqual(len(response.results), 0)


# ---------------------------------------------------------------------------
# Phrase search (quoted strings)
# ---------------------------------------------------------------------------

class TestPhraseSearch(unittest.TestCase):
    def setUp(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "the world is a beautiful place", "A")
        ix.add_page("https://b.com/", "beautiful world without end", "B")
        ix.add_page("https://c.com/", "world beautiful separate words", "C")
        ix.compute_tf_idf()
        self.engine = SearchEngine(ix)

    def test_quoted_query_activates_phrase_mode(self) -> None:
        response = self.engine.search('"beautiful place"')
        self.assertTrue(response.phrase_mode)

    def test_phrase_matches_consecutive_only(self) -> None:
        response = self.engine.search('"beautiful place"')
        urls = {r.url for r in response.results}
        self.assertIn("https://a.com/", urls)
        self.assertNotIn("https://c.com/", urls)

    def test_unquoted_multi_term_is_keyword_mode(self) -> None:
        response = self.engine.search("beautiful place")
        self.assertFalse(response.phrase_mode)

    def test_empty_quotes_returns_empty(self) -> None:
        response = self.engine.search('""')
        self.assertEqual(len(response.results), 0)

    def test_single_quoted_word_same_as_keyword(self) -> None:
        r_quoted = self.engine.search('"world"')
        r_keyword = self.engine.search("world")
        self.assertEqual(
            {r.url for r in r_quoted.results},
            {r.url for r in r_keyword.results},
        )


# ---------------------------------------------------------------------------
# Spelling suggestions
# ---------------------------------------------------------------------------

class TestSpellingSuggestions(unittest.TestCase):
    def setUp(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "love life beauty truth wisdom", "A")
        ix.compute_tf_idf()
        self.engine = SearchEngine(ix)

    def test_suggestions_on_zero_results(self) -> None:
        response = self.engine.search("luve")  # typo for 'love'
        self.assertGreater(len(response.suggestions), 0)
        self.assertIn("love", response.suggestions)

    def test_no_suggestions_on_success(self) -> None:
        response = self.engine.search("love")
        self.assertEqual(response.suggestions, [])

    def test_suggestions_deduplicated(self) -> None:
        response = self.engine.search("luve")
        self.assertEqual(len(response.suggestions), len(set(response.suggestions)))


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

class TestFormatResponse(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = _make_engine()

    def test_format_includes_query(self) -> None:
        response = self.engine.search("love")
        output = SearchEngine.format_response(response)
        self.assertIn("love", output)

    def test_format_includes_result_count(self) -> None:
        response = self.engine.search("love")
        output = SearchEngine.format_response(response)
        self.assertIn(str(len(response.results)), output)

    def test_format_no_results_message(self) -> None:
        response = self.engine.search("xyzzy")
        output = SearchEngine.format_response(response)
        self.assertIn("No pages found", output)

    def test_format_shows_url(self) -> None:
        response = self.engine.search("love")
        output = SearchEngine.format_response(response)
        self.assertIn("https://a.com/", output)

    def test_format_suggestions_shown_on_zero_results(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "love beauty", "A")
        ix.compute_tf_idf()
        engine = SearchEngine(ix)
        response = engine.search("luve")
        output = SearchEngine.format_response(response)
        self.assertIn("Did you mean", output)

    def test_max_display_truncates(self) -> None:
        ix = Indexer()
        for i in range(25):
            ix.add_page(f"https://example.com/{i}/", "love", f"P{i}")
        ix.compute_tf_idf()
        engine = SearchEngine(ix)
        response = engine.search("love")
        output = SearchEngine.format_response(response, max_display=5)
        self.assertIn("more result", output)

    def test_phrase_mode_label_in_output(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "good friends together", "A")
        ix.compute_tf_idf()
        engine = SearchEngine(ix)
        response = engine.search('"good friends"')
        output = SearchEngine.format_response(response)
        self.assertIn("phrase", output)

    def test_timing_shown_in_output(self) -> None:
        response = self.engine.search("love")
        output = SearchEngine.format_response(response)
        self.assertIn("ms", output)


# ---------------------------------------------------------------------------
# SearchResult / SearchResponse dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses(unittest.TestCase):
    def test_search_result_fields(self) -> None:
        r = SearchResult(rank=1, url="https://a.com/", title="A", score=1.5, frequency=3)
        self.assertEqual(r.rank, 1)
        self.assertEqual(r.score, 1.5)

    def test_search_response_fields(self) -> None:
        resp = SearchResponse(
            query="test",
            phrase_mode=False,
            results=[],
            duration_ms=1.23,
            suggestions=[],
        )
        self.assertEqual(resp.query, "test")
        self.assertFalse(resp.phrase_mode)


# ---------------------------------------------------------------------------
# Integration: engine over larger vocabulary
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):
    def test_multi_page_tfidf_ranking_order(self) -> None:
        """Page with more occurrences of a term should rank higher."""
        ix = Indexer()
        ix.add_page("https://a.com/", "love love love love love", "A")
        ix.add_page("https://b.com/", "love is in the air", "B")
        ix.compute_tf_idf()
        engine = SearchEngine(ix)
        response = engine.search("love")
        self.assertEqual(response.results[0].url, "https://a.com/")

    def test_case_insensitive_query(self) -> None:
        ix = Indexer()
        ix.add_page("https://a.com/", "Love is eternal", "A")
        ix.compute_tf_idf()
        engine = SearchEngine(ix)
        r_lower = engine.search("love")
        r_upper = engine.search("LOVE")
        self.assertEqual(
            {r.url for r in r_lower.results},
            {r.url for r in r_upper.results},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
