"""
search.py - High-level search interface wrapping the inverted index.

Design decisions:
- **SearchEngine** is a thin facade over :class:`~indexer.Indexer` that adds:
  - Automatic phrase detection (quoted strings → phrase mode).
  - Query normalisation and validation.
  - Result formatting for the CLI.
  - Built-in benchmarking via :mod:`time`.
- Keeping search logic separate from the index allows the indexer to be
  tested in isolation and the search layer to be swapped independently.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from indexer import Indexer, PostingEntry, tokenise

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single ranked search result."""

    rank: int
    """1-based position in the result list."""

    url: str
    """URL of the matching page."""

    title: str
    """Page title."""

    score: float
    """Aggregated TF-IDF score (higher = more relevant)."""

    frequency: int
    """Raw occurrence count of the primary query term."""

    snippet: str = ""
    """Short excerpt from the page (populated if text is stored)."""


@dataclass
class SearchResponse:
    """Container for query results plus diagnostics."""

    query: str
    """Original query string."""

    phrase_mode: bool
    """Whether phrase search was used."""

    results: list[SearchResult]
    """Ranked list of results."""

    duration_ms: float
    """Wall-clock query time in milliseconds."""

    suggestions: list[str]
    """Spelling suggestions for unknown terms (empty if all terms found)."""


# ---------------------------------------------------------------------------
# SearchEngine
# ---------------------------------------------------------------------------

class SearchEngine:
    """
    High-level search interface over a loaded :class:`~indexer.Indexer`.

    Parameters
    ----------
    indexer:
        A fully built (and TF-IDF computed) :class:`~indexer.Indexer` instance.
    """

    def __init__(self, indexer: Indexer) -> None:
        self._indexer = indexer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str) -> SearchResponse:
        """
        Execute a search query and return a :class:`SearchResponse`.

        Phrase detection:
            If the query is wrapped in double-quotes (``"good friends"``),
            phrase mode is automatically activated.

        Parameters
        ----------
        query:
            Raw query string from the user.

        Returns
        -------
        SearchResponse
            Contains ranked results, timing, and spelling suggestions.
        """
        start = time.perf_counter()

        query = query.strip()
        phrase_mode = False

        # Detect quoted phrase
        if query.startswith('"') and query.endswith('"') and len(query) > 2:
            query = query[1:-1].strip()
            phrase_mode = True

        postings: list[PostingEntry] = self._indexer.find(query, phrase=phrase_mode)

        # Spelling suggestions for zero-result queries
        suggestions: list[str] = []
        if not postings:
            terms = tokenise(query, remove_stopwords=False)
            for term in terms:
                s = self._indexer.suggest(term)
                suggestions.extend(s)
            suggestions = list(dict.fromkeys(suggestions))  # deduplicate, preserve order

        results = [
            SearchResult(
                rank=i + 1,
                url=p.url,
                title=p.title or p.url,
                score=p.tf_idf,
                frequency=p.frequency,
            )
            for i, p in enumerate(postings)
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "Query '%s' (phrase=%s): %d results in %.2f ms.",
            query, phrase_mode, len(results), elapsed_ms,
        )

        return SearchResponse(
            query=query,
            phrase_mode=phrase_mode,
            results=results,
            duration_ms=elapsed_ms,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_response(response: SearchResponse, max_display: int = 20) -> str:
        """
        Render a :class:`SearchResponse` as a human-readable string.

        Parameters
        ----------
        response:
            The search response to format.
        max_display:
            Maximum number of results to show (default 20).

        Returns
        -------
        str
            Multi-line formatted output for the CLI.
        """
        lines: list[str] = []
        mode_label = "phrase" if response.phrase_mode else "keyword"
        lines.append(
            f"\nSearch results for '{response.query}' "
            f"[{mode_label} mode] — {len(response.results)} result(s) "
            f"({response.duration_ms:.2f} ms)"
        )
        lines.append("=" * 70)

        if not response.results:
            lines.append("  No pages found.")
            if response.suggestions:
                lines.append(
                    "\n  Did you mean: " + ", ".join(
                        f"'{s}'" for s in response.suggestions
                    ) + "?"
                )
            return "\n".join(lines)

        for result in response.results[:max_display]:
            lines.append(f"\n  [{result.rank}] {result.title}")
            lines.append(f"       URL   : {result.url}")
            lines.append(f"       Score : {result.score:.4f}  |  Freq: {result.frequency}")

        if len(response.results) > max_display:
            lines.append(
                f"\n  … and {len(response.results) - max_display} more result(s)."
            )

        return "\n".join(lines)
