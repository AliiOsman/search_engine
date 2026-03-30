"""
indexer.py - Inverted index with TF-IDF weighting and positional information.

Design decisions:
- **Inverted index** maps term → {url → PostingEntry} for O(1) average-case
  lookup per term.
- **PostingEntry** stores raw frequency, a sorted positions list (for phrase
  search), and the pre-computed TF-IDF score (computed after all pages are
  indexed so IDF can be calculated globally).
- **TF-IDF** uses log-normalised term frequency and smooth IDF to prevent
  division by zero:
      tf(t, d)  = 1 + log(count(t, d))   if count > 0, else 0
      idf(t)    = log((1 + N) / (1 + df(t))) + 1
      score     = tf * idf
- **Phrase search** is O(P · k) where P is the number of candidate postings
  for the rarest term and k is the phrase length (positional intersection).
- Serialisation uses JSON (human-readable, no binary dependency).

Complexity:
- Build:  O(T)  where T = total tokens across all pages.
- Lookup: O(df) where df = document frequency of the queried term.
- Phrase: O(P · k) in the worst case (see above).
- Save/Load: O(T) proportional to index size.
"""

from __future__ import annotations

import json
import logging
import math
import re
import string
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-words (excluded from indexing to reduce noise; kept minimal for recall)
# ---------------------------------------------------------------------------
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "it", "as", "be",
        "was", "are", "were", "that", "this", "which", "have", "has",
        "had", "not", "do", "does", "did",
    }
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PostingEntry:
    """Statistics for one (term, document) pair."""

    url: str
    """URL of the document."""

    frequency: int = 0
    """Raw occurrence count of the term in this document."""

    positions: list[int] = field(default_factory=list)
    """0-based token offsets of the term in the document (for phrase search)."""

    tf_idf: float = 0.0
    """TF-IDF score, populated after :meth:`Indexer.compute_tf_idf` is called."""

    title: str = ""
    """Page title, stored for display convenience."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PostingEntry":
        """Deserialise from a plain dict."""
        return cls(**data)


@dataclass
class IndexStats:
    """Global statistics about the index."""

    num_documents: int = 0
    num_terms: int = 0
    total_tokens: int = 0
    build_duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Tokeniser (module-level for reuse in search)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s'-]")  # keep apostrophes and hyphens initially
_CLEAN_RE = re.compile(r"[^a-z0-9]")  # strip anything non-alphanumeric after lower


def tokenise(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Normalise and tokenise *text* into a list of lowercase tokens.

    Steps:
        1. Lower-case.
        2. Remove punctuation (preserving intra-word apostrophes/hyphens).
        3. Split on whitespace.
        4. Strip remaining non-alphanumeric characters from token boundaries.
        5. Optionally filter stop-words and empty tokens.

    Parameters
    ----------
    text:
        Raw text to tokenise.
    remove_stopwords:
        If ``True`` (default), common stop-words are excluded.

    Returns
    -------
    list[str]
        Ordered list of normalised tokens.
    """
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = text.split()
    cleaned: list[str] = []
    for tok in tokens:
        tok = _CLEAN_RE.sub("", tok)
        if not tok:
            continue
        if remove_stopwords and tok in _STOP_WORDS:
            continue
        cleaned.append(tok)
    return cleaned


def tokenise_with_positions(text: str) -> list[str]:
    """
    Tokenise preserving all positions (stop-words retained) for phrase search.

    This uses a separate pass that *does not* remove stop-words so that phrase
    positions align with the original token stream.
    """
    return tokenise(text, remove_stopwords=False)


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class Indexer:
    """
    Builds and manages an inverted index over a collection of web pages.

    The index maps each unique term to a list of :class:`PostingEntry` objects,
    one per document containing that term.

    Usage
    -----
    >>> from crawler import CrawledPage
    >>> indexer = Indexer()
    >>> indexer.add_page(page)          # add pages one at a time
    >>> indexer.compute_tf_idf()        # must call before searching
    >>> indexer.save("data/index.json")
    """

    def __init__(self) -> None:
        # term → {url → PostingEntry}
        self._index: dict[str, dict[str, PostingEntry]] = defaultdict(dict)
        # url → page title (for display)
        self._titles: dict[str, str] = {}
        # number of documents indexed
        self._num_docs: int = 0
        # total tokens processed (for stats)
        self._total_tokens: int = 0
        # flag: has TF-IDF been computed?
        self._tf_idf_computed: bool = False

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_page(self, url: str, text: str, title: str = "") -> None:
        """
        Index a single page.

        Parameters
        ----------
        url:
            Canonical URL of the page.
        text:
            Full visible text content.
        title:
            Page title (stored for display; not indexed separately).
        """
        self._titles[url] = title
        self._num_docs += 1

        # Build positional token stream (stop-words kept for phrase alignment)
        all_tokens = tokenise_with_positions(text)
        self._total_tokens += len(all_tokens)

        for position, token in enumerate(all_tokens):
            if token in _STOP_WORDS:
                continue  # stop-words not added to index, but position counted

            entry = self._index[token].setdefault(
                url, PostingEntry(url=url, title=title)
            )
            entry.frequency += 1
            entry.positions.append(position)

        # Invalidate TF-IDF cache
        self._tf_idf_computed = False
        logger.debug("Indexed page: %s (%d tokens)", url, len(all_tokens))

    def compute_tf_idf(self) -> None:
        """
        (Re)compute TF-IDF scores for all (term, document) pairs.

        Must be called once after all pages have been added, and again if
        pages are added later.

        Formula
        -------
        tf(t, d)  = 1 + log₁₀(freq(t, d))   [log normalisation]
        idf(t)    = log₁₀((1 + N) / (1 + df(t))) + 1   [smooth IDF]
        score     = tf × idf
        """
        N = self._num_docs
        if N == 0:
            return

        for term, postings in self._index.items():
            df = len(postings)
            idf = math.log10((1 + N) / (1 + df)) + 1
            for entry in postings.values():
                tf = 1 + math.log10(entry.frequency) if entry.frequency > 0 else 0.0
                entry.tf_idf = tf * idf

        self._tf_idf_computed = True
        logger.debug("TF-IDF scores computed for %d terms.", len(self._index))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_postings(self, term: str) -> list[PostingEntry]:
        """
        Return all postings for *term*, sorted by descending TF-IDF score.

        Parameters
        ----------
        term:
            A single search term (case-insensitive).

        Returns
        -------
        list[PostingEntry]
            Postings sorted by TF-IDF (highest first).  Empty list if not found.
        """
        term = term.lower().strip()
        postings = self._index.get(term, {})
        return sorted(postings.values(), key=lambda e: e.tf_idf, reverse=True)

    def find(self, query: str, phrase: bool = False) -> list[PostingEntry]:
        """
        Search the index for a query string and return ranked results.

        For multi-term queries:
        - **Default (bag-of-words)**: Returns pages containing *all* query
          terms, ranked by summed TF-IDF score.
        - **Phrase mode** (``phrase=True``): Returns only pages where the
          query terms appear consecutively in order (positional intersection).

        Parameters
        ----------
        query:
            One or more space-separated search terms.
        phrase:
            If ``True``, enforce consecutive positional match.

        Returns
        -------
        list[PostingEntry]
            Aggregated result entries (one per matching page) sorted by score.
        """
        if not self._tf_idf_computed:
            self.compute_tf_idf()

        terms = tokenise(query, remove_stopwords=False)
        if not terms:
            return []

        if len(terms) == 1:
            return self.get_postings(terms[0])

        if phrase:
            return self._phrase_search(terms)
        return self._boolean_and_search(terms)

    def _boolean_and_search(self, terms: list[str]) -> list[PostingEntry]:
        """
        Return pages containing ALL terms, ranked by summed TF-IDF.

        Complexity: O(df_min + k·df_min) where df_min is the document frequency
        of the rarest term and k is the number of query terms.
        """
        # Start with candidate URLs from the rarest term (optimisation)
        posting_dicts = [self._index.get(t, {}) for t in terms]
        if any(len(p) == 0 for p in posting_dicts):
            return []

        # Intersect by URL
        candidate_urls: set[str] = set(posting_dicts[0].keys())
        for pd in posting_dicts[1:]:
            candidate_urls &= set(pd.keys())

        if not candidate_urls:
            return []

        # Aggregate scores
        results: list[PostingEntry] = []
        for url in candidate_urls:
            combined_score = sum(pd[url].tf_idf for pd in posting_dicts)
            # Use the first term's entry as the representative; override score
            base = posting_dicts[0][url]
            entry = PostingEntry(
                url=url,
                frequency=base.frequency,
                positions=base.positions,
                tf_idf=combined_score,
                title=base.title,
            )
            results.append(entry)

        return sorted(results, key=lambda e: e.tf_idf, reverse=True)

    def _phrase_search(self, terms: list[str]) -> list[PostingEntry]:
        """
        Return only pages where *terms* appear as a consecutive phrase.

        Algorithm (positional intersection):
        1. Find candidate pages containing ALL terms.
        2. For each candidate, check whether any position p exists such that
           term[i] appears at position p+i for all i.

        Complexity: O(P · k · F) where P = candidate pages, k = phrase length,
        F = avg frequency of each term.
        """
        posting_dicts = [self._index.get(t, {}) for t in terms]
        if any(len(p) == 0 for p in posting_dicts):
            return []

        candidate_urls: set[str] = set(posting_dicts[0].keys())
        for pd in posting_dicts[1:]:
            candidate_urls &= set(pd.keys())

        results: list[PostingEntry] = []
        for url in candidate_urls:
            # Anchor on first term's positions
            anchor_positions = posting_dicts[0][url].positions
            for start_pos in anchor_positions:
                matched = True
                for offset, pd in enumerate(posting_dicts[1:], start=1):
                    if (start_pos + offset) not in pd[url].positions:
                        matched = False
                        break
                if matched:
                    combined_score = sum(pd[url].tf_idf for pd in posting_dicts)
                    entry = PostingEntry(
                        url=url,
                        frequency=posting_dicts[0][url].frequency,
                        positions=posting_dicts[0][url].positions,
                        tf_idf=combined_score,
                        title=posting_dicts[0][url].title,
                    )
                    results.append(entry)
                    break  # one match per document is enough

        return sorted(results, key=lambda e: e.tf_idf, reverse=True)

    def suggest(self, term: str, max_suggestions: int = 5) -> list[str]:
        """
        Return spelling suggestions for an unknown *term* using edit distance.

        Uses Damerau-Levenshtein distance (transposition + insert/delete/sub)
        capped at distance 2 for performance.

        Complexity: O(V · L²) where V = vocabulary size, L = term length.

        Parameters
        ----------
        term:
            A misspelled or unknown term.
        max_suggestions:
            Maximum number of suggestions to return (default 5).
        """
        term = term.lower().strip()
        if term in self._index:
            return [term]

        candidates: list[tuple[int, str]] = []
        for vocab_term in self._index:
            dist = _damerau_levenshtein(term, vocab_term)
            if dist <= 2:
                candidates.append((dist, vocab_term))

        candidates.sort(key=lambda x: (x[0], -len(self._index[x[1]])))
        return [c[1] for c in candidates[:max_suggestions]]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Persist the index to *path* as a JSON file.

        The format is a top-level dict with keys ``"metadata"``, ``"titles"``,
        and ``"index"``.

        Parameters
        ----------
        path:
            Destination file path (created/overwritten).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "metadata": {
                "num_documents": self._num_docs,
                "num_terms": len(self._index),
                "total_tokens": self._total_tokens,
                "tf_idf_computed": self._tf_idf_computed,
            },
            "titles": self._titles,
            "index": {
                term: {url: entry.to_dict() for url, entry in postings.items()}
                for term, postings in self._index.items()
            },
        }

        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        logger.info("Index saved to %s (%d terms).", path, len(self._index))

    @classmethod
    def load(cls, path: str | Path) -> "Indexer":
        """
        Load a previously saved index from *path*.

        Parameters
        ----------
        path:
            Path to a JSON index file produced by :meth:`save`.

        Returns
        -------
        Indexer
            A fully initialised :class:`Indexer` instance ready for queries.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file is not a valid index JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        instance = cls()
        meta = payload.get("metadata", {})
        instance._num_docs = meta.get("num_documents", 0)
        instance._total_tokens = meta.get("total_tokens", 0)
        instance._tf_idf_computed = meta.get("tf_idf_computed", False)
        instance._titles = payload.get("titles", {})

        for term, postings_data in payload.get("index", {}).items():
            for url, entry_data in postings_data.items():
                instance._index[term][url] = PostingEntry.from_dict(entry_data)

        logger.info(
            "Index loaded from %s (%d terms, %d docs).",
            path, len(instance._index), instance._num_docs,
        )
        return instance

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> IndexStats:
        """Return a snapshot of global index statistics."""
        return IndexStats(
            num_documents=self._num_docs,
            num_terms=len(self._index),
            total_tokens=self._total_tokens,
        )

    def print_postings(self, term: str) -> str:
        """
        Format the posting list for *term* as a human-readable string.

        Parameters
        ----------
        term:
            The term to look up.

        Returns
        -------
        str
            Formatted multi-line string, or a "not found" message.
        """
        term = term.lower().strip()
        postings = self._index.get(term)
        if not postings:
            return f"Term '{term}' not found in index."

        lines = [f"Postings for '{term}' ({len(postings)} document(s)):"]
        lines.append("-" * 60)
        for entry in sorted(postings.values(), key=lambda e: e.tf_idf, reverse=True):
            lines.append(f"  URL      : {entry.url}")
            lines.append(f"  Title    : {entry.title or '(no title)'}")
            lines.append(f"  Frequency: {entry.frequency}")
            lines.append(f"  TF-IDF   : {entry.tf_idf:.4f}")
            preview = entry.positions[:10]
            suffix = "…" if len(entry.positions) > 10 else ""
            lines.append(f"  Positions: {preview}{suffix}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Damerau-Levenshtein distance (for spell suggestion)
# ---------------------------------------------------------------------------

def _damerau_levenshtein(s1: str, s2: str) -> int:
    """
    Compute the optimal string alignment (OSA) distance between *s1* and *s2*.

    Supports single-character transpositions in addition to insertions,
    deletions, and substitutions. O(|s1|*|s2|) time and space.
    Returns the distance capped at 3 (beyond that, too different for suggestions).
    """
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return min(len2, 3)
    if len2 == 0:
        return min(len1, 3)
    if abs(len1 - len2) > 2:
        return 3

    prev2: list = []
    prev1: list = list(range(len2 + 1))

    for i in range(1, len1 + 1):
        curr: list = [i] + [0] * len2
        for j in range(1, len2 + 1):
            sub_cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,
                prev1[j] + 1,
                prev1[j - 1] + sub_cost,
            )
            if (prev2 and i > 1 and j > 1
                    and s1[i - 1] == s2[j - 2]
                    and s1[i - 2] == s2[j - 1]):
                curr[j] = min(curr[j], prev2[j - 2] + 1)
        prev2 = prev1
        prev1 = curr

    dist = prev1[len2]
    return dist if dist <= 2 else 3
