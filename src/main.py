"""
main.py - Command-line interface shell for the Quote Search Engine.

Commands
--------
build           Crawl the target website, build the inverted index, save to disk.
load            Load a previously built index from disk.
print <term>    Print the posting list for a single term.
find <query>    Search the index; use "quoted phrases" for phrase mode.
stats           Display index statistics.
benchmark       Run a small benchmark suite against the loaded index.
help            Show command reference.
quit / exit     Exit the shell.

Usage
-----
    python main.py [--index data/index.json] [--url https://quotes.toscrape.com/]
                   [--delay 6] [--max-pages N] [--log-level DEBUG|INFO|WARNING]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from crawler import Crawler, CrawledPage
from indexer import Indexer
from search import SearchEngine, SearchResponse


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

_HELP_TEXT = """
╔══════════════════════════════════════════════════════════╗
║               Quote Search Engine — Commands             ║
╠══════════════════════════════════════════════════════════╣
║  build              Crawl website & build index          ║
║  load               Load index from disk                 ║
║  print <term>       Show posting list for <term>         ║
║  find <query>       Keyword search (all terms required)  ║
║  find "<phrase>"    Phrase search (consecutive match)    ║
║  stats              Show index statistics                ║
║  benchmark          Run search benchmark                 ║
║  help               Show this help                       ║
║  quit / exit        Exit the shell                       ║
╚══════════════════════════════════════════════════════════╝
"""


def _banner() -> str:
    return r"""
 _____             _         _____                     _     
|  _  |           | |       /  ___|                   | |    
| | | |_   _  ___ | |_ ___  \ `--.  ___  __ _ _ __ ___| |__  
| | | | | | |/ _ \| __/ _ \  `--. \/ _ \/ _` | '__/ __| '_ \ 
\ \/' / |_| | (_) | ||  __/ /\__/ /  __/ (_| | | | (__| | | |
 \_/\_\\__,_|\___/ \__\___| \____/ \___|\__,_|_|  \___|_| |_|
                                                             
                                                             
  Quote Search Engine  |  TF-IDF Ranked  |  Type 'help' for commands
"""


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

class Shell:
    """
    Interactive REPL shell for the search engine.

    Parameters
    ----------
    index_path:
        Path to the JSON index file (used by ``build`` and ``load``).
    start_url:
        Seed URL for the crawler (used by ``build``).
    delay:
        Politeness delay in seconds (used by ``build``).
    max_pages:
        Maximum pages to crawl, or ``None`` for unlimited.
    """

    def __init__(
        self,
        index_path: Path,
        start_url: str,
        delay: float,
        max_pages: int | None,
    ) -> None:
        self.index_path = index_path
        self.start_url = start_url
        self.delay = delay
        self.max_pages = max_pages

        self._indexer: Indexer | None = None
        self._engine: SearchEngine | None = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive shell loop."""
        print(_banner())
        print(f"  Index file : {self.index_path}")
        print(f"  Target URL : {self.start_url}")
        print(f"  Politeness : {self.delay}s\n")

        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not raw:
                continue

            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            dispatch = {
                "build":     self._cmd_build,
                "load":      self._cmd_load,
                "print":     self._cmd_print,
                "find":      self._cmd_find,
                "stats":     self._cmd_stats,
                "benchmark": self._cmd_benchmark,
                "help":      self._cmd_help,
                "quit":      self._cmd_quit,
                "exit":      self._cmd_quit,
            }

            handler = dispatch.get(cmd)
            if handler is None:
                print(f"Unknown command '{cmd}'. Type 'help' for commands.")
                continue

            try:
                handler(args)
            except SystemExit:
                break
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).exception("Unhandled error in command '%s'.", cmd)
                print(f"[ERROR] {exc}")

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _cmd_build(self, _args: str) -> None:
        """Crawl the website, build the index, and save it."""
        print(f"Starting crawl of {self.start_url} …")
        print(f"Politeness delay: {self.delay}s | Max pages: {self.max_pages or 'unlimited'}")
        print("This may take several minutes. Please wait.\n")

        crawler = Crawler(
            start_url=self.start_url,
            politeness_delay=self.delay,
            max_pages=self.max_pages,
        )

        self._indexer = Indexer()
        t0 = time.perf_counter()
        pages: list[CrawledPage] = crawler.crawl()

        print(f"\nCrawled {len(pages)} pages. Building index …")
        for page in pages:
            self._indexer.add_page(url=page.url, text=page.text, title=page.title)

        self._indexer.compute_tf_idf()
        build_time = time.perf_counter() - t0

        self._indexer.save(self.index_path)
        self._engine = SearchEngine(self._indexer)

        stats = self._indexer.stats
        print(
            f"\n✓ Index built in {build_time:.1f}s — "
            f"{stats.num_documents} documents, "
            f"{stats.num_terms} unique terms, "
            f"{stats.total_tokens:,} total tokens."
        )
        print(f"  Saved to: {self.index_path}")

    def _cmd_load(self, _args: str) -> None:
        """Load the index from disk."""
        if not self.index_path.exists():
            print(f"[ERROR] No index found at '{self.index_path}'. Run 'build' first.")
            return

        print(f"Loading index from {self.index_path} …")
        t0 = time.perf_counter()
        self._indexer = Indexer.load(self.index_path)
        self._engine = SearchEngine(self._indexer)
        elapsed = time.perf_counter() - t0

        stats = self._indexer.stats
        print(
            f"✓ Index loaded in {elapsed:.2f}s — "
            f"{stats.num_documents} documents, "
            f"{stats.num_terms} unique terms."
        )

    def _cmd_print(self, args: str) -> None:
        """Print the posting list for a term."""
        if not self._require_index():
            return

        term = args.strip()
        if not term:
            print("Usage: print <term>")
            return

        print(self._indexer.print_postings(term))  # type: ignore[union-attr]

    def _cmd_find(self, args: str) -> None:
        """Search the index for a query."""
        if not self._require_index():
            return

        query = args.strip()
        if not query:
            print("Usage: find <query>   or   find \"<phrase>\"")
            return

        response: SearchResponse = self._engine.search(query)  # type: ignore[union-attr]
        print(SearchEngine.format_response(response))

    def _cmd_stats(self, _args: str) -> None:
        """Display index statistics."""
        if not self._require_index():
            return

        stats = self._indexer.stats  # type: ignore[union-attr]
        print("\nIndex Statistics")
        print("-" * 40)
        print(f"  Documents  : {stats.num_documents}")
        print(f"  Unique terms: {stats.num_terms}")
        print(f"  Total tokens: {stats.total_tokens:,}")

    def _cmd_benchmark(self, _args: str) -> None:
        """Run a small benchmark over common queries."""
        if not self._require_index():
            return

        queries = [
            "love",
            "life world",
            "good friends",
            "man",
            "truth beauty",
            '"the world"',
        ]
        print("\nBenchmark — 5 runs per query")
        print("-" * 55)
        print(f"  {'Query':<25} {'Avg (ms)':>10} {'Results':>8}")
        print("-" * 55)

        engine: SearchEngine = self._engine  # type: ignore[assignment]
        for query in queries:
            times = []
            n_results = 0
            for _ in range(5):
                r = engine.search(query)
                times.append(r.duration_ms)
                n_results = len(r.results)
            avg = sum(times) / len(times)
            label = query if len(query) <= 24 else query[:21] + "…"
            print(f"  {label:<25} {avg:>10.3f} {n_results:>8}")

        print("-" * 55)

    def _cmd_help(self, _args: str) -> None:
        print(_HELP_TEXT)

    def _cmd_quit(self, _args: str) -> None:
        print("Goodbye.")
        raise SystemExit(0)

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def _require_index(self) -> bool:
        """Print an error and return False if no index is loaded."""
        if self._indexer is None or self._engine is None:
            print("[ERROR] No index loaded. Run 'build' or 'load' first.")
            return False
        return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quote Search Engine — inverted index with TF-IDF ranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index", default="data/index.json",
        help="Path to the JSON index file.",
    )
    parser.add_argument(
        "--url", default="https://quotes.toscrape.com/",
        help="Seed URL for the crawler.",
    )
    parser.add_argument(
        "--delay", type=float, default=6.0,
        help="Politeness delay in seconds between requests.",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Maximum number of pages to crawl (omit for unlimited).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    args = _parse_args()
    _configure_logging(args.log_level)

    shell = Shell(
        index_path=Path(args.index),
        start_url=args.url,
        delay=args.delay,
        max_pages=args.max_pages,
    )
    shell.run()


if __name__ == "__main__":
    main()
