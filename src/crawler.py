"""
crawler.py - Web crawler for the quotes.toscrape.com search engine.

Design decisions:
- BFS traversal ensures all reachable pages are discovered systematically.
- Politeness window of ≥6 s between requests avoids overloading the server.
- robots.txt is respected via urllib.robotparser (O(1) per-URL check after parse).
- Exponential back-off on transient HTTP errors (429, 5xx) with a configurable
  maximum of 3 retries before a URL is skipped gracefully.
- Only same-domain, non-fragment, non-duplicate URLs are enqueued (O(1) set lookup).

Complexity:
- Time:  O(P · T) where P = number of pages crawled, T = avg. tokens per page.
- Space: O(P) for visited set and URL queue.
"""

from __future__ import annotations

import logging
import time
import urllib.robotparser
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CrawledPage:
    """Represents a single successfully crawled web page."""

    url: str
    """Canonical URL of the page (fragment stripped)."""

    title: str
    """Text content of the <title> element, or empty string."""

    text: str
    """Concatenated visible text of the page body."""

    links: list[str]
    """Absolute, same-domain URLs discovered on this page."""


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

class Crawler:
    """
    Breadth-first web crawler restricted to a single domain.

    Parameters
    ----------
    start_url:
        The seed URL from which crawling begins.
    politeness_delay:
        Minimum seconds to wait between successive HTTP requests (default 6).
    max_retries:
        Maximum number of retry attempts for transient errors (default 3).
    max_pages:
        Hard cap on the number of pages crawled (``None`` = unlimited).
    user_agent:
        User-Agent header sent with every request.
    timeout:
        HTTP request timeout in seconds (default 10).
    """

    DEFAULT_DELAY: float = 6.0
    DEFAULT_RETRIES: int = 3
    DEFAULT_TIMEOUT: int = 10

    def __init__(
        self,
        start_url: str,
        politeness_delay: float = DEFAULT_DELAY,
        max_retries: int = DEFAULT_RETRIES,
        max_pages: int | None = None,
        user_agent: str = "QuoteSearchBot/1.0",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        # Normalise seed URL: strip trailing slash from non-root paths
        parsed_seed = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(start_url)
        seed_path = parsed_seed.path if parsed_seed.path else "/"
        if seed_path != "/" and seed_path.endswith("/"):
            seed_path = seed_path.rstrip("/")
        self.start_url = parsed_seed._replace(path=seed_path).geturl()
        self.politeness_delay = politeness_delay
        self.max_retries = max_retries
        self.max_pages = max_pages
        self.user_agent = user_agent
        self.timeout = timeout

        parsed = urlparse(self.start_url)
        self.base_scheme: str = parsed.scheme
        self.base_netloc: str = parsed.netloc

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})

        self._robot_parser: urllib.robotparser.RobotFileParser | None = None
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(self) -> list[CrawledPage]:
        """
        Crawl the target website and return all discovered pages.

        Returns
        -------
        list[CrawledPage]
            Pages in BFS discovery order.
        """
        self._load_robots_txt()

        visited: set[str] = set()
        queue: deque[str] = deque([self.start_url])
        pages: list[CrawledPage] = []

        logger.info("Starting crawl from %s", self.start_url)

        while queue:
            if self.max_pages is not None and len(pages) >= self.max_pages:
                logger.info("Reached max_pages limit (%d).", self.max_pages)
                break

            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            if not self._is_allowed(url):
                logger.debug("robots.txt disallows %s – skipping.", url)
                continue

            page = self._fetch_with_retry(url)
            if page is None:
                continue

            pages.append(page)
            logger.info("[%d] Crawled: %s", len(pages), url)

            for link in page.links:
                if link not in visited:
                    queue.append(link)

        logger.info("Crawl complete. %d pages collected.", len(pages))
        return pages

    def iter_crawl(self) -> Iterator[CrawledPage]:
        """
        Generator variant of :meth:`crawl` for memory-efficient streaming.

        Yields
        ------
        CrawledPage
            One page at a time in BFS order.
        """
        self._load_robots_txt()

        visited: set[str] = set()
        queue: deque[str] = deque([self.start_url])
        count = 0

        while queue:
            if self.max_pages is not None and count >= self.max_pages:
                break

            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            if not self._is_allowed(url):
                continue

            page = self._fetch_with_retry(url)
            if page is None:
                continue

            count += 1
            for link in page.links:
                if link not in visited:
                    queue.append(link)

            yield page

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_robots_txt(self) -> None:
        """Fetch and parse robots.txt for the target domain."""
        robots_url = f"{self.base_scheme}://{self.base_netloc}/robots.txt"
        self._robot_parser = urllib.robotparser.RobotFileParser(robots_url)
        try:
            self._politeness_sleep()
            self._robot_parser.read()
            logger.debug("robots.txt loaded from %s", robots_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load robots.txt (%s) – proceeding without.", exc)
            self._robot_parser = None

    def _is_allowed(self, url: str) -> bool:
        """Return ``True`` if ``url`` is permitted by robots.txt."""
        if self._robot_parser is None:
            return True
        return self._robot_parser.can_fetch(self.user_agent, url)

    def _politeness_sleep(self) -> None:
        """Block until at least ``politeness_delay`` seconds have elapsed."""
        elapsed = time.monotonic() - self._last_request_time
        wait = self.politeness_delay - elapsed
        if wait > 0:
            logger.debug("Politeness sleep %.2f s", wait)
            time.sleep(wait)

    def _fetch_with_retry(self, url: str) -> CrawledPage | None:
        """
        Fetch *url* with exponential back-off retry on transient errors.

        Returns ``None`` if all retries are exhausted or the error is permanent.
        """
        backoff = self.politeness_delay
        for attempt in range(1, self.max_retries + 1):
            self._politeness_sleep()
            try:
                response = self._session.get(url, timeout=self.timeout)
                self._last_request_time = time.monotonic()

                if response.status_code == 200:
                    return self._parse_page(url, response.text)

                if response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "HTTP %d for %s – attempt %d/%d, retrying in %.0f s.",
                        response.status_code, url, attempt, self.max_retries, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                logger.warning("HTTP %d for %s – skipping.", response.status_code, url)
                return None

            except requests.RequestException as exc:
                logger.warning(
                    "Request error for %s – attempt %d/%d: %s",
                    url, attempt, self.max_retries, exc,
                )
                time.sleep(backoff)
                backoff *= 2

        logger.error("All retries exhausted for %s.", url)
        return None

    def _parse_page(self, url: str, html: str) -> CrawledPage:
        """
        Parse HTML and extract title, visible text, and same-domain links.

        Parameters
        ----------
        url:
            Canonical URL of the page.
        html:
            Raw HTML string.
        """
        soup = BeautifulSoup(html, "html.parser")

        # -- Title ----------------------------------------------------------
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # -- Visible text ---------------------------------------------------
        # Remove script/style before extracting text.
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)

        # -- Links ----------------------------------------------------------
        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href: str = anchor["href"]
            absolute = urljoin(url, href)
            canonical, _ = urldefrag(absolute)  # strip fragment
            canonical = self._normalise_url(canonical)
            if self._is_same_domain(canonical):
                links.append(canonical)

        return CrawledPage(url=url, title=title, text=text, links=links)

    @staticmethod
    def _normalise_url(url: str) -> str:
        """Normalise a URL for deduplication (strip trailing slash from non-root paths)."""
        parsed = urlparse(url)
        path = parsed.path
        if path != '/' and path.endswith('/'):
            path = path.rstrip('/')
        return parsed._replace(path=path).geturl()

    def _is_same_domain(self, url: str) -> bool:
        """Return ``True`` if *url* belongs to the target domain."""
        parsed = urlparse(url)
        return (
            parsed.scheme in ("http", "https")
            and parsed.netloc == self.base_netloc
        )
