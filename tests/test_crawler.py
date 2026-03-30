"""
test_crawler.py - Unit and integration tests for crawler.py.

Coverage targets:
- URL normalisation and deduplication
- Same-domain filtering
- HTML parsing (title, text, links)
- Politeness delay enforcement
- Retry logic on HTTP errors (429, 500)
- robots.txt compliance
- Graceful handling of network errors
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from urllib.parse import urlparse

import requests

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crawler import Crawler, CrawledPage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int = 200, text: str = "") -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    return resp


SIMPLE_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
  <p>Hello world. This is a test page.</p>
  <a href="/page2">Page Two</a>
  <a href="https://external.com/page">External</a>
  <a href="/page2#section">Fragment duplicate</a>
</body>
</html>
"""

QUOTES_HTML = """
<html>
<head><title>Quotes Page</title></head>
<body>
  <div class="quote">
    <span class="text">"The world is a book."</span>
    <small class="author">Augustine</small>
  </div>
  <a href="/page/2/">Next page</a>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestCrawlerDomainFilter(unittest.TestCase):
    """Tests for _is_same_domain helper."""

    def setUp(self) -> None:
        self.crawler = Crawler("https://quotes.toscrape.com/")

    def test_same_domain_http(self) -> None:
        self.assertTrue(self.crawler._is_same_domain("https://quotes.toscrape.com/page/2/"))

    def test_same_domain_subpath(self) -> None:
        self.assertTrue(self.crawler._is_same_domain("https://quotes.toscrape.com/author/Einstein/"))

    def test_external_domain_rejected(self) -> None:
        self.assertFalse(self.crawler._is_same_domain("https://external.com/"))

    def test_ftp_scheme_rejected(self) -> None:
        self.assertFalse(self.crawler._is_same_domain("ftp://quotes.toscrape.com/"))

    def test_subdomain_rejected(self) -> None:
        self.assertFalse(self.crawler._is_same_domain("https://sub.quotes.toscrape.com/"))


class TestCrawlerParsePage(unittest.TestCase):
    """Tests for _parse_page HTML extraction."""

    def setUp(self) -> None:
        self.crawler = Crawler("https://quotes.toscrape.com/")

    def test_title_extracted(self) -> None:
        page = self.crawler._parse_page("https://quotes.toscrape.com/", SIMPLE_HTML)
        self.assertEqual(page.title, "Test Page")

    def test_text_extracted(self) -> None:
        page = self.crawler._parse_page("https://quotes.toscrape.com/", SIMPLE_HTML)
        self.assertIn("Hello world", page.text)

    def test_script_excluded_from_text(self) -> None:
        html = "<html><body><script>var x=1;</script><p>Clean text</p></body></html>"
        page = self.crawler._parse_page("https://quotes.toscrape.com/", html)
        self.assertNotIn("var x", page.text)
        self.assertIn("Clean text", page.text)

    def test_style_excluded_from_text(self) -> None:
        html = "<html><body><style>.red{color:red}</style><p>Visible</p></body></html>"
        page = self.crawler._parse_page("https://quotes.toscrape.com/", html)
        self.assertNotIn("color:red", page.text)

    def test_same_domain_links_included(self) -> None:
        page = self.crawler._parse_page("https://quotes.toscrape.com/", SIMPLE_HTML)
        self.assertIn("https://quotes.toscrape.com/page2", page.links)

    def test_external_links_excluded(self) -> None:
        page = self.crawler._parse_page("https://quotes.toscrape.com/", SIMPLE_HTML)
        self.assertNotIn("https://external.com/page", page.links)

    def test_fragment_stripped_from_links(self) -> None:
        page = self.crawler._parse_page("https://quotes.toscrape.com/", SIMPLE_HTML)
        # /page2#section should be normalised to /page2 (already in list, not duplicated)
        fragment_urls = [l for l in page.links if "#" in l]
        self.assertEqual(fragment_urls, [], "Fragment URLs should be stripped")

    def test_missing_title_returns_empty_string(self) -> None:
        html = "<html><body><p>No title here</p></body></html>"
        page = self.crawler._parse_page("https://quotes.toscrape.com/", html)
        self.assertEqual(page.title, "")

    def test_url_stored_on_page(self) -> None:
        url = "https://quotes.toscrape.com/author/Einstein/"
        page = self.crawler._parse_page(url, SIMPLE_HTML)
        self.assertEqual(page.url, url)

    def test_relative_links_made_absolute(self) -> None:
        page = self.crawler._parse_page("https://quotes.toscrape.com/", QUOTES_HTML)
        self.assertIn("https://quotes.toscrape.com/page/2", page.links)


class TestCrawlerPoliteness(unittest.TestCase):
    """Tests for politeness delay enforcement."""

    @patch("crawler.time.sleep")
    def test_politeness_sleep_called(self, mock_sleep: MagicMock) -> None:
        crawler = Crawler("https://quotes.toscrape.com/", politeness_delay=6.0)
        # Set last_request_time to now so a full delay is needed
        crawler._last_request_time = time.monotonic()
        crawler._politeness_sleep()
        mock_sleep.assert_called_once()
        sleep_duration = mock_sleep.call_args[0][0]
        self.assertAlmostEqual(sleep_duration, 6.0, delta=0.1)

    @patch("crawler.time.sleep")
    def test_no_sleep_if_delay_already_passed(self, mock_sleep: MagicMock) -> None:
        crawler = Crawler("https://quotes.toscrape.com/", politeness_delay=1.0)
        crawler._last_request_time = time.monotonic() - 5.0  # 5 seconds ago
        crawler._politeness_sleep()
        mock_sleep.assert_not_called()


class TestCrawlerRetry(unittest.TestCase):
    """Tests for retry logic on transient HTTP errors."""

    def setUp(self) -> None:
        self.crawler = Crawler(
            "https://quotes.toscrape.com/",
            politeness_delay=0,
            max_retries=3,
        )
        # Bypass real sleep
        self.crawler._politeness_sleep = MagicMock()

    @patch("requests.Session.get")
    def test_retries_on_500(self, mock_get: MagicMock) -> None:
        """Should retry on 500, then return page on 200."""
        mock_get.side_effect = [
            _make_response(500),
            _make_response(200, SIMPLE_HTML),
        ]
        with patch("crawler.time.sleep"):
            page = self.crawler._fetch_with_retry("https://quotes.toscrape.com/")
        self.assertIsNotNone(page)
        self.assertEqual(page.title, "Test Page")

    @patch("requests.Session.get")
    def test_returns_none_after_max_retries(self, mock_get: MagicMock) -> None:
        """Should give up after max_retries and return None."""
        mock_get.return_value = _make_response(500)
        with patch("crawler.time.sleep"):
            page = self.crawler._fetch_with_retry("https://quotes.toscrape.com/")
        self.assertIsNone(page)

    @patch("requests.Session.get")
    def test_permanent_404_not_retried(self, mock_get: MagicMock) -> None:
        """404 is a permanent error; should not retry."""
        mock_get.return_value = _make_response(404)
        with patch("crawler.time.sleep"):
            page = self.crawler._fetch_with_retry("https://quotes.toscrape.com/")
        self.assertIsNone(page)
        self.assertEqual(mock_get.call_count, 1)  # called exactly once

    @patch("requests.Session.get")
    def test_network_exception_retried(self, mock_get: MagicMock) -> None:
        """ConnectionError should trigger retry."""
        mock_get.side_effect = [
            requests.ConnectionError("Connection refused"),
            _make_response(200, SIMPLE_HTML),
        ]
        with patch("crawler.time.sleep"):
            page = self.crawler._fetch_with_retry("https://quotes.toscrape.com/")
        self.assertIsNotNone(page)


class TestCrawlerRobots(unittest.TestCase):
    """Tests for robots.txt compliance."""

    def test_allowed_returns_true_when_no_robot_parser(self) -> None:
        crawler = Crawler("https://quotes.toscrape.com/")
        crawler._robot_parser = None
        self.assertTrue(crawler._is_allowed("https://quotes.toscrape.com/anything"))

    def test_disallowed_url_rejected(self) -> None:
        crawler = Crawler("https://quotes.toscrape.com/")
        mock_parser = MagicMock()
        mock_parser.can_fetch.return_value = False
        crawler._robot_parser = mock_parser
        self.assertFalse(crawler._is_allowed("https://quotes.toscrape.com/secret/"))

    def test_allowed_url_accepted(self) -> None:
        crawler = Crawler("https://quotes.toscrape.com/")
        mock_parser = MagicMock()
        mock_parser.can_fetch.return_value = True
        crawler._robot_parser = mock_parser
        self.assertTrue(crawler._is_allowed("https://quotes.toscrape.com/"))


class TestCrawlerBFS(unittest.TestCase):
    """Integration-style tests for BFS traversal and deduplication."""

    def _make_crawler(self) -> Crawler:
        crawler = Crawler("https://quotes.toscrape.com/", politeness_delay=0)
        crawler._politeness_sleep = MagicMock()
        crawler._load_robots_txt = MagicMock()
        crawler._is_allowed = MagicMock(return_value=True)
        return crawler

    def test_deduplication_prevents_revisit(self) -> None:
        """A URL discovered multiple times should only be fetched once."""
        html_start = (
            '<html><head><title>Start</title></head>'
            '<body><a href="/page2/">P2</a></body></html>'
        )
        html_p2 = (
            '<html><head><title>Page 2</title></head>'
            '<body><a href="/">Start</a></body></html>'
        )
        crawler = self._make_crawler()
        crawler._session = MagicMock()
        crawler._session.get.side_effect = [
            _make_response(200, html_start),
            _make_response(200, html_p2),
        ]
        pages = crawler.crawl()
        self.assertEqual(len(pages), 2)
        # Each unique URL fetched exactly once (2 total, cycle broken by visited set)
        self.assertEqual(crawler._session.get.call_count, 2)

    def test_max_pages_respected(self) -> None:
        """Crawl stops after max_pages even if queue is non-empty."""
        html = (
            '<html><head><title>P</title></head>'
            '<body><a href="/p1/">P1</a><a href="/p2/">P2</a></body></html>'
        )
        crawler = self._make_crawler()
        crawler._session = MagicMock()
        crawler._session.get.return_value = _make_response(200, html)
        crawler.max_pages = 1
        pages = crawler.crawl()
        self.assertEqual(len(pages), 1)

    def test_crawl_returns_crawled_page_objects(self) -> None:
        crawler = self._make_crawler()
        crawler._session = MagicMock()
        crawler._session.get.return_value = _make_response(200, SIMPLE_HTML)
        crawler.max_pages = 1
        pages = crawler.crawl()
        self.assertIsInstance(pages[0], CrawledPage)
        self.assertEqual(pages[0].title, "Test Page")


# ---------------------------------------------------------------------------
# CrawledPage dataclass
# ---------------------------------------------------------------------------

class TestCrawledPage(unittest.TestCase):
    def test_fields_stored_correctly(self) -> None:
        page = CrawledPage(
            url="https://example.com",
            title="Example",
            text="Some text",
            links=["https://example.com/about"],
        )
        self.assertEqual(page.url, "https://example.com")
        self.assertEqual(page.title, "Example")
        self.assertIn("https://example.com/about", page.links)


if __name__ == "__main__":
    unittest.main(verbosity=2)
