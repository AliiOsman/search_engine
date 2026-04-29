# Quote Search Engine

A command-line search tool that crawls [quotes.toscrape.com](https://quotes.toscrape.com/), builds a TF-IDF ranked inverted index, and allows you to search for words and phrases across all crawled pages.

---

## Project Overview and Purpose

This tool was built to demonstrate core information retrieval concepts including web crawling, inverted indexing, and ranked search. It:

- **Crawls** quotes.toscrape.com using a breadth-first search strategy, respecting a 6-second politeness window between requests
- **Indexes** every word found across all pages, storing frequency and positional data for each term
- **Ranks** search results using TF-IDF scoring (terms that appear often in a page but rarely across all pages rank highest)
- **Supports phrase search** by matching terms that appear consecutively in the original text
- **Suggests corrections** for misspelled search terms using Damerau-Levenshtein edit distance

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `requests` | >= 2.31.0 | Making HTTP requests to crawl pages |
| `beautifulsoup4` | >= 4.12.0 | Parsing HTML to extract text and links |

All other functionality uses Python's standard library (`json`, `re`, `collections`, `urllib`, `unittest`).

---

## Installation and Setup

**Requirements:** Python 3.10 or higher

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

**Step 2 — Create a virtual environment (recommended):**
```bash
python -m venv .venv

# Mac/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

**Step 3 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — Start the program:**
```bash
python src/main.py
```

You will see the interactive shell prompt `>` where you can enter commands.

---

## Usage

### Starting the program

```bash
python src/main.py
```

Optional arguments:

```bash
python src/main.py --max-pages 50      # limit pages crawled (recommended for speed)
python src/main.py --delay 6           # politeness delay in seconds (minimum 6)
python src/main.py --index data/index.json  # custom index file path
python src/main.py --log-level DEBUG   # show detailed logging
```

---

### Command 1 — `build`

Crawls the website, builds the inverted index, and saves it to disk. This only needs to be run once — after that you can use `load`.

```
> build
```

Example output:
```
Starting crawl of https://quotes.toscrape.com/ …
Politeness delay: 6.0s | Max pages: 50
This may take several minutes. Please wait.

07:43:40 [INFO] crawler: Starting crawl from https://quotes.toscrape.com/
07:43:40 [INFO] crawler: [1] Crawled:  https://quotes.toscrape.com/
07:43:46 [INFO] crawler: [2] Crawled: https://quotes.toscrape.com/login
07:43:52 [INFO] crawler: [3] Crawled: https://quotes.toscrape.com/author/Albert-Einstein
...

Crawled 50 pages. Building index …
07:48:44 [INFO] indexer: Index saved to data/index.json (1653 terms).

✓ Index built in 304.2s — 50 documents, 1653 unique terms, 11,343 total tokens.
  Saved to: data/index.json
```

> **Note:** With the default 6-second politeness delay the build command takes several minutes. Use `--max-pages 50` to limit crawling to the main pages only.

---

### Command 2 — `load`

Loads a previously built index from disk. Much faster than rebuilding — use this every time after the first `build`.

```
> load
```

Example output:
```
Loading index from data/index.json …
07:49:33 [INFO] indexer: Index loaded from data/index.json (1653 terms, 50 docs).
✓ Index loaded in 0.02s — 50 documents, 1653 unique terms.
```

> **Note:** You must run `build` at least once before `load` will work.

---

### Command 3 — `print <term>`

Prints the full inverted index entry for a single word, showing every page it appears on along with its frequency, TF-IDF score, and token positions.

```
> print nonsense
```

Example output:
```
Postings for 'nonsense' (3 document(s)):
------------------------------------------------------------
  URL      : https://quotes.toscrape.com/tag/life/page/1
  Title    : Quotes to Scrape
  Frequency: 1
  TF-IDF   : 2.1055
  Positions: [473]

  URL      : https://quotes.toscrape.com/page/2
  Title    : Quotes to Scrape
  Frequency: 1
  TF-IDF   : 2.1055
  Positions: [417]

  URL      : https://quotes.toscrape.com/tag/life
  Title    : Quotes to Scrape
  Frequency: 1
  TF-IDF   : 2.1055
  Positions: [473]
```

---

### Command 4 — `find <query>`

Searches the index and returns all pages containing the query terms, ranked by TF-IDF score.

**Single word search:**
```
> find indifference
```

**Multi-word search** (returns pages containing ALL terms):
```
> find good friends
```

**Phrase search** (wrap in quotes to match consecutive words in exact order):
```
> find "the world"
```

Example output:
```
Search results for 'good friends' [keyword mode] — 18 result(s) (0.17 ms)
======================================================================

  [1] Quotes to Scrape
       URL   : https://quotes.toscrape.com/tag/friends/page/1
       Score : 4.2599  |  Freq: 3

  [2] Quotes to Scrape
       URL   : https://quotes.toscrape.com/tag/friends
       Score : 4.2599  |  Freq: 3
    ...

```

If no results are found, the engine suggests corrections automatically:
```
> find beautifull
Search results for 'beautifull' [keyword mode] — 0 result(s) (23.15 ms)
======================================================================
  No pages found.

  Close! Spelling's hard sometimes... perhaps you meant:'beautiful'?
```

---
## Testing

Run the test script from the root of the repo:

**Mac/Linux:**
```bash
./run_tests.sh
```

**Windows:**
```bash
bash run_tests.sh
```

This will automatically:
1. Install the `coverage` package if not already installed
2. Run all 117 tests across all three test files
3. Print a coverage report showing any untested lines
4. Generate a visual HTML report — open `htmlcov/index.html` in your browser to see line-by-line coverage

### Run a specific test file manually

```bash
python -m unittest tests/test_crawler.py -v
python -m unittest tests/test_indexer.py -v
python -m unittest tests/test_search.py -v
```

### What the tests cover

| File | Tests | Areas covered |
|---|---|---|
| `test_crawler.py` | 30 | Domain filtering, HTML parsing, politeness delay, retry logic, robots.txt, BFS deduplication |
| `test_indexer.py` | 54 | Tokenisation, TF-IDF correctness, boolean AND search, phrase search, spell suggestions, save/load round-trip |
| `test_search.py` | 32 | Keyword and phrase search, auto phrase detection, result ranking, formatting, edge cases |

All tests use mocks — **no live network requests are made during testing.**

---

## Project Structure

```
├── src/
│   ├── crawler.py      # BFS web crawler with politeness window and retry logic
│   ├── indexer.py      # Inverted index with TF-IDF scoring and phrase search
│   ├── search.py       # Search facade handling queries and result formatting
│   └── main.py         # Interactive CLI shell
├── tests/
│   ├── test_crawler.py
│   ├── test_indexer.py
│   └── test_search.py
├── data/
│   └── index.json      # Generated after running build (not committed to git)
├── requirements.txt
├── run_tests.sh
└── README.md
```