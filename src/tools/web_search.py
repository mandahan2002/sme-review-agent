"""Web search tool — DuckDuckGo HTML (no key) or Brave Search API (optional).

Set BRAVE_SEARCH_API_KEY in .env to use Brave Search, which gives more
reliable results.  Falls back to DuckDuckGo HTML scraping when no key is set.
"""
from __future__ import annotations

import os

import httpx
from bs4 import BeautifulSoup

_DDG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _brave_search(query: str, max_results: int) -> str:
    """Search via Brave Search API (requires BRAVE_SEARCH_API_KEY)."""
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    resp = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max_results, "text_decorations": "false"},
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()

    entries: list[str] = []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        parts = [f"Title: {item.get('title', '')}"]
        if item.get("url"):
            parts.append(f"URL: {item['url']}")
        if item.get("description"):
            parts.append(f"Snippet: {item['description']}")
        entries.append("\n".join(parts))

    return "\n\n---\n\n".join(entries) if entries else f"[No results for: {query!r}]"


def _duckduckgo_search(query: str, max_results: int) -> str:
    """Search via DuckDuckGo HTML scraping (no API key required)."""
    resp = httpx.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query, "kl": "en-us"},
        headers=_DDG_HEADERS,
        timeout=15.0,
        follow_redirects=True,
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    entries: list[str] = []

    for item in soup.select(".result")[:max_results]:
        title_el = item.select_one(".result__title a")
        snippet_el = item.select_one(".result__snippet")

        title = title_el.get_text(strip=True) if title_el else ""
        href = (title_el.get("href") or "") if title_el else ""
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        url = href if href and not href.startswith("//duckduckgo") else ""

        parts: list[str] = []
        if title:
            parts.append(f"Title: {title}")
        if url:
            parts.append(f"URL: {url}")
        if snippet:
            parts.append(f"Snippet: {snippet}")
        if parts:
            entries.append("\n".join(parts))

    return "\n\n---\n\n".join(entries) if entries else f"[No results for: {query!r}]"


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web and return formatted results as plain text.

    Uses Brave Search API when BRAVE_SEARCH_API_KEY is set, otherwise
    falls back to DuckDuckGo HTML scraping.

    Never raises — returns an error string on failure so the agent can
    continue gracefully without search results.
    """
    try:
        if os.environ.get("BRAVE_SEARCH_API_KEY"):
            return _brave_search(query, max_results)
        return _duckduckgo_search(query, max_results)
    except Exception as exc:  # noqa: BLE001
        return f"[Search unavailable: {exc}]"
