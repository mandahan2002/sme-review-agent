"""SAP Help Portal search via help.sap.com public search API.

No authentication required.  Searches official SAP product documentation,
procedural guides, and release notes across all SAP products.
"""
from __future__ import annotations

import httpx

_BASE_URL = "https://help.sap.com/docs/search"
_HEADERS = {
    "User-Agent": "SMEReviewAgent/1.0 (SAP learning content reviewer)",
    "Accept": "application/json",
}


def search_sap_help(query: str, max_results: int = 5) -> str:
    """Search SAP Help Portal and return formatted results.

    Queries help.sap.com's public search index which covers:
      - SAP S/4HANA, ECC, BTP, Fiori, and all SAP product documentation
      - Procedural guides, configuration guides, release notes
      - SAP API documentation

    Never raises — returns an error string on failure.
    """
    try:
        resp = httpx.get(
            _BASE_URL,
            params={
                "q": query,
                "state": "PRODUCTION",
                "language": "en-US",
                "version": "latest",
            },
            headers=_HEADERS,
            timeout=15.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        return f"[SAP Help Portal unavailable: {exc}]"
    except Exception as exc:
        return f"[SAP Help Portal error: {exc}]"

    return _format_results(data, query, max_results)


def _format_results(data: dict, query: str, max_results: int) -> str:
    """Parse API response into a readable string for the LLM."""
    # The API wraps results under data.hits or hits depending on version
    hits = (
        data.get("data", {}).get("hits")
        or data.get("hits")
        or []
    )

    if not hits:
        return f"[No SAP Help Portal results for: {query!r}]"

    entries: list[str] = []
    for hit in hits[:max_results]:
        parts: list[str] = []

        title = hit.get("title") or hit.get("name", "")
        url = hit.get("url") or hit.get("link", "")
        product = hit.get("deliverable") or hit.get("product", "")
        version = hit.get("version", "")
        # Highlighted snippet — may be a list or string
        raw_snippet = (
            hit.get("highlight", {}).get("body")
            or hit.get("snippet")
            or hit.get("description")
            or ""
        )
        snippet = (
            " … ".join(raw_snippet) if isinstance(raw_snippet, list) else raw_snippet
        )

        if title:
            parts.append(f"Title: {title}")
        if product:
            label = f"Product: {product}"
            if version:
                label += f" ({version})"
            parts.append(label)
        if url:
            parts.append(f"URL: https://help.sap.com{url}" if url.startswith("/") else f"URL: {url}")
        if snippet:
            parts.append(f"Snippet: {snippet[:400]}")

        if parts:
            entries.append("\n".join(parts))

    if not entries:
        return f"[No SAP Help Portal results for: {query!r}]"

    return "\n\n---\n\n".join(entries)
