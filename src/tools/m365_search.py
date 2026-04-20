"""Microsoft 365 search via Microsoft Graph Search API.

Searches SharePoint and OneDrive content (learning materials, internal knowledge
base, style guides) and optionally retrieves document text for RAG context.

Required .env variables:
    M365_TENANT_ID      Azure AD tenant ID
    M365_CLIENT_ID      App registration client ID
    M365_CLIENT_SECRET  App registration client secret

Optional:
    M365_SITE_ID        Restrict search to a specific SharePoint site ID
                        (leave blank to search all accessible content)

Azure AD app registration must have:
    Application permission: Sites.Read.All  (for SharePoint)
    Application permission: Files.Read.All  (for OneDrive)
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Token cache (module-level, reused across calls within one process)
# ---------------------------------------------------------------------------

_token_cache: dict[str, Any] = {"token": None, "expires_at": 0.0}


def _get_access_token() -> str:
    """Obtain (or return cached) Azure AD access token via client credentials."""
    if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["token"]  # type: ignore[return-value]

    tenant_id = os.environ["M365_TENANT_ID"]
    client_id = os.environ["M365_CLIENT_ID"]
    client_secret = os.environ["M365_CLIENT_SECRET"]

    resp = httpx.post(
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://graph.microsoft.com/.default",
        },
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()

    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
    return _token_cache["token"]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Optional: fetch plain-text content of a driveItem
# ---------------------------------------------------------------------------

def _fetch_item_text(download_url: str, max_chars: int = 2_000) -> str:
    """Download a driveItem and return the first max_chars of its text."""
    try:
        resp = httpx.get(download_url, timeout=20.0, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")

        # Only attempt to decode text-based formats
        if any(t in content_type for t in ("text/", "application/xml", "application/json")):
            return resp.text[:max_chars]

        # For Word (.docx) — extract raw XML text as a best-effort
        if "officedocument.wordprocessingml" in content_type or resp.url.path.endswith(".docx"):
            import zipfile, io
            from bs4 import BeautifulSoup
            try:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    xml = z.read("word/document.xml")
                soup = BeautifulSoup(xml, "xml")
                return soup.get_text(separator=" ", strip=True)[:max_chars]
            except Exception:
                pass
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

_ENTITY_TYPES = ["driveItem", "listItem"]


def search_m365(query: str, max_results: int = 5) -> str:
    """Search Microsoft 365 (SharePoint / OneDrive) and return formatted results.

    Uses the Microsoft Graph Search API.  Optionally fetches document text
    snippets for driveItems with a public download URL.

    Never raises — returns an error/warning string on failure so the review
    agent can degrade gracefully when M365 is not configured.
    """
    # Check configuration
    required = ("M365_TENANT_ID", "M365_CLIENT_ID", "M365_CLIENT_SECRET")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        return f"[M365 search not configured — missing: {', '.join(missing)}]"

    try:
        token = _get_access_token()
    except Exception as exc:
        return f"[M365 auth failed: {exc}]"

    # Build search request — optionally scoped to a single site
    site_id = os.environ.get("M365_SITE_ID", "")
    request_body: dict[str, Any] = {
        "entityTypes": _ENTITY_TYPES,
        "query": {"queryString": query},
        "size": max_results,
        "fields": ["name", "webUrl", "lastModifiedDateTime", "summary", "description"],
    }
    if site_id:
        request_body["sharePointOneDriveOptions"] = {
            "includeContent": "sharedContent",
        }

    try:
        resp = httpx.post(
            "https://graph.microsoft.com/v1.0/search/query",
            json={"requests": [request_body]},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=20.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return f"[M365 search HTTP error {exc.response.status_code}: {exc.response.text[:200]}]"
    except httpx.HTTPError as exc:
        return f"[M365 search unavailable: {exc}]"

    data = resp.json()

    # Parse hits
    entries: list[str] = []
    try:
        hits_containers = data["value"][0].get("hitsContainers", [])
        for container in hits_containers:
            for hit in container.get("hits", []):
                resource = hit.get("resource", {})
                parts: list[str] = []

                name = resource.get("name") or resource.get("displayName", "")
                web_url = resource.get("webUrl", "")
                modified = resource.get("lastModifiedDateTime", "")[:10]  # date only
                summary = hit.get("summary", "")

                if name:
                    parts.append(f"File: {name}")
                if web_url:
                    parts.append(f"URL: {web_url}")
                if modified:
                    parts.append(f"Last modified: {modified}")
                if summary:
                    parts.append(f"Snippet: {summary}")

                # Best-effort: fetch text content for text/XML files
                download_url = resource.get("@microsoft.graph.downloadUrl", "")
                if download_url and name and any(
                    name.lower().endswith(ext)
                    for ext in (".txt", ".dita", ".xml", ".html", ".htm", ".md", ".docx")
                ):
                    text = _fetch_item_text(download_url)
                    if text:
                        parts.append(f"Content preview:\n{text[:1_000]}")

                if parts:
                    entries.append("\n".join(parts))
    except (KeyError, IndexError, TypeError) as exc:
        return f"[M365 response parse error: {exc}]"

    if not entries:
        return f"[No M365 results for: {query!r}]"

    return "\n\n---\n\n".join(entries)


def m365_configured() -> bool:
    """Return True if all required M365 environment variables are set."""
    return all(os.environ.get(k) for k in ("M365_TENANT_ID", "M365_CLIENT_ID", "M365_CLIENT_SECRET"))
