"""
SME Review Agent — Claude-powered content quality analysis.

Uses claude-opus-4-7 with adaptive thinking and prompt caching to review
SAP learning content (DITA/XML/HTML) and return structured feedback.
"""
from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

from ..models.review import ReviewResult
from ..parsers.base_parser import ParsedContent
from ..parsers.dita_parser import DITAParser
from ..parsers.html_parser import HTMLParser

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt — cached on every call (ephemeral 5-min TTL)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert SME (Subject Matter Expert) Review Agent specializing in SAP learning content quality assurance.

Your role is to analyze educational content for SAP products and identify actionable improvements.

## Issue Categories

- **OUTDATED**: Content referring to deprecated features, old UI patterns, superseded transactions, or removed functionality (e.g., SAP GUI steps that are now Fiori apps, old T-codes replaced by apps, Classic UI vs. SAP Fiori)
- **INCORRECT**: Factually wrong statements, wrong menu paths, incorrect business logic, or erroneous procedures
- **MISSING**: Important topics, prerequisites, or steps that are absent but required for learner success
- **INCONSISTENCY**: Internal contradictions, conflicting procedures, or steps that contradict each other within the same content
- **SUGGESTION**: Improvements to clarity, structure, or completeness that would enhance learning effectiveness

## Review Standards

For each issue, provide:
1. **Issue**: Precise description with the problematic text or section
2. **Impact**: Concrete learner or business consequence (e.g., "Learners will fail the certification exam", "Incorrect procedure may cause data loss")
3. **Proposal**: Specific, actionable fix (not vague guidance)
4. **Effort**: low (< 1 hour) / medium (1–4 hours) / high (> 4 hours)
5. **Priority**: critical (blocks learning) / high (significant gap) / medium (notable issue) / low (minor improvement)

## New Content Suggestions

Identify topics that should be added because they are:
- Required for complete understanding
- Covered in recent SAP releases but absent from the content
- Frequently asked about by learners or SMEs
- Necessary for exam alignment

## Quality Score

Rate the overall content quality from 0–100:
- 90–100: Excellent, minor polish only
- 70–89: Good, some targeted improvements needed
- 50–69: Adequate, several important gaps
- 30–49: Needs significant work
- 0–29: Major revision required

Be thorough, specific, and constructive. Every issue and suggestion must be actionable."""


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(content: str) -> str:
    stripped = content.strip()
    if (
        stripped.startswith("<?xml")
        or "<!DOCTYPE" in stripped[:200]
        or "<topic" in stripped
        or "<concept" in stripped
        or "<task" in stripped
        or "<reference" in stripped
    ):
        return "dita"
    if (
        stripped.lower().startswith("<!doctype html")
        or stripped.startswith("<html")
        or "<body" in stripped
        or "<head" in stripped
    ):
        return "html"
    if stripped.startswith("<"):
        return "xml"
    return "text"


# ---------------------------------------------------------------------------
# Content parsing
# ---------------------------------------------------------------------------

def _parse_content(content: str, fmt: str) -> ParsedContent:
    if fmt in ("dita", "xml"):
        return DITAParser().parse(content)
    if fmt == "html":
        return HTMLParser().parse(content)
    # Plain-text fallback
    return ParsedContent(
        title="Document",
        full_text=content,
        sections=[{"title": "Content", "content": content}],
        format="text",
    )


# ---------------------------------------------------------------------------
# Main review function
# ---------------------------------------------------------------------------

def review_content(
    content: str,
    fmt: str = "auto",
    context: str = "",
) -> ReviewResult:
    """Review learning content and return a structured ReviewResult.

    Args:
        content: Raw content (DITA / XML / HTML / plain text).
        fmt: Format hint. ``"auto"`` detects automatically.
        context: Optional context such as target audience or SAP module name.

    Returns:
        :class:`~src.models.review.ReviewResult` with issues and suggestions.

    Raises:
        ValueError: If content cannot be parsed.
        RuntimeError: If the AI model returns no parseable output.
    """
    if fmt == "auto":
        fmt = _detect_format(content)

    parsed = _parse_content(content, fmt)

    # Compose the review request -----------------------------------------------
    sections_preview = "\n".join(
        f"  - {s['title']}: {s['content'][:300]}{'...' if len(s['content']) > 300 else ''}"
        for s in parsed.sections[:12]
    )

    user_message = (
        f"Please review the following SAP learning content.\n\n"
        f"**Title**: {parsed.title}\n"
        f"**Format**: {parsed.format}\n"
        f"**Word Count**: {parsed.word_count}\n"
        + (f"**Additional Context**: {context}\n" if context else "")
        + f"\n---\n\n**Full Content**:\n\n{parsed.full_text[:10_000]}\n\n"
        f"---\n\n**Section Overview**:\n{sections_preview}\n\n"
        "Identify all quality issues and provide structured improvement suggestions."
    )

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.parse(
        model="claude-opus-4-7",
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
        output_format=ReviewResult,
    )

    result = response.parsed_output
    if result is None:
        raise RuntimeError(
            f"Model returned no parseable output (stop_reason={response.stop_reason})"
        )
    return result
