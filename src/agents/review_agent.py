"""
SME Review Agent — Claude-powered content quality analysis.

Two-phase approach:
  1. Research phase  — claude-haiku-4-5 uses search_web tool to look up SAP Notes,
                       current product status, and fact-check specific claims.
  2. Review phase    — claude-opus-4-7 uses the research findings + adaptive
                       thinking to produce a structured ReviewResult.
"""
from __future__ import annotations

import os
from typing import Any

import anthropic
from dotenv import load_dotenv

from ..models.review import ReviewResult
from ..parsers.base_parser import ParsedContent
from ..parsers.dita_parser import DITAParser
from ..parsers.html_parser import HTMLParser
from ..tools.web_search import search_web

load_dotenv()

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_RESEARCH_SYSTEM = """\
You are a technical fact-checker for SAP learning content.

Your job is to search the web for information that will help verify the accuracy
of SAP content under review.  Focus on:
  • SAP Notes cited or implied (e.g. "SAP Note 1234567")
  • Transaction codes (T-codes) and whether they have Fiori replacements in S/4HANA
  • Product version status (e.g. SAP ECC end-of-maintenance dates)
  • SAP Help documentation for the specific feature or procedure described

Be targeted: perform 2–4 high-value searches, then write a concise research summary
(bullet points) that the reviewer can use to spot inaccuracies in the content.
"""

_REVIEW_SYSTEM = """\
You are an expert SME (Subject Matter Expert) Review Agent specializing in SAP \
learning content quality assurance.

Your role is to analyze educational content for SAP products and identify \
actionable improvements.

## Issue Categories

- **OUTDATED**: Content referring to deprecated features, old UI patterns, \
superseded transactions, or removed functionality (e.g. SAP GUI steps now \
replaced by Fiori apps, T-codes replaced by apps, Classic UI vs. SAP Fiori)
- **INCORRECT**: Factually wrong statements, wrong menu paths, incorrect \
business logic, or erroneous procedures
- **MISSING**: Important topics, prerequisites, or steps that are absent but \
required for learner success
- **INCONSISTENCY**: Internal contradictions, conflicting procedures, or steps \
that contradict each other within the same content
- **SUGGESTION**: Improvements to clarity, structure, or completeness that \
would enhance learning effectiveness

## Review Standards

For each issue, provide:
1. **Issue**: Precise description with the problematic text or section
2. **Impact**: Concrete learner or business consequence
3. **Proposal**: Specific, actionable fix (not vague guidance)
4. **Effort**: low (< 1 hour) / medium (1–4 hours) / high (> 4 hours)
5. **Priority**: critical / high / medium / low

## Quality Score

Rate overall content quality 0–100:
- 90–100: Excellent, minor polish only
- 70–89: Good, some targeted improvements needed
- 50–69: Adequate, several important gaps
- 30–49: Needs significant work
- 0–29: Major revision required

Be thorough, specific, and constructive. Every issue and suggestion must be actionable.\
"""

# ---------------------------------------------------------------------------
# Tool schema for Claude
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_web",
        "description": (
            "Search the web for up-to-date information about SAP products, "
            "SAP Notes, SAP transactions, SAP Fiori apps, or SAP Help documentation. "
            "Use this to verify technical accuracy of the content being reviewed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Specific search query. "
                        "Examples: 'SAP Note 3456789', "
                        "'SAP S/4HANA ME21N Fiori replacement app', "
                        "'SAP ECC 6.0 end of mainstream maintenance date'"
                    ),
                }
            },
            "required": ["query"],
        },
    }
]

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
    return ParsedContent(
        title="Document",
        full_text=content,
        sections=[{"title": "Content", "content": content}],
        format="text",
    )


# ---------------------------------------------------------------------------
# Phase 1: Research with Tool Use (claude-haiku-4-5, fast + cheap)
# ---------------------------------------------------------------------------


def _research_phase(
    client: anthropic.Anthropic,
    parsed: ParsedContent,
    context: str,
) -> str:
    """Run a tool-use agentic loop to gather web research for the review.

    Returns a plain-text research summary, or an empty string on failure.
    """
    research_prompt = (
        f"Please research the following SAP learning content to help verify its accuracy.\n\n"
        f"**Title**: {parsed.title}\n"
        f"**Format**: {parsed.format}\n"
        + (f"**Context**: {context}\n" if context else "")
        + f"\n**Content excerpt** (first 3 000 chars):\n\n{parsed.full_text[:3_000]}\n\n"
        "Search for SAP Notes, current product/transaction status, Fiori equivalents, "
        "and any other specific claims that need verification. "
        "After searching, write a concise research summary."
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": research_prompt}]

    try:
        for _ in range(8):  # safety cap — max ~4 search round-trips
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=3_000,
                system=_RESEARCH_SYSTEM,
                messages=messages,
                tools=_TOOLS,
            )

            if resp.stop_reason == "end_turn":
                texts = [b.text for b in resp.content if hasattr(b, "text")]
                return "\n".join(texts).strip()

            if resp.stop_reason == "tool_use":
                tool_results: list[dict[str, Any]] = []
                for block in resp.content:
                    if block.type == "tool_use" and block.name == "search_web":
                        query = block.input.get("query", "")
                        result = search_web(query)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result[:2_500],
                            }
                        )

                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": tool_results})

    except Exception:  # noqa: BLE001 — research failure must not abort the review
        pass

    return ""


# ---------------------------------------------------------------------------
# Phase 2: Structured review (claude-opus-4-7, adaptive thinking)
# ---------------------------------------------------------------------------


def _review_phase(
    client: anthropic.Anthropic,
    parsed: ParsedContent,
    context: str,
    research_summary: str,
) -> ReviewResult:
    sections_preview = "\n".join(
        f"  - {s['title']}: {s['content'][:300]}{'...' if len(s['content']) > 300 else ''}"
        for s in parsed.sections[:12]
    )

    research_block = (
        f"\n\n---\n\n**Web Research Findings** (use these to verify accuracy):\n\n"
        f"{research_summary}"
        if research_summary
        else ""
    )

    user_message = (
        f"Please review the following SAP learning content.\n\n"
        f"**Title**: {parsed.title}\n"
        f"**Format**: {parsed.format}\n"
        f"**Word Count**: {parsed.word_count}\n"
        + (f"**Additional Context**: {context}\n" if context else "")
        + f"\n---\n\n**Full Content**:\n\n{parsed.full_text[:10_000]}\n\n"
        f"---\n\n**Section Overview**:\n{sections_preview}"
        + research_block
        + "\n\nIdentify all quality issues and provide structured improvement suggestions."
    )

    response = client.messages.parse(
        model="claude-opus-4-7",
        max_tokens=8_000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": _REVIEW_SYSTEM,
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


# ---------------------------------------------------------------------------
# Public API
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

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    research_summary = _research_phase(client, parsed, context)
    return _review_phase(client, parsed, context, research_summary)
