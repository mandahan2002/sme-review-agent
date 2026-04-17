"""Tests for the SME Review Agent.

Unit tests use mocks and run without an API key.
Integration tests (marked with @pytest.mark.integration) require ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.models.review import Effort, IssueType, Priority, ReviewIssue, ReviewResult, NewContentSuggestion
from src.parsers.dita_parser import DITAParser
from src.parsers.html_parser import HTMLParser

SAMPLE_DIR = Path(__file__).parent / "sample_content"


# ---------------------------------------------------------------------------
# Parser tests (no API calls)
# ---------------------------------------------------------------------------


class TestDITAParser:
    def test_parses_sample_dita(self) -> None:
        content = (SAMPLE_DIR / "sample.dita").read_text()
        result = DITAParser().parse(content)
        assert result.title, "Title should not be empty"
        assert result.word_count > 50
        assert len(result.sections) > 0
        assert result.format == "dita"

    def test_raises_on_invalid_xml(self) -> None:
        with pytest.raises(ValueError, match="Invalid XML"):
            DITAParser().parse("<broken xml")

    def test_extracts_sections(self) -> None:
        dita = """<?xml version="1.0"?>
        <concept id="test">
          <title>Test Concept</title>
          <conbody>
            <section id="s1">
              <title>Section One</title>
              <p>Some content here.</p>
            </section>
            <section id="s2">
              <title>Section Two</title>
              <p>More content here.</p>
            </section>
          </conbody>
        </concept>"""
        result = DITAParser().parse(dita)
        assert result.title == "Test Concept"
        assert any(s["title"] == "Section One" for s in result.sections)


class TestHTMLParser:
    def test_parses_sample_html(self) -> None:
        content = (SAMPLE_DIR / "sample.html").read_text()
        result = HTMLParser().parse(content)
        assert result.title, "Title should not be empty"
        assert result.word_count > 50
        assert len(result.sections) > 0
        assert result.format == "html"

    def test_removes_script_and_style(self) -> None:
        html = """<html><body>
        <script>alert('xss')</script>
        <style>body {color: red}</style>
        <h1>Title</h1>
        <p>Content</p>
        </body></html>"""
        result = HTMLParser().parse(html)
        assert "alert" not in result.full_text
        assert "color: red" not in result.full_text
        assert "Content" in result.full_text

    def test_extracts_sections_by_heading(self) -> None:
        html = """<html><body>
        <h2>Section A</h2><p>Content A</p>
        <h2>Section B</h2><p>Content B</p>
        </body></html>"""
        result = HTMLParser().parse(html)
        titles = [s["title"] for s in result.sections]
        assert "Section A" in titles
        assert "Section B" in titles


# ---------------------------------------------------------------------------
# Review model tests
# ---------------------------------------------------------------------------


class TestReviewModels:
    def test_review_result_serialization(self) -> None:
        issue = ReviewIssue(
            id="ISSUE-001",
            type=IssueType.OUTDATED,
            section="Overview",
            issue="Content refers to SAP ECC which is end-of-life",
            impact="Learners may prepare for deprecated system",
            proposal="Update to SAP S/4HANA equivalent procedures",
            effort=Effort.MEDIUM,
            priority=Priority.HIGH,
        )
        suggestion = NewContentSuggestion(
            topic="SAP S/4HANA Fiori App for Purchase Orders",
            rationale="Modern UI replacement for ME21N",
            affected_audience="MM consultants migrating to S/4HANA",
            recommended_structure="Step-by-step procedure with screenshots",
        )
        result = ReviewResult(
            overall_quality_score=55,
            issues=[issue],
            new_content_suggestions=[suggestion],
            summary="Content needs update for S/4HANA.",
        )
        data = result.model_dump()
        assert data["overall_quality_score"] == 55
        assert data["issues"][0]["type"] == "outdated"
        assert data["issues"][0]["priority"] == "high"

    def test_quality_score_bounds(self) -> None:
        with pytest.raises(Exception):
            ReviewResult(
                overall_quality_score=101,
                issues=[],
                new_content_suggestions=[],
                summary="bad",
            )


# ---------------------------------------------------------------------------
# Review agent unit tests (mocked API)
# ---------------------------------------------------------------------------


def _make_mock_result() -> ReviewResult:
    return ReviewResult(
        overall_quality_score=62,
        issues=[
            ReviewIssue(
                id="ISSUE-001",
                type=IssueType.OUTDATED,
                section="Overview",
                issue="References SAP ECC 6.0 which reaches end of mainstream maintenance in 2027",
                impact="Learners may not realize they should be learning S/4HANA procedures",
                proposal="Add note that S/4HANA uses Manage Purchase Orders Fiori app; update screenshots",
                effort=Effort.MEDIUM,
                priority=Priority.HIGH,
            )
        ],
        new_content_suggestions=[
            NewContentSuggestion(
                topic="Creating Purchase Orders in SAP S/4HANA Fiori",
                rationale="S/4HANA uses the Manage Purchase Orders app (F0842A) instead of ME21N GUI",
                affected_audience="MM consultants, procurement teams migrating to S/4HANA",
                recommended_structure="Parallel procedure: classic ME21N then Fiori app comparison",
            )
        ],
        summary="Content is functional but targets ECC 6.0; needs S/4HANA equivalent procedures.",
    )


class TestReviewAgentUnit:
    def test_review_content_dita(self) -> None:
        mock_response = MagicMock()
        mock_response.parsed_output = _make_mock_result()
        mock_response.stop_reason = "end_turn"

        with patch("src.agents.review_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.parse.return_value = mock_response
            from src.agents.review_agent import review_content

            content = (SAMPLE_DIR / "sample.dita").read_text()
            result = review_content(content, fmt="dita")

        assert isinstance(result, ReviewResult)
        assert 0 <= result.overall_quality_score <= 100
        assert len(result.issues) > 0
        assert result.issues[0].type == IssueType.OUTDATED

    def test_review_content_html(self) -> None:
        mock_response = MagicMock()
        mock_response.parsed_output = _make_mock_result()
        mock_response.stop_reason = "end_turn"

        with patch("src.agents.review_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.parse.return_value = mock_response
            from src.agents.review_agent import review_content

            content = (SAMPLE_DIR / "sample.html").read_text()
            result = review_content(content, fmt="html")

        assert isinstance(result, ReviewResult)

    def test_raises_on_null_output(self) -> None:
        mock_response = MagicMock()
        mock_response.parsed_output = None
        mock_response.stop_reason = "refusal"

        with patch("src.agents.review_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.parse.return_value = mock_response
            from src.agents.review_agent import review_content

            with pytest.raises(RuntimeError, match="no parseable output"):
                review_content("some content", fmt="text")

    def test_auto_format_detection_dita(self) -> None:
        mock_response = MagicMock()
        mock_response.parsed_output = _make_mock_result()

        with patch("src.agents.review_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.parse.return_value = mock_response
            from src.agents.review_agent import review_content

            dita_content = '<?xml version="1.0"?><concept id="x"><title>T</title><conbody/></concept>'
            review_content(dita_content)  # fmt="auto"
            call_kwargs = MockClient.return_value.messages.parse.call_args
            # Should still be called (format was detected)
            assert call_kwargs is not None


# ---------------------------------------------------------------------------
# Integration tests (require ANTHROPIC_API_KEY, skipped otherwise)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestReviewAgentIntegration:
    def test_real_dita_review(self) -> None:
        from src.agents.review_agent import review_content

        content = (SAMPLE_DIR / "sample.dita").read_text()
        result = review_content(content, fmt="dita", context="SAP MM certification course")

        assert isinstance(result, ReviewResult)
        assert 0 <= result.overall_quality_score <= 100
        assert len(result.issues) >= 1, "Should detect at least one issue in sample content"
        for issue in result.issues:
            assert issue.id
            assert issue.issue
            assert issue.impact
            assert issue.proposal

    def test_real_html_review(self) -> None:
        from src.agents.review_agent import review_content

        content = (SAMPLE_DIR / "sample.html").read_text()
        result = review_content(content, fmt="html")

        assert isinstance(result, ReviewResult)
        assert result.summary
