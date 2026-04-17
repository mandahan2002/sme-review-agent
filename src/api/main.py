"""FastAPI application for the SME Review Agent."""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..agents.review_agent import review_content
from ..models.review import ReviewResult

app = FastAPI(
    title="SME Review Agent",
    description="AI-powered SAP learning content quality review",
    version="1.0.0",
)


class ReviewRequest(BaseModel):
    content: str
    format: str = "auto"
    context: Optional[str] = None


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "SME Review Agent"}


@app.post("/review", response_model=ReviewResult)
def review(request: ReviewRequest) -> ReviewResult:
    """Submit learning content for AI-powered SME review.

    Accepts DITA, XML, HTML, or plain text.
    Returns structured review with issues and improvement suggestions.
    """
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")

    try:
        return review_content(
            content=request.content,
            fmt=request.format,
            context=request.context or "",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Review failed: {exc}") from exc
