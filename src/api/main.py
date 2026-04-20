"""FastAPI application for the SME Review Agent."""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..agents.review_agent import review_content
from ..models.review import ReviewResult

_ALLOWED_EXTENSIONS = {".dita", ".xml", ".html", ".htm", ".txt"}
_EXTENSION_TO_FORMAT = {
    ".dita": "dita",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".txt": "text",
}

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
    """Submit learning content as JSON for AI-powered SME review.

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


@app.post("/review/file", response_model=ReviewResult)
async def review_file(
    file: UploadFile = File(..., description="DITA / XML / HTML / TXTファイル"),
    context: Optional[str] = Form(None, description="追加コンテキスト（対象読者、SAPモジュール名など）"),
) -> ReviewResult:
    """ファイルをアップロードしてAI SMEレビューを実行する。

    対応フォーマット: .dita / .xml / .html / .htm / .txt
    """
    import os
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"対応していない拡張子です: '{ext}'。使用可能: {', '.join(_ALLOWED_EXTENSIONS)}",
        )

    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("utf-8", errors="replace")

    if not content.strip():
        raise HTTPException(status_code=400, detail="ファイルが空です")

    fmt = _EXTENSION_TO_FORMAT.get(ext, "auto")

    try:
        return review_content(
            content=content,
            fmt=fmt,
            context=context or "",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Review failed: {exc}") from exc
