from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Effort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IssueType(str, Enum):
    OUTDATED = "outdated"
    INCORRECT = "incorrect"
    MISSING = "missing"
    INCONSISTENCY = "inconsistency"
    SUGGESTION = "suggestion"


class ReviewIssue(BaseModel):
    id: str = Field(description="Unique identifier e.g. ISSUE-001")
    type: IssueType = Field(description="Category of the issue")
    section: str = Field(description="Section or topic where the issue was found")
    issue: str = Field(description="Clear description of the problem")
    impact: str = Field(description="Learner or business impact if not addressed")
    proposal: str = Field(description="Specific actionable recommendation")
    effort: Effort = Field(description="Estimated implementation effort")
    priority: Priority = Field(description="Priority for addressing this issue")


class NewContentSuggestion(BaseModel):
    topic: str = Field(description="Topic that should be added")
    rationale: str = Field(description="Why this content is needed")
    affected_audience: str = Field(description="Who needs this content")
    recommended_structure: str = Field(description="Suggested structure or format")


class ReviewResult(BaseModel):
    overall_quality_score: int = Field(
        ge=0, le=100, description="Overall content quality score from 0 to 100"
    )
    issues: List[ReviewIssue] = Field(description="Identified issues requiring attention")
    new_content_suggestions: List[NewContentSuggestion] = Field(
        description="Topics that should be added to the content"
    )
    summary: str = Field(description="Executive summary of the review findings")
