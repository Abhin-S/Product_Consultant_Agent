from typing import Literal

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    type: Literal["task"]
    title: str
    description: str
    priority: Literal["low", "medium", "high"]


class NotionDatabaseMetadata(BaseModel):
    name: str = Field(min_length=3, max_length=120)
    idea_description: str = Field(min_length=10, max_length=300)
    risk_level: Literal["Low", "Medium", "High"]
    confidence_score: int = Field(ge=0, le=100)
    tags: list[str] = Field(default_factory=list, max_length=8)


class InsightOutput(BaseModel):
    idea_summary: str
    risks: list[str] = Field(min_length=2, max_length=6)
    opportunities: list[str] = Field(min_length=2, max_length=6)
    recommendations: list[str] = Field(min_length=2, max_length=6)
    actions: list[ActionItem] = Field(min_length=1, max_length=5)
    confidence_score: float = Field(ge=0.0, le=1.0)
    notion_page_content: str = ""
    database_metadata: NotionDatabaseMetadata | None = None