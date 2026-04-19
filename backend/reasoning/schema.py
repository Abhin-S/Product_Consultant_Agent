from typing import Literal

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    type: Literal["task"]
    title: str
    description: str
    priority: Literal["low", "medium", "high"]


class InsightOutput(BaseModel):
    idea_summary: str
    risks: list[str] = Field(min_length=2, max_length=6)
    opportunities: list[str] = Field(min_length=2, max_length=6)
    recommendations: list[str] = Field(min_length=2, max_length=6)
    actions: list[ActionItem] = Field(min_length=1, max_length=5)
    confidence_score: float = Field(ge=0.0, le=1.0)