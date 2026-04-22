from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ActionItem(BaseModel):
    type: Literal["task"]
    title: str
    description: str
    priority: Literal["low", "medium", "high"]
    decision_type: Literal[
        "positioning",
        "differentiation",
        "messaging",
        "trust",
        "audience",
        "pricing",
        "narrative",
        "other",
    ] = "other"
    impact: Literal["low", "medium", "high"] = "medium"


class NotionDatabaseMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str | None = Field(default=None, min_length=3, max_length=120)
    brand_positioning: str | None = Field(
        default=None,
        min_length=10,
        max_length=300,
        validation_alias=AliasChoices("brand_positioning", "idea_description"),
    )
    brand_risk_level: Literal["Low", "Medium", "High"] | None = Field(
        default=None,
        validation_alias=AliasChoices("brand_risk_level", "risk_level")
    )
    confidence_score: int | None = Field(default=None, ge=0, le=100)
    tags: list[str] | None = Field(default=None, max_length=8)


class InsightOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    abstention_message: str | None = None
    brand_diagnosis: str | None = Field(
        default=None,
        validation_alias=AliasChoices("brand_diagnosis", "idea_summary"),
    )
    market_insight: str | None = None
    suggested_positioning: list[str] | None = Field(
        default=None,
        max_length=6,
        validation_alias=AliasChoices("suggested_positioning", "recommendations"),
    )
    risks: list[str] | None = Field(default=None, max_length=6)
    opportunities: list[str] | None = Field(default=None, max_length=6)
    final_positioning: str | None = None
    target_audience: str | None = None
    chosen_strategy: str | None = None
    rejected_directions: list[str] | None = Field(default=None, max_length=6)
    trade_offs: list[str] | None = Field(
        default=None,
        max_length=6,
        validation_alias=AliasChoices("trade_offs", "tradeoffs"),
    )
    actions: list[ActionItem] = Field(default_factory=list, max_length=6)
    confidence_score: float = Field(ge=0.0, le=1.0)
    notion_page_content: str | None = None
    database_metadata: NotionDatabaseMetadata | None = None
