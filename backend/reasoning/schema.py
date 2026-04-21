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

    name: str = Field(min_length=3, max_length=120)
    brand_positioning: str = Field(
        min_length=10,
        max_length=300,
        validation_alias=AliasChoices("brand_positioning", "idea_description"),
    )
    brand_risk_level: Literal["Low", "Medium", "High"] = Field(
        validation_alias=AliasChoices("brand_risk_level", "risk_level")
    )
    confidence_score: int = Field(ge=0, le=100)
    tags: list[str] = Field(default_factory=list, max_length=8)


class InsightOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    brand_diagnosis: str = Field(validation_alias=AliasChoices("brand_diagnosis", "idea_summary"))
    market_insight: str = ""
    suggested_positioning: list[str] = Field(
        default_factory=list,
        max_length=6,
        validation_alias=AliasChoices("suggested_positioning", "recommendations"),
    )
    risks: list[str] = Field(min_length=2, max_length=6)
    opportunities: list[str] = Field(min_length=2, max_length=6)
    final_positioning: str = ""
    target_audience: str = ""
    chosen_strategy: str = ""
    rejected_directions: list[str] = Field(default_factory=list, max_length=6)
    trade_offs: list[str] = Field(
        default_factory=list,
        max_length=6,
        validation_alias=AliasChoices("trade_offs", "tradeoffs"),
    )
    actions: list[ActionItem] = Field(min_length=1, max_length=6)
    confidence_score: float = Field(ge=0.0, le=1.0)
    notion_page_content: str = ""
    database_metadata: NotionDatabaseMetadata | None = None