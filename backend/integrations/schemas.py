from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class UserIntegrationCreate(BaseModel):
    provider: Literal["notion", "jira"]
    access_token: str
    workspace_id: str | None = None
    database_id: str | None = None


class UserIntegrationOut(BaseModel):
    provider: str
    workspace_id: str | None
    database_id: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)