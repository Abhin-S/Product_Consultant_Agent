from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from actions.executor import execute_actions
from auth.dependencies import get_current_user
from auth.models import User
from database import get_db
from evaluation.models import ActionLog, AnalysisSession
from integrations.models import UserIntegration
from reasoning.schema import ActionItem, InsightOutput


router = APIRouter(tags=["execution"])


class ExecuteRequest(BaseModel):
    session_id: UUID
    target: str
    selected_action_indices: list[int] | None = None


@router.post("/execute")
async def execute_session_actions(
    payload: ExecuteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    if payload.target not in {"notion", "jira", "both"}:
        raise HTTPException(status_code=422, detail="target must be one of: notion, jira, both")

    session_result = await db.execute(select(AnalysisSession).where(AnalysisSession.id == payload.session_id))
    session = session_result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this session")

    insight = InsightOutput.model_validate(session.raw_output)
    actions: list[ActionItem] = list(insight.actions)

    if payload.selected_action_indices is not None:
        selected = set(payload.selected_action_indices)
        actions = [action for idx, action in enumerate(actions) if idx in selected]

    providers = ["notion", "jira"] if payload.target == "both" else [payload.target]
    integrations_result = await db.execute(
        select(UserIntegration).where(
            UserIntegration.user_id == current_user.id,
            UserIntegration.provider.in_(providers),
        )
    )
    available = {row.provider for row in integrations_result.scalars().all()}

    for provider in providers:
        if provider not in available:
            provider_label = "Notion" if provider == "notion" else "Jira"
            raise HTTPException(
                status_code=400,
                detail=f"Please connect your {provider_label} account in /integrations before executing.",
            )

    results = await execute_actions(actions=actions, target=payload.target, user_id=current_user.id, db=db)

    action_by_title = {action.title: action for action in actions}
    for result in results:
        action = action_by_title.get(result.action_title)
        db.add(
            ActionLog(
                session_id=session.id,
                action_type=(action.type if action else "task"),
                title=result.action_title,
                description=(action.description if action else ""),
                priority=(action.priority if action else "medium"),
                target_provider=result.target_provider,
                status=result.status,
                external_id=result.external_id,
                error_message=result.error_message,
            )
        )

    await db.commit()

    return [
        {
            "action_title": result.action_title,
            "target_provider": result.target_provider,
            "external_id": result.external_id,
            "status": result.status,
            "error_message": result.error_message,
        }
        for result in results
    ]