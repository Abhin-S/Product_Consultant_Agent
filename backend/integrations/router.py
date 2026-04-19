from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from auth.models import User
from database import get_db
from integrations.encryption import encrypt_token
from integrations.jira_oauth import jira_oauth_stub
from integrations.models import UserIntegration
from integrations.notion_oauth import notion_oauth_stub
from integrations.schemas import UserIntegrationCreate, UserIntegrationOut


router = APIRouter(prefix="/integrations", tags=["integrations"])


@router.post("/connect", response_model=UserIntegrationOut)
async def connect_integration(
    payload: UserIntegrationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserIntegrationOut:
    encrypted = encrypt_token(payload.access_token)

    existing_result = await db.execute(
        select(UserIntegration).where(
            UserIntegration.user_id == current_user.id,
            UserIntegration.provider == payload.provider,
        )
    )
    existing = existing_result.scalar_one_or_none()

    if existing is not None:
        existing.access_token_encrypted = encrypted
        existing.workspace_id = payload.workspace_id
        existing.database_id = payload.database_id
        existing.updated_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(existing)
        return UserIntegrationOut.model_validate(existing)

    row = UserIntegration(
        user_id=current_user.id,
        provider=payload.provider,
        access_token_encrypted=encrypted,
        workspace_id=payload.workspace_id,
        database_id=payload.database_id,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return UserIntegrationOut.model_validate(row)


@router.get("", response_model=list[UserIntegrationOut])
async def list_integrations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[UserIntegrationOut]:
    result = await db.execute(select(UserIntegration).where(UserIntegration.user_id == current_user.id))
    rows = result.scalars().all()
    return [UserIntegrationOut.model_validate(row) for row in rows]


@router.delete("/{provider}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_integration(
    provider: Literal["notion", "jira"],
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    result = await db.execute(
        select(UserIntegration).where(
            UserIntegration.user_id == current_user.id,
            UserIntegration.provider == provider,
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Integration not found")

    await db.execute(
        delete(UserIntegration).where(
            UserIntegration.user_id == current_user.id,
            UserIntegration.provider == provider,
        )
    )
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{provider}/connect-oauth")
async def connect_oauth_stub(provider: Literal["notion", "jira"]) -> None:
    if provider == "notion":
        notion_oauth_stub()
    jira_oauth_stub()


@router.get("/{provider}/callback")
async def oauth_callback_stub(provider: Literal["notion", "jira"]) -> None:
    if provider == "notion":
        notion_oauth_stub()
    jira_oauth_stub()