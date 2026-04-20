from datetime import datetime, timezone
import re
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from auth.models import User
from database import get_db
from integrations.encryption import encrypt_token
from integrations.models import UserIntegration
from integrations.schemas import UserIntegrationCreate, UserIntegrationOut


router = APIRouter(prefix="/integrations", tags=["integrations"])

NOTION_ID_RE = re.compile(r"[0-9a-fA-F]{32}")
NOTION_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def _to_notion_uuid(raw_32_hex: str) -> str:
    value = raw_32_hex.lower()
    return f"{value[:8]}-{value[8:12]}-{value[12:16]}-{value[16:20]}-{value[20:32]}"


def _extract_notion_id_or_raise(value: str, *, field_label: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise HTTPException(status_code=422, detail=f"Invalid Notion {field_label}. Value is empty.")

    exact_uuid = NOTION_UUID_RE.fullmatch(candidate)
    if exact_uuid is not None:
        return exact_uuid.group(0).lower()

    exact_raw = NOTION_ID_RE.fullmatch(candidate)
    if exact_raw is not None:
        return _to_notion_uuid(exact_raw.group(0))

    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid Notion {field_label}. Provide a Notion ID or full Notion URL "
                f"(for example: https://www.notion.so/...)."
            ),
        )

    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in {"notion.so", "notion.site"}:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid Notion {field_label}. URL must be from notion.so or notion.site.",
        )

    search_space = f"{parsed.path}?{parsed.query}"

    uuid_match = NOTION_UUID_RE.search(search_space)
    if uuid_match is not None:
        return uuid_match.group(0).lower()

    raw_match = NOTION_ID_RE.search(search_space)
    if raw_match is not None:
        return _to_notion_uuid(raw_match.group(0))

    raise HTTPException(
        status_code=422,
        detail=(
            f"Invalid Notion {field_label}. Could not find a Notion ID in the provided URL. "
            f"Use an ID or a direct page/database URL."
        ),
    )


def _normalize_optional_notion_id(raw_value: str | None, *, field_label: str) -> str | None:
    if raw_value is None:
        return None

    stripped = raw_value.strip()
    if not stripped:
        return None

    return _extract_notion_id_or_raise(stripped, field_label=field_label)


@router.post("/connect", response_model=UserIntegrationOut)
async def connect_integration(
    payload: UserIntegrationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserIntegrationOut:
    encrypted = encrypt_token(payload.access_token)

    normalized_workspace_id = payload.workspace_id
    normalized_database_id = payload.database_id
    if payload.provider == "notion":
        normalized_workspace_id = _normalize_optional_notion_id(
            payload.workspace_id,
            field_label="report page id",
        )
        normalized_database_id = _normalize_optional_notion_id(
            payload.database_id,
            field_label="database id",
        )

    existing_result = await db.execute(
        select(UserIntegration).where(
            UserIntegration.user_id == current_user.id,
            UserIntegration.provider == payload.provider,
        )
    )
    existing = existing_result.scalar_one_or_none()

    if existing is not None:
        existing.access_token_encrypted = encrypted

        if payload.workspace_id is not None:
            existing.workspace_id = normalized_workspace_id

        if payload.database_id is not None:
            existing.database_id = normalized_database_id

        existing.updated_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(existing)
        return UserIntegrationOut.model_validate(existing)

    row = UserIntegration(
        user_id=current_user.id,
        provider=payload.provider,
        access_token_encrypted=encrypted,
        workspace_id=normalized_workspace_id,
        database_id=normalized_database_id,
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

