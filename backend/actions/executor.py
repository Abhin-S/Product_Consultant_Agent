from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from actions.jira_client import create_jira_issue
from actions.notion_client import create_notion_actions_database, create_notion_task
from integrations.encryption import decrypt_token
from integrations.models import UserIntegration
from reasoning.schema import ActionItem


@dataclass
class ActionResult:
    action_title: str
    target_provider: str
    external_id: str | None
    status: str
    error_message: str | None


async def execute_actions(
    actions: list[ActionItem],
    target: Literal["notion", "jira", "both"],
    user_id: UUID,
    db: AsyncSession,
) -> list[ActionResult]:
    providers = ["notion", "jira"] if target == "both" else [target]

    result = await db.execute(
        select(UserIntegration).where(
            UserIntegration.user_id == user_id,
            UserIntegration.provider.in_(providers),
        )
    )
    integrations = {row.provider: row for row in result.scalars().all()}

    out: list[ActionResult] = []

    for provider in providers:
        integration = integrations.get(provider)
        if integration is None:
            for action in actions:
                out.append(
                    ActionResult(
                        action_title=action.title,
                        target_provider=provider,
                        external_id=None,
                        status="failed",
                        error_message="Integration not connected. Please connect via /integrations.",
                    )
                )
            continue

        try:
            decrypted_token = decrypt_token(integration.access_token_encrypted)
        except ValueError as exc:
            for action in actions:
                out.append(
                    ActionResult(
                        action_title=action.title,
                        target_provider=provider,
                        external_id=None,
                        status="failed",
                        error_message=str(exc),
                    )
                )
            continue

        notion_database_id = integration.database_id
        if provider == "notion" and not notion_database_id:
            parent_page_id = (integration.workspace_id or "").strip()
            if not parent_page_id:
                message = (
                    "Notion database_id is required. Provide an existing Database ID, or provide a Notion "
                    "parent page id to auto-create the action database."
                )
                for action in actions:
                    out.append(
                        ActionResult(
                            action_title=action.title,
                            target_provider=provider,
                            external_id=None,
                            status="failed",
                            error_message=message,
                        )
                    )
                continue

            try:
                notion_database_id = create_notion_actions_database(
                    user_token=decrypted_token,
                    parent_page_id=parent_page_id,
                )
                integration.database_id = notion_database_id
                await db.commit()
                await db.refresh(integration)
            except Exception as exc:
                for action in actions:
                    out.append(
                        ActionResult(
                            action_title=action.title,
                            target_provider=provider,
                            external_id=None,
                            status="failed",
                            error_message=str(exc),
                        )
                    )
                continue

        for action in actions:
            try:
                external_id: str | None = None

                if provider == "notion":
                    if not notion_database_id:
                        raise ValueError("Notion database_id is required for execution")
                    external_id = create_notion_task(
                        user_token=decrypted_token,
                        database_id=notion_database_id,
                        title=action.title,
                        description=action.description,
                        priority=action.priority,
                    )
                else:
                    if not integration.workspace_id or not integration.database_id:
                        raise ValueError("Jira URL (workspace_id) and project key (database_id) are required")
                    external_id = create_jira_issue(
                        user_token=decrypted_token,
                        jira_url=integration.workspace_id,
                        project_key=integration.database_id,
                        title=action.title,
                        description=action.description,
                        priority=action.priority,
                    )

                if external_id is None:
                    out.append(
                        ActionResult(
                            action_title=action.title,
                            target_provider=provider,
                            external_id=None,
                            status="failed",
                            error_message="Provider API call failed",
                        )
                    )
                else:
                    out.append(
                        ActionResult(
                            action_title=action.title,
                            target_provider=provider,
                            external_id=external_id,
                            status="executed",
                            error_message=None,
                        )
                    )
            except Exception as exc:
                out.append(
                    ActionResult(
                        action_title=action.title,
                        target_provider=provider,
                        external_id=None,
                        status="failed",
                        error_message=str(exc),
                    )
                )

    return out