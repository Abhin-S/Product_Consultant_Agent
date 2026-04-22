from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Literal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from actions.jira_client import create_jira_issue
from actions.notion_client import (
    append_notion_report_to_page,
    create_notion_task,
    discover_notion_database_id,
    discover_notion_parent_page_id,
    get_notion_database_parent_page_id,
)
from integrations.encryption import decrypt_token
from integrations.models import UserIntegration
from reasoning.schema import ActionItem


logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    action_title: str
    target_provider: str
    external_id: str | None
    status: str
    error_message: str | None
    insight_page_url: str | None = None


@dataclass
class NotionExecutionContext:
    session_id: str
    notion_page_content: str
    database_metadata: dict | None = None


def _fallback_notion_content(actions: list[ActionItem]) -> str:
    lines = [
        "📌 Execution Summary",
        "Structured insight fields were unavailable for this run, so only executable actions are listed.",
        "",
        "🛠 Action Items",
    ]
    for idx, action in enumerate(actions, start=1):
        decision_type = action.decision_type.replace("_", " ").title()
        lines.append(
            f"* Task {idx}: {action.title}: {action.description} (Decision Type: {decision_type}; Impact: {action.impact.capitalize()})"
        )

    return "\n".join(lines)


async def execute_actions(
    actions: list[ActionItem],
    target: Literal["notion", "jira", "both"],
    user_id: UUID,
    db: AsyncSession,
    notion_context: NotionExecutionContext | None = None,
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

        notion_database_id = (integration.database_id or "").strip() or None
        notion_parent_page_id = (integration.workspace_id or "").strip() or None
        notion_insight_page_url: str | None = None
        should_persist_integration = False

        if provider == "notion" and not notion_database_id:
            try:
                notion_database_id = discover_notion_database_id(user_token=decrypted_token)
                if notion_database_id and notion_database_id != integration.database_id:
                    integration.database_id = notion_database_id
                    should_persist_integration = True
            except Exception as exc:
                logger.warning("Could not auto-discover Notion database: %s", exc)

        if provider == "notion" and notion_database_id and not notion_parent_page_id:
            try:
                notion_parent_page_id = get_notion_database_parent_page_id(
                    user_token=decrypted_token,
                    database_id=notion_database_id,
                )
                if notion_parent_page_id and notion_parent_page_id != integration.workspace_id:
                    integration.workspace_id = notion_parent_page_id
                    should_persist_integration = True
            except Exception as exc:
                logger.warning("Could not infer Notion parent page from database: %s", exc)

        if provider == "notion" and not notion_parent_page_id:
            try:
                notion_parent_page_id = discover_notion_parent_page_id(user_token=decrypted_token)
                if notion_parent_page_id and notion_parent_page_id != integration.workspace_id:
                    integration.workspace_id = notion_parent_page_id
                    should_persist_integration = True
            except Exception as exc:
                logger.warning("Could not auto-discover Notion parent page: %s", exc)

        if provider == "notion" and not notion_database_id:
            message = (
                "No Notion database_id was provided and no accessible database could be discovered. "
                "Share an existing database with the integration, or provide Database ID in Integrations."
            )
            for action in actions:
                out.append(
                    ActionResult(
                        action_title=action.title,
                        target_provider=provider,
                        external_id=None,
                        status="failed",
                        error_message=message,
                        insight_page_url=notion_insight_page_url,
                    )
                )
            continue

        if provider == "notion" and should_persist_integration:
            await db.commit()
            await db.refresh(integration)

        if provider == "notion" and notion_context is not None:
            content = (notion_context.notion_page_content or "").strip()
            if not content:
                content = _fallback_notion_content(actions)

            if notion_parent_page_id:
                try:
                    insight_page = append_notion_report_to_page(
                        user_token=decrypted_token,
                        page_id=notion_parent_page_id,
                        session_id=notion_context.session_id,
                        notion_page_content=content,
                        metadata=notion_context.database_metadata,
                    )
                    notion_insight_page_url = (
                        str(insight_page.get("url")) if insight_page.get("url") is not None else None
                    )
                except Exception as exc:
                    logger.warning("Notion report page update failed: %s", exc)
            else:
                logger.warning(
                    "No Notion report page id available; skipping report page update for session %s",
                    notion_context.session_id,
                )

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
                        decision_type=action.decision_type,
                        impact=action.impact,
                        insight_note_url=notion_insight_page_url,
                        session_reference=(notion_context.session_id if notion_context else None),
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
                            insight_page_url=(notion_insight_page_url if provider == "notion" else None),
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
                            insight_page_url=(notion_insight_page_url if provider == "notion" else None),
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
                        insight_page_url=(notion_insight_page_url if provider == "notion" else None),
                    )
                )

    return out