from __future__ import annotations

import logging

import httpx


logger = logging.getLogger(__name__)


def create_notion_task(
    user_token: str,
    database_id: str,
    title: str,
    description: str,
    priority: str,
) -> str | None:
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {user_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    payload = {
        "parent": {"database_id": database_id},
        "properties": {
            "Name": {"title": [{"text": {"content": title}}]},
            "Description": {"rich_text": [{"text": {"content": description}}]},
            "Priority": {"select": {"name": priority.capitalize()}},
            "Status": {"select": {"name": "To Do"}},
        },
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("id")
    except Exception as exc:
        logger.error("Failed to create Notion task: %s", exc)
        return None