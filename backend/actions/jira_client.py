from __future__ import annotations

import logging

import httpx


logger = logging.getLogger(__name__)


PRIORITY_MAP = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
}


def create_jira_issue(
    user_token: str,
    jira_url: str,
    project_key: str,
    title: str,
    description: str,
    priority: str,
) -> str | None:
    url = f"{jira_url.rstrip('/')}/rest/api/3/issue"
    headers = {
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": title,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}],
                    }
                ],
            },
            "issuetype": {"name": "Story"},
            "priority": {"name": PRIORITY_MAP.get(priority, "Medium")},
        }
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("key")
    except Exception as exc:
        logger.error("Failed to create Jira issue: %s", exc)
        return None