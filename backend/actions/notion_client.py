from __future__ import annotations

import logging
from typing import Any

import httpx


logger = logging.getLogger(__name__)

NOTION_API_BASE_URL = "https://api.notion.com/v1"
NOTION_API_VERSION = "2022-06-28"


def _notion_headers(user_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {user_token}",
        "Notion-Version": NOTION_API_VERSION,
        "Content-Type": "application/json",
    }


def _response_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}

    return payload if isinstance(payload, dict) else {}


def _normalize_name(value: str) -> str:
    return value.strip().lower()


def _find_property(
    properties: dict[str, dict[str, Any]],
    *,
    property_type: str | None = None,
    exact_name: str | None = None,
    contains_name: str | None = None,
) -> str | None:
    for name, definition in properties.items():
        if property_type is not None and definition.get("type") != property_type:
            continue

        lowered_name = _normalize_name(name)
        if exact_name is not None and lowered_name != _normalize_name(exact_name):
            continue
        if contains_name is not None and _normalize_name(contains_name) not in lowered_name:
            continue

        return name

    return None


def _get_select_options(properties: dict[str, dict[str, Any]], property_name: str) -> list[dict[str, Any]]:
    definition = properties.get(property_name) or {}
    select_meta = definition.get("select") if isinstance(definition, dict) else {}
    options = select_meta.get("options") if isinstance(select_meta, dict) else None
    if not isinstance(options, list):
        return []
    return [option for option in options if isinstance(option, dict)]


def _get_status_options(properties: dict[str, dict[str, Any]], property_name: str) -> list[dict[str, Any]]:
    definition = properties.get(property_name) or {}
    status_meta = definition.get("status") if isinstance(definition, dict) else {}
    options = status_meta.get("options") if isinstance(status_meta, dict) else None
    if not isinstance(options, list):
        return []
    return [option for option in options if isinstance(option, dict)]


def _missing_options(existing_options: list[dict[str, Any]], desired_options: list[dict[str, str]]) -> list[dict[str, str]]:
    existing_names = {
        _normalize_name(str(option.get("name") or ""))
        for option in existing_options
        if str(option.get("name") or "").strip()
    }

    return [
        option
        for option in desired_options
        if _normalize_name(option["name"]) not in existing_names
    ]


def _append_select_update(
    updates: dict[str, dict[str, Any]],
    property_name: str,
    options: list[dict[str, str]],
) -> None:
    if not options:
        return

    current = updates.get(property_name)
    if isinstance(current, dict):
        select_meta = current.get("select")
        if isinstance(select_meta, dict):
            existing = select_meta.get("options")
            if isinstance(existing, list):
                existing_names = {
                    _normalize_name(str(option.get("name") or ""))
                    for option in existing
                    if isinstance(option, dict) and str(option.get("name") or "").strip()
                }

                for option in options:
                    option_name = str(option.get("name") or "").strip()
                    if not option_name:
                        continue
                    normalized = _normalize_name(option_name)
                    if normalized in existing_names:
                        continue
                    existing_names.add(normalized)
                    existing.append(option)

                return

    updates[property_name] = {"select": {"options": options}}


def _build_notion_error_message(response: httpx.Response, *, context: str = "request") -> str:
    fallback_text = response.text.strip() or response.reason_phrase
    payload = _response_payload(response)

    code = str(payload.get("code") or "").strip()
    message = str(payload.get("message") or "").strip()
    details = f"{code}: {message}" if code and message else (message or fallback_text)

    if response.status_code == 404:
        return (
            f"Notion returned 404 during {context} ({details}). Verify the target id and ensure "
            "the integration is added in Notion -> Connections."
        )
    if response.status_code == 401:
        return (
            f"Notion authentication failed during {context} (401 - {details}). Use a valid internal "
            "integration token (secret_...) for the connected user."
        )
    if response.status_code == 400:
        if code == "validation_error":
            return f"Notion validation error during {context} ({details})."
        return (
            f"Notion rejected the request during {context} (400 - {details})."
        )

    return f"Notion API error during {context} ({response.status_code} - {details})."


def create_notion_actions_database(user_token: str, parent_page_id: str) -> str:
    if not parent_page_id.strip():
        raise ValueError("Notion parent page id is required to auto-create a database.")

    payload = {
        "parent": {"type": "page_id", "page_id": parent_page_id.strip()},
        "title": [{"type": "text", "text": {"content": "Product Consultant Actions"}}],
        "properties": {
            "Name": {"title": {}},
            "Description": {"rich_text": {}},
            "Priority": {
                "select": {
                    "options": [
                        {"name": "High", "color": "red"},
                        {"name": "Medium", "color": "yellow"},
                        {"name": "Low", "color": "green"},
                    ]
                }
            },
            "Status": {
                "select": {
                    "options": [
                        {"name": "To Do", "color": "default"},
                        {"name": "In Progress", "color": "blue"},
                        {"name": "Done", "color": "green"},
                    ]
                }
            },
        },
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{NOTION_API_BASE_URL}/databases",
                json=payload,
                headers=_notion_headers(user_token),
            )
    except httpx.RequestError as exc:
        message = f"Notion request failed while creating database: {exc}"
        logger.error("Failed to create Notion database: %s", message)
        raise ValueError(message) from exc

    if response.is_error:
        message = _build_notion_error_message(response, context="database creation")
        logger.error("Failed to create Notion database: %s", message)
        raise ValueError(message)

    data = _response_payload(response)
    database_id = data.get("id")
    if not database_id:
        raise ValueError("Notion created a database but did not return its id.")

    return str(database_id)


def ensure_notion_task_schema(user_token: str, database_id: str) -> dict[str, str]:
    priority_options = [
        {"name": "High", "color": "red"},
        {"name": "Medium", "color": "yellow"},
        {"name": "Low", "color": "green"},
    ]
    status_options = [
        {"name": "To Do", "color": "default"},
        {"name": "In Progress", "color": "blue"},
        {"name": "Done", "color": "green"},
    ]

    try:
        with httpx.Client(timeout=10.0) as client:
            fetch_response = client.get(
                f"{NOTION_API_BASE_URL}/databases/{database_id}",
                headers=_notion_headers(user_token),
            )
            if fetch_response.is_error:
                message = _build_notion_error_message(fetch_response, context="database lookup")
                logger.error("Failed to inspect Notion database schema: %s", message)
                raise ValueError(message)

            database_payload = _response_payload(fetch_response)
            raw_properties = database_payload.get("properties")
            if not isinstance(raw_properties, dict) or not raw_properties:
                raise ValueError("Notion database schema response was invalid.")

            properties: dict[str, dict[str, Any]] = {
                name: definition
                for name, definition in raw_properties.items()
                if isinstance(name, str) and isinstance(definition, dict)
            }

            title_property = _find_property(properties, property_type="title")
            if title_property is None:
                raise ValueError("Notion database must have one title property.")

            updates: dict[str, dict[str, Any]] = {}

            description_property = _find_property(
                properties,
                exact_name="Description",
                property_type="rich_text",
            )
            if description_property is None:
                description_property = _find_property(
                    properties,
                    contains_name="description",
                    property_type="rich_text",
                )
            if description_property is None:
                description_property = "Description" if "Description" not in properties else "Description Text"
                updates[description_property] = {"rich_text": {}}

            priority_property = _find_property(
                properties,
                exact_name="Priority",
                property_type="select",
            )
            if priority_property is None:
                priority_property = _find_property(
                    properties,
                    contains_name="priority",
                    property_type="select",
                )
            if priority_property is None:
                priority_property = "Priority" if "Priority" not in properties else "Priority Select"
                updates[priority_property] = {"select": {"options": priority_options}}
            else:
                missing_priority_options = _missing_options(
                    _get_select_options(properties, priority_property),
                    priority_options,
                )
                _append_select_update(updates, priority_property, missing_priority_options)

            status_property = _find_property(properties, exact_name="Status", property_type="select")
            status_type = "select"
            if status_property is None:
                status_property = _find_property(properties, exact_name="Status", property_type="status")
                if status_property is not None:
                    status_type = "status"
            if status_property is None:
                status_property = _find_property(properties, contains_name="status", property_type="select")
            if status_property is None:
                status_property = _find_property(properties, contains_name="status", property_type="status")
                if status_property is not None:
                    status_type = "status"

            if status_property is None:
                status_property = "Status" if "Status" not in properties else "Status Select"
                status_type = "select"
                updates[status_property] = {"select": {"options": status_options}}

            if status_type == "select":
                missing_status_options = _missing_options(
                    _get_select_options(properties, status_property),
                    status_options,
                )
                _append_select_update(updates, status_property, missing_status_options)

            status_default = "To Do"
            if status_type == "status":
                available_status_options = _get_status_options(properties, status_property)
                normalized_map = {
                    _normalize_name(str(option.get("name") or "")): str(option.get("name") or "")
                    for option in available_status_options
                    if str(option.get("name") or "").strip()
                }
                status_default = (
                    normalized_map.get("to do")
                    or normalized_map.get("not started")
                    or next(iter(normalized_map.values()), "Not started")
                )

            if updates:
                update_response = client.patch(
                    f"{NOTION_API_BASE_URL}/databases/{database_id}",
                    json={"properties": updates},
                    headers=_notion_headers(user_token),
                )
                if update_response.is_error:
                    message = _build_notion_error_message(update_response, context="database schema update")
                    logger.error("Failed to update Notion database schema: %s", message)
                    raise ValueError(message)

            return {
                "title": title_property,
                "description": description_property,
                "priority": priority_property,
                "status": status_property,
                "status_type": status_type,
                "status_default": status_default,
            }
    except httpx.RequestError as exc:
        message = f"Notion request failed while preparing schema: {exc}"
        logger.error("Failed to prepare Notion schema: %s", message)
        raise ValueError(message) from exc


def create_notion_task(
    user_token: str,
    database_id: str,
    title: str,
    description: str,
    priority: str,
) -> str:
    schema = ensure_notion_task_schema(user_token=user_token, database_id=database_id)

    normalized_priority = priority.strip().lower()
    priority_value = "Medium"
    if normalized_priority in {"high", "medium", "low"}:
        priority_value = normalized_priority.capitalize()

    status_value_key = "status" if schema.get("status_type") == "status" else "select"
    status_value_name = schema.get("status_default") or "To Do"
    payload = {
        "parent": {"database_id": database_id},
        "properties": {
            schema["title"]: {"title": [{"text": {"content": title}}]},
            schema["description"]: {"rich_text": [{"text": {"content": description}}]},
            schema["priority"]: {"select": {"name": priority_value}},
            schema["status"]: {status_value_key: {"name": status_value_name}},
        },
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{NOTION_API_BASE_URL}/pages",
                json=payload,
                headers=_notion_headers(user_token),
            )
    except httpx.RequestError as exc:
        message = f"Notion request failed: {exc}"
        logger.error("Failed to create Notion task: %s", message)
        raise ValueError(message) from exc

    if response.is_error:
        message = _build_notion_error_message(response, context="page creation")
        logger.error("Failed to create Notion task: %s", message)
        raise ValueError(message)

    try:
        data = response.json()
    except ValueError as exc:
        logger.error("Failed to create Notion task: invalid JSON response")
        raise ValueError("Notion returned an invalid response payload.") from exc

    page_id = data.get("id")
    if not page_id:
        logger.error("Failed to create Notion task: missing page id in response")
        raise ValueError("Notion API did not return a created page id.")

    return str(page_id)