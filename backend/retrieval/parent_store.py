from __future__ import annotations

import json
import logging
from pathlib import Path

from config import settings


logger = logging.getLogger(__name__)
_parent_store: dict[str, dict] | None = None


def _store_path() -> Path:
    return Path(settings.PARENT_STORE_PATH)


def load_parent_store(force_reload: bool = False) -> dict[str, dict]:
    global _parent_store

    if _parent_store is not None and not force_reload:
        return _parent_store

    path = _store_path()
    if not path.exists():
        _parent_store = {}
        return _parent_store

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            _parent_store = raw
        else:
            _parent_store = {}
    except Exception as exc:
        logger.warning("Failed to load parent store '%s': %s", path, exc)
        _parent_store = {}

    return _parent_store


def warm_parent_store() -> None:
    load_parent_store(force_reload=False)


def get_parent_chunk(parent_id: str) -> dict | None:
    if not parent_id:
        return None
    store = load_parent_store(force_reload=False)
    value = store.get(parent_id)
    return value if isinstance(value, dict) else None


def save_parent_chunks(parents: list, *, replace_existing_sources: bool = True) -> int:
    store = load_parent_store(force_reload=False)

    if replace_existing_sources:
        sources = {
            getattr(parent, "source", "")
            for parent in parents
            if getattr(parent, "source", "")
        }
        if sources:
            remove_ids = [
                parent_id
                for parent_id, payload in store.items()
                if isinstance(payload, dict) and str(payload.get("source", "")) in sources
            ]
            for parent_id in remove_ids:
                store.pop(parent_id, None)

    added = 0
    for parent in parents:
        parent_id = getattr(parent, "parent_id", "")
        if not parent_id:
            continue

        entry = {
            "text": getattr(parent, "text", ""),
            "source": getattr(parent, "source", "unknown"),
            "parent_index": int(getattr(parent, "parent_index", 0)),
        }

        if parent_id not in store:
            added += 1
        store[parent_id] = entry

    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=True), encoding="utf-8")

    return added