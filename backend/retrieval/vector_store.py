from __future__ import annotations

from datetime import datetime, timedelta, timezone

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata

from config import settings


_client: chromadb.PersistentClient | None = None
_collection: Collection | None = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    return _client


def get_collection() -> Collection:
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name="case_studies",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _parse_inserted_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def purge_expired_dynamic_documents(hours: int = 24) -> int:
    collection = get_collection()
    results = collection.get(where={"doc_type": "dynamic"}, include=["metadatas"])

    ids: list[str] = results.get("ids", [])
    metadatas: list[Metadata] = results.get("metadatas", [])
    if not ids or not metadatas:
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    delete_ids: list[str] = []

    for doc_id, metadata in zip(ids, metadatas):
        inserted_at = _parse_inserted_at(str(metadata.get("inserted_at")))
        if inserted_at is not None and inserted_at < cutoff:
            delete_ids.append(doc_id)

    if delete_ids:
        collection.delete(ids=delete_ids)

    return len(delete_ids)