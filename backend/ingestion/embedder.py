from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.chunker import Chunk
from retrieval.vector_store import get_collection


MODEL_NAME = "all-MiniLM-L6-v2"
_embedder: SentenceTransformer | None = None
_query_embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_QUERY_EMBED_CACHE_MAX = 2048


def load_embedder_model() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def get_embedder_model() -> SentenceTransformer:
    return load_embedder_model()


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedder_model()
    return model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def embed_query(query: str) -> np.ndarray:
    normalized = " ".join((query or "").split()).strip().lower()
    if normalized:
        cached = _query_embed_cache.get(normalized)
        if cached is not None:
            _query_embed_cache.move_to_end(normalized)
            return cached

    vector = embed_texts([query])[0]
    if normalized:
        _query_embed_cache[normalized] = vector
        _query_embed_cache.move_to_end(normalized)
        while len(_query_embed_cache) > _QUERY_EMBED_CACHE_MAX:
            _query_embed_cache.popitem(last=False)

    return vector


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    return None


def _extract_batch_limit_from_error(exc: Exception) -> int | None:
    match = re.search(r"maximum batch size\s+([0-9]+)", str(exc), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return max(int(match.group(1)), 1)
    except ValueError:
        return None


def _resolve_chroma_batch_size(collection, desired: int) -> int:
    candidates: list[int] = []
    client = getattr(collection, "_client", None)

    for obj in (collection, client):
        if obj is None:
            continue

        for attr_name in ("max_batch_size", "_max_batch_size"):
            value = getattr(obj, attr_name, None)
            candidate = _coerce_positive_int(value)
            if candidate is not None:
                candidates.append(candidate)

        getter = getattr(obj, "get_max_batch_size", None)
        if callable(getter):
            try:
                candidate = _coerce_positive_int(getter())
                if candidate is not None:
                    candidates.append(candidate)
            except Exception:
                continue

    if candidates:
        return max(1, min(min(candidates), desired))

    # Conservative default when Chroma does not expose the limit.
    return max(1, min(5000, desired))


def _build_chunk_metadata(chunk: Chunk, now: str) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "doc_type": "local",
        "inserted_at": chunk.inserted_at or now,
        "chunk_type": chunk.chunk_type,
        "section": chunk.section,
        "pages": chunk.pages,
        "has_table": bool(chunk.has_table),
    }

    if chunk.parent_id is not None:
        metadata["parent_id"] = chunk.parent_id
    if chunk.parent_index is not None:
        metadata["parent_index"] = int(chunk.parent_index)
    if chunk.child_index is not None:
        metadata["child_index"] = int(chunk.child_index)

    for raw_key, value in chunk.extra_metadata.items():
        key = str(raw_key)
        if key in metadata or value is None:
            continue
        if isinstance(value, bool):
            metadata[key] = value
            continue
        if isinstance(value, (str, int, float)):
            metadata[key] = value

    return metadata


def upsert_local_chunks(chunks: list[Chunk], *, replace_existing_sources: bool = True) -> int:
    if not chunks:
        return 0

    collection = get_collection()
    if replace_existing_sources:
        sources = sorted({chunk.source for chunk in chunks if chunk.source})
        for source in sources:
            try:
                collection.delete(where={"source": source})
            except Exception:
                continue

    if not chunks:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    batch_size = _resolve_chroma_batch_size(collection, desired=len(chunks))

    inserted = 0
    index = 0
    while index < len(chunks):
        current_batch_size = min(batch_size, len(chunks) - index)
        batch = chunks[index : index + current_batch_size]

        new_ids = [f"{chunk.source}_{chunk.chunk_index}" for chunk in batch]
        new_docs = [chunk.text for chunk in batch]
        new_embeddings = embed_texts(new_docs).tolist()
        new_metadatas = [_build_chunk_metadata(chunk, now) for chunk in batch]

        try:
            collection.upsert(
                ids=new_ids,
                documents=new_docs,
                embeddings=new_embeddings,
                metadatas=new_metadatas,
            )
            inserted += len(batch)
            index += len(batch)
        except ValueError as exc:
            updated_limit = _extract_batch_limit_from_error(exc)
            if updated_limit is not None and updated_limit < current_batch_size:
                batch_size = max(updated_limit, 1)
                continue
            raise

    return inserted