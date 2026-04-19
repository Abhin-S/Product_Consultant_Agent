from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.chunker import Chunk
from retrieval.vector_store import get_collection


MODEL_NAME = "all-MiniLM-L6-v2"
_embedder: SentenceTransformer | None = None


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
    return embed_texts([query])[0]


def upsert_local_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0

    collection = get_collection()
    chunk_ids = [f"{chunk.source}_{chunk.chunk_index}" for chunk in chunks]

    existing = collection.get(ids=chunk_ids, include=[])
    existing_ids = set(existing.get("ids", []))

    new_chunks = [chunk for chunk, chunk_id in zip(chunks, chunk_ids) if chunk_id not in existing_ids]
    if not new_chunks:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    new_ids = [f"{chunk.source}_{chunk.chunk_index}" for chunk in new_chunks]
    new_docs = [chunk.text for chunk in new_chunks]
    new_embeddings = embed_texts(new_docs).tolist()
    new_metadatas = [
        {
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "doc_type": "local",
            "inserted_at": chunk.inserted_at or now,
            "parent_id": chunk.parent_id,
            "parent_index": chunk.parent_index,
            "child_index": chunk.child_index,
            "chunk_type": chunk.chunk_type,
        }
        for chunk in new_chunks
    ]

    collection.upsert(ids=new_ids, documents=new_docs, embeddings=new_embeddings, metadatas=new_metadatas)
    return len(new_chunks)