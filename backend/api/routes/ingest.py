from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config import settings
from auth.dependencies import get_current_user
from auth.models import User
from ingestion.chunker import chunk_documents_hierarchical
from ingestion.embedder import upsert_local_chunks
from ingestion.loader import load_documents
from ingestion.preprocessor import preprocess_documents
from retrieval.parent_store import save_parent_chunks


router = APIRouter(tags=["ingestion"])


class IngestRequest(BaseModel):
    docs_dir: str | None = None


@router.post("/ingest")
async def ingest_documents(
    payload: IngestRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    docs = load_documents(payload.docs_dir)
    preprocessed = preprocess_documents(docs)
    parents, child_chunks = chunk_documents_hierarchical(
        preprocessed,
        parent_chunk_size=settings.PARENT_CHUNK_SIZE,
        parent_overlap=settings.PARENT_CHUNK_OVERLAP,
        child_chunk_size=settings.CHILD_CHUNK_SIZE,
        child_overlap=settings.CHILD_CHUNK_OVERLAP,
    )
    parent_saved = save_parent_chunks(parents)
    inserted = upsert_local_chunks(child_chunks)

    sources = sorted({doc.source for doc in preprocessed})
    return {"chunks_ingested": inserted, "parent_chunks_saved": parent_saved, "sources": sources}