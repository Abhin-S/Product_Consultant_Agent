from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from config import settings
from ingestion.loader import Document


ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    doc_type: str = "local"
    inserted_at: str | None = None
    relevance_score: float | None = None
    parent_id: str | None = None
    parent_index: int | None = None
    child_index: int | None = None
    chunk_type: str = "chunk"


@dataclass
class ParentChunk:
    parent_id: str
    source: str
    parent_index: int
    text: str


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = ENCODING.encode(text)
    if not tokens:
        return []

    step = max(chunk_size - overlap, 1)
    windows: list[str] = []

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        token_window = tokens[start:end]
        if not token_window:
            continue
        windows.append(ENCODING.decode(token_window))
        if end >= len(tokens):
            break

    return windows


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 600,
    overlap: int = 100,
    doc_type: str = "local",
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for doc in documents:
        split_chunks = _chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        for idx, text in enumerate(split_chunks):
            chunks.append(
                Chunk(text=text, source=doc.source, chunk_index=idx, doc_type=doc_type, chunk_type=doc_type)
            )

    return chunks


def chunk_documents_hierarchical(
    documents: list[Document],
    parent_chunk_size: int | None = None,
    parent_overlap: int | None = None,
    child_chunk_size: int | None = None,
    child_overlap: int | None = None,
) -> tuple[list[ParentChunk], list[Chunk]]:
    parent_size = parent_chunk_size or settings.PARENT_CHUNK_SIZE
    parent_ov = parent_overlap or settings.PARENT_CHUNK_OVERLAP
    child_size = child_chunk_size or settings.CHILD_CHUNK_SIZE
    child_ov = child_overlap or settings.CHILD_CHUNK_OVERLAP

    parents: list[ParentChunk] = []
    children: list[Chunk] = []

    for doc in documents:
        parent_texts = _chunk_text(doc.text, chunk_size=parent_size, overlap=parent_ov)
        child_counter = 0

        for parent_index, parent_text in enumerate(parent_texts):
            parent_id = f"{doc.source}::parent::{parent_index}"
            parents.append(
                ParentChunk(
                    parent_id=parent_id,
                    source=doc.source,
                    parent_index=parent_index,
                    text=parent_text,
                )
            )

            child_texts = _chunk_text(parent_text, chunk_size=child_size, overlap=child_ov)
            for child_index, child_text in enumerate(child_texts):
                children.append(
                    Chunk(
                        text=child_text,
                        source=doc.source,
                        chunk_index=child_counter,
                        doc_type="local",
                        parent_id=parent_id,
                        parent_index=parent_index,
                        child_index=child_index,
                        chunk_type="child",
                    )
                )
                child_counter += 1

    return parents, children