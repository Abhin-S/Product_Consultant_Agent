from __future__ import annotations

from dataclasses import dataclass

from config import settings
from ingestion.chunker import Chunk, count_tokens
from retrieval.retriever import RetrievedDoc


@dataclass
class ContextBundle:
    docs: list[RetrievedDoc]
    used_fallback: bool
    total_tokens: int
    local_tokens: int
    dynamic_tokens: int


def build_context_bundle(
    local_docs: list[RetrievedDoc],
    dynamic_chunks: list[Chunk],
) -> ContextBundle:
    dynamic_docs = [
        RetrievedDoc(
            text=chunk.text,
            source=chunk.source,
            similarity_score=float(chunk.relevance_score or 0.0),
            doc_type="dynamic",
        )
        for chunk in dynamic_chunks
    ]

    local_items = [(doc, count_tokens(doc.text)) for doc in local_docs]
    dynamic_items = [(doc, count_tokens(doc.text)) for doc in dynamic_docs]

    local_tokens = sum(tokens for _, tokens in local_items)
    dynamic_tokens = sum(tokens for _, tokens in dynamic_items)
    total_tokens = local_tokens + dynamic_tokens

    cap = settings.MAX_CONTEXT_TOKENS

    while dynamic_items and total_tokens > cap:
        _, removed = dynamic_items.pop()
        dynamic_tokens -= removed
        total_tokens -= removed

    while local_items and total_tokens > cap:
        _, removed = local_items.pop()
        local_tokens -= removed
        total_tokens -= removed

    merged_docs = [doc for doc, _ in local_items] + [doc for doc, _ in dynamic_items]

    return ContextBundle(
        docs=merged_docs,
        used_fallback=bool(dynamic_chunks),
        total_tokens=max(total_tokens, 0),
        local_tokens=max(local_tokens, 0),
        dynamic_tokens=max(dynamic_tokens, 0),
    )