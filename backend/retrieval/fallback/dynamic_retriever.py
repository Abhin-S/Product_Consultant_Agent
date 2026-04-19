from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.chunker import Chunk, chunk_documents
from ingestion.loader import Document
from retrieval.fallback.news_client import fetch_news_articles
from retrieval.fallback.news_filter import filter_articles
from retrieval.retriever import RetrievedDoc


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


async def retrieve_dynamic_chunks(
    query: str,
    local_docs: list[RetrievedDoc],
    embedder: SentenceTransformer,
) -> tuple[list[Chunk], dict]:
    articles = await fetch_news_articles(query)

    query_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    if local_docs:
        local_embeddings = embedder.encode(
            [doc.text for doc in local_docs],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        kb_centroid_emb = _normalize(np.mean(local_embeddings, axis=0))
    else:
        kb_centroid_emb = query_emb

    filtered_articles, filter_stats = filter_articles(
        articles=articles,
        query=query,
        query_emb=query_emb,
        kb_centroid_emb=kb_centroid_emb,
        embedder=embedder,
    )

    article_docs: list[Document] = []
    relevance_by_source: dict[str, float] = {}

    for idx, article in enumerate(filtered_articles):
        combined_text = " ".join(
            [
                str(article.get("title") or ""),
                str(article.get("description") or ""),
                str(article.get("content") or ""),
            ]
        ).strip()
        if not combined_text:
            continue

        source = str(article.get("url") or f"dynamic://article/{idx}")
        relevance_by_source[source] = float(article.get("relevance_score") or 0.0)
        article_docs.append(Document(text=combined_text, source=source))

    chunks = chunk_documents(article_docs, chunk_size=400, overlap=0, doc_type="dynamic")

    inserted_at = datetime.now(timezone.utc).isoformat()
    for chunk in chunks:
        chunk.doc_type = "dynamic"
        chunk.inserted_at = inserted_at
        chunk.relevance_score = relevance_by_source.get(chunk.source, 0.0)

    return chunks, filter_stats