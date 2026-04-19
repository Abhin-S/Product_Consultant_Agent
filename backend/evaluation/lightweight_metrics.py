from __future__ import annotations

from retrieval.fallback.context_builder import ContextBundle
from retrieval.retriever import RetrievedDoc
from config import settings


def compute_avg_similarity(retrieved_docs: list[RetrievedDoc]) -> float:
    """Return the mean similarity score for retrieved documents."""
    if not retrieved_docs:
        return 0.0
    return float(sum(doc.similarity_score for doc in retrieved_docs) / len(retrieved_docs))


def compute_similarity_distribution(retrieved_docs: list[RetrievedDoc]) -> dict:
    """Return min/max/mean and threshold stats for similarity scores."""
    if not retrieved_docs:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "above_threshold": 0, "total": 0}

    scores = [float(doc.similarity_score) for doc in retrieved_docs]
    return {
        "min": float(min(scores)),
        "max": float(max(scores)),
        "mean": float(sum(scores) / len(scores)),
        "above_threshold": int(sum(1 for score in scores if score > settings.CONFIDENCE_THRESHOLD)),
        "total": len(scores),
    }


def compute_context_token_ratio(context_bundle: ContextBundle) -> dict:
    """Return token totals and local/dynamic context proportions."""
    total = max(int(context_bundle.total_tokens), 0)
    local = max(int(context_bundle.local_tokens), 0)
    dynamic = max(int(context_bundle.dynamic_tokens), 0)

    if total == 0:
        return {
            "total_tokens": 0,
            "local_tokens": 0,
            "dynamic_tokens": 0,
            "local_ratio": 0.0,
            "dynamic_ratio": 0.0,
        }

    return {
        "total_tokens": total,
        "local_tokens": local,
        "dynamic_tokens": dynamic,
        "local_ratio": float(local / total),
        "dynamic_ratio": float(dynamic / total),
    }


def compute_fallback_stats(
    used_fallback: bool,
    articles_fetched: int,
    articles_surviving: int,
    avg_fallback_relevance: float | None,
) -> dict:
    """Return normalized fallback statistics used for Tier 1 logging."""
    if not used_fallback:
        return {
            "used_fallback": False,
            "articles_fetched": 0,
            "articles_surviving": 0,
            "avg_fallback_relevance": None,
        }

    return {
        "used_fallback": True,
        "articles_fetched": max(int(articles_fetched), 0),
        "articles_surviving": max(int(articles_surviving), 0),
        "avg_fallback_relevance": (
            float(avg_fallback_relevance) if avg_fallback_relevance is not None else None
        ),
    }


def compute_generation_stats(
    latency_ms: float,
    retry_count: int,
    validation_passed: bool,
) -> dict:
    """Return normalized generation diagnostics for Tier 1 logging."""
    return {
        "latency_ms": float(max(latency_ms, 0.0)),
        "retry_count": max(int(retry_count), 0),
        "validation_passed": bool(validation_passed),
    }