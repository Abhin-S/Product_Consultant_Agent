from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer

from config import settings


logger = logging.getLogger(__name__)


def _combined_text(article: dict) -> str:
    return " ".join(
        [
            str(article.get("title") or ""),
            str(article.get("description") or ""),
            str(article.get("content") or ""),
        ]
    ).strip()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _parse_published_at(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def filter_articles(
    articles: list[dict],
    query: str,
    query_emb: np.ndarray,
    kb_centroid_emb: np.ndarray,
    embedder: SentenceTransformer,
) -> tuple[list[dict], dict]:
    stats = {
        "fetched": len(articles),
        "after_language": 0,
        "after_length": 0,
        "after_gibberish": 0,
        "after_relevance": 0,
        "after_dedup": 0,
        "after_recency": 0,
    }

    # Step 1: Language filter
    language_filtered: list[dict] = []
    for article in articles:
        combined = _combined_text(article)
        if not combined:
            continue
        try:
            if detect(combined) == "en":
                article["_combined"] = combined
                language_filtered.append(article)
        except Exception:
            continue

    stats["after_language"] = len(language_filtered)
    logger.info("Fallback filter step language: %d", len(language_filtered))

    # Step 2: Length filter
    length_filtered = [a for a in language_filtered if len(a["_combined"].split()) >= 150]
    stats["after_length"] = len(length_filtered)
    logger.info("Fallback filter step length: %d", len(length_filtered))

    # Step 3: Gibberish filter
    gibberish_filtered: list[dict] = []
    for article in length_filtered:
        combined = article["_combined"]
        letters = sum(1 for c in combined if c.isalpha())
        total_chars = max(len(combined), 1)
        alpha_ratio = letters / total_chars

        words = [w for w in combined.split() if w]
        avg_word_len = float(sum(len(w) for w in words) / max(len(words), 1))

        if alpha_ratio >= 0.70 and avg_word_len <= 15:
            gibberish_filtered.append(article)

    stats["after_gibberish"] = len(gibberish_filtered)
    logger.info("Fallback filter step gibberish: %d", len(gibberish_filtered))

    # Step 4: Relevance filter
    relevance_filtered: list[dict] = []
    if gibberish_filtered:
        article_embeddings = embedder.encode(
            [article["_combined"] for article in gibberish_filtered],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for article, article_emb in zip(gibberish_filtered, article_embeddings):
            query_sim = _cosine_similarity(article_emb, query_emb)
            kb_sim = _cosine_similarity(article_emb, kb_centroid_emb)
            score = (0.6 * query_sim) + (0.4 * kb_sim)
            if score >= settings.NEWS_RELEVANCE_THRESHOLD:
                article["relevance_score"] = score
                relevance_filtered.append(article)

    stats["after_relevance"] = len(relevance_filtered)
    logger.info("Fallback filter step relevance: %d", len(relevance_filtered))

    # Step 5: Deduplication by title embeddings
    deduped: list[dict] = []
    accepted_title_embeddings: list[np.ndarray] = []

    for article in relevance_filtered:
        title = str(article.get("title") or "")
        if not title:
            continue

        title_emb = embedder.encode(
            [title],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        if any(_cosine_similarity(title_emb, prev_emb) > 0.85 for prev_emb in accepted_title_embeddings):
            continue

        deduped.append(article)
        accepted_title_embeddings.append(title_emb)

    stats["after_dedup"] = len(deduped)
    logger.info("Fallback filter step dedup: %d", len(deduped))

    # Step 6: Recency filter
    now = datetime.now(timezone.utc)
    strict_cutoff = now - timedelta(days=settings.NEWS_MAX_AGE_DAYS)
    max_cutoff = now - timedelta(days=365)

    strict_recent: list[dict] = []
    relaxed_recent: list[dict] = []

    for article in deduped:
        published = _parse_published_at(str(article.get("publishedAt") or ""))
        if published is None:
            continue
        if published >= strict_cutoff:
            strict_recent.append(article)
        if published >= max_cutoff:
            relaxed_recent.append(article)

    final_articles = strict_recent if len(strict_recent) >= 2 else relaxed_recent
    stats["after_recency"] = len(final_articles)
    logger.info("Fallback filter step recency: %d", len(final_articles))

    if len(final_articles) < 2:
        logger.warning("Fewer than 2 fallback articles survived filtering")

    for article in final_articles:
        article.pop("_combined", None)

    return final_articles, stats