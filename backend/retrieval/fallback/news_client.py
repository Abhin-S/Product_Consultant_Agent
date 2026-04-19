from __future__ import annotations

import logging

import httpx

from config import settings


logger = logging.getLogger(__name__)


def _normalize_article(article: dict) -> dict:
    return {
        "title": article.get("title") or "",
        "description": article.get("description") or "",
        "content": article.get("content") or "",
        "url": article.get("url") or "",
        "publishedAt": article.get("publishedAt") or article.get("published_at") or "",
    }


async def _fetch_newsapi(query: str) -> list[dict]:
    if not settings.NEWS_API_KEY:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 10,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params, headers={"X-Api-Key": settings.NEWS_API_KEY})
        if response.status_code == 429:
            logger.warning("NewsAPI rate limited (429); returning empty list")
            return []
        response.raise_for_status()
        payload = response.json()
        return [_normalize_article(article) for article in payload.get("articles", [])]
    except Exception as exc:
        logger.warning("NewsAPI request failed: %s", exc)
        return []


async def _fetch_gnews(query: str) -> list[dict]:
    if not settings.GNEWS_API_KEY:
        return []

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "sortby": "relevance",
        "max": 10,
        "apikey": settings.GNEWS_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
        if response.status_code == 429:
            logger.warning("GNews rate limited (429); returning empty list")
            return []
        response.raise_for_status()
        payload = response.json()
        return [_normalize_article(article) for article in payload.get("articles", [])]
    except Exception as exc:
        logger.warning("GNews request failed: %s", exc)
        return []


async def fetch_news_articles(query: str) -> list[dict]:
    primary = await _fetch_newsapi(query)

    articles = list(primary)
    if settings.GNEWS_API_KEY and len(primary) < 3:
        secondary = await _fetch_gnews(query)
        seen_urls = {article.get("url") for article in articles}
        for article in secondary:
            if article.get("url") not in seen_urls:
                articles.append(article)

    return articles