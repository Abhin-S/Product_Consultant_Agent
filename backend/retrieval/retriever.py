from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import time

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from config import settings
from ingestion.embedder import embed_query
from retrieval.parent_store import get_parent_chunk
from retrieval.vector_store import get_collection


logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    text: str
    source: str
    similarity_score: float
    doc_type: str
    chunk_id: str | None = None
    parent_id: str | None = None
    parent_index: int | None = None
    child_index: int | None = None
    chunk_type: str = "chunk"
    section: str = ""
    pages: str = ""
    has_table: bool = False
    topic: str = ""
    subtopic: str = ""
    source_rel: str = ""


genai.configure(api_key=settings.GOOGLE_API_KEY)
_retrieval_model_cache: dict[str, genai.GenerativeModel] = {}
_retrieval_llm_cooldown_until: float = 0.0


def _candidate_model_names() -> list[str]:
    names = [settings.GEMMA_MODEL_NAME]
    fallback = settings.LLM_FALLBACK_MODEL_NAME.strip()
    if fallback and fallback not in names:
        names.append(fallback)
    return names


def _get_retrieval_model(model_name: str) -> genai.GenerativeModel:
    model = _retrieval_model_cache.get(model_name)
    if model is None:
        model = genai.GenerativeModel(model_name)
        _retrieval_model_cache[model_name] = model
    return model


def _request_options() -> dict | None:
    timeout = int(settings.MODEL_REQUEST_TIMEOUT_SECONDS)
    if timeout > 0:
        return {"timeout": timeout}
    return None


def _generate_with_options(model: genai.GenerativeModel, prompt: str, generation_config: dict) -> object:
    options = _request_options()
    if options is None:
        return model.generate_content(prompt, generation_config=generation_config)
    return model.generate_content(
        prompt,
        generation_config=generation_config,
        request_options=options,
    )


def _json_mode_not_supported(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "response_mime_type" in msg
        or "response mime type" in msg
        or "json mode" in msg
    )


def _is_rate_limited_error(exc: Exception) -> bool:
    if isinstance(exc, google_exceptions.ResourceExhausted):
        return True

    msg = str(exc).lower()
    return (
        "quota exceeded" in msg
        or "rate limit" in msg
        or "resource exhausted" in msg
        or "429" in msg
    )


def _set_retrieval_llm_cooldown(exc: Exception) -> None:
    global _retrieval_llm_cooldown_until

    if not _is_rate_limited_error(exc):
        return

    msg = str(exc)
    retry_seconds = 20.0
    retry_match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", msg, flags=re.IGNORECASE)
    if retry_match:
        try:
            retry_seconds = max(float(retry_match.group(1)), 5.0)
        except ValueError:
            retry_seconds = 20.0

    _retrieval_llm_cooldown_until = max(_retrieval_llm_cooldown_until, time.time() + retry_seconds)


def _retrieval_llm_in_cooldown() -> bool:
    return time.time() < _retrieval_llm_cooldown_until


def _deterministic_query_variants(query: str, multi_query_count: int) -> list[str]:
    """
    Lightweight lexical fallback for query expansion when retrieval LLM helpers are unavailable.
    """
    normalized = " ".join((query or "").split())
    if not normalized:
        return []

    variants: list[str] = [normalized]

    unquoted = re.sub(r"[\"'`]", "", normalized).strip()
    if unquoted and unquoted != normalized:
        variants.append(unquoted)

    parts = [
        part.strip()
        for part in re.split(r"(?i)\band\b|&|/|,", normalized)
        if len(part.strip().split()) >= 2
    ]
    variants.extend(parts)

    stopwords = {
        "about",
        "after",
        "also",
        "and",
        "are",
        "for",
        "from",
        "into",
        "that",
        "the",
        "this",
        "with",
    }
    tokens = re.findall(r"[A-Za-z0-9]+", normalized.lower())
    condensed_tokens: list[str] = []
    for token in tokens:
        if len(token) <= 3 or token in stopwords:
            continue
        if token not in condensed_tokens:
            condensed_tokens.append(token)
    if condensed_tokens:
        variants.append(" ".join(condensed_tokens[:10]))

    variants.append(f"{normalized} case study")
    variants.append(f"{normalized} brand strategy")

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in variants:
        key = candidate.casefold()
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate)

    return unique[: max(multi_query_count + 1, 1)]


def _generate_json_response(prompt: str, temperature: float) -> object:
    last_error: Exception | None = None

    if _retrieval_llm_in_cooldown():
        raise RuntimeError("Retrieval helper LLM is in temporary cooldown after rate limiting")

    for model_name in _candidate_model_names():
        model = _get_retrieval_model(model_name)
        try:
            return _generate_with_options(
                model,
                prompt,
                {"temperature": temperature, "response_mime_type": "application/json"},
            )
        except Exception as exc:
            try:
                if _json_mode_not_supported(exc):
                    return _generate_with_options(model, prompt, {"temperature": temperature})
            except Exception as inner_exc:
                exc = inner_exc

            _set_retrieval_llm_cooldown(exc)
            last_error = exc
            logger.info("Retrieval LLM model fallback from '%s': %s", model_name, exc)
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("No retrieval LLM model names configured")


def _extract_text(response: object) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        merged = "".join(str(getattr(part, "text", "")) for part in parts).strip()
        if merged:
            return merged

    return ""


def _json_candidates(text: str) -> list[str]:
    raw = (text or "").strip()
    candidates: list[str] = []
    if raw:
        candidates.append(raw)

    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE):
        block = block.strip()
        if block:
            candidates.append(block)

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(raw[idx:])
            snippet = raw[idx : idx + end].strip()
            if snippet:
                candidates.append(snippet)
        except Exception:
            continue

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _parse_json_payload(text: str):
    last_error: Exception | None = None
    list_fallback = None
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            if list_fallback is None and isinstance(parsed, list):
                list_fallback = parsed
        except Exception as exc:
            last_error = exc
            continue

    if list_fallback is not None:
        return list_fallback

    raise ValueError(f"No valid JSON payload found in model response: {last_error}")


def _to_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _generate_query_variants(query: str) -> list[str]:
    if settings.BYPASS_LLM_CALLS or not settings.ENABLE_QUERY_EXPANSION:
        return [query]

    if settings.MULTI_QUERY_COUNT <= 1:
        return [query]

    if _retrieval_llm_in_cooldown():
        return _deterministic_query_variants(query, settings.MULTI_QUERY_COUNT)

    prompt = (
        "You are a retrieval query rewriter for product strategy case studies.\n"
        "Generate diverse paraphrases to improve semantic retrieval coverage.\n"
        "Return ONLY JSON with schema: {\"queries\": [\"...\"]}.\n"
        f"Generate {settings.MULTI_QUERY_COUNT} alternatives for this query:\n"
        f"{query}"
    )

    try:
        response = _generate_json_response(prompt, temperature=0.2)
        payload = _parse_json_payload(_extract_text(response))
        generated = payload.get("queries", []) if isinstance(payload, dict) else []
        clean = [q.strip() for q in generated if isinstance(q, str) and q.strip()]
    except Exception as exc:
        logger.info("Query expansion fallback: %s", exc)
        deterministic = _deterministic_query_variants(query, settings.MULTI_QUERY_COUNT)
        clean = deterministic[1:] if len(deterministic) > 1 else []

    unique: list[str] = []
    seen: set[str] = set()
    for q in [query] + clean:
        key = q.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique[: max(settings.MULTI_QUERY_COUNT + 1, 1)]


def _build_corrective_query_variants(query: str, existing_variants: list[str]) -> list[str]:
    existing = {variant.casefold() for variant in existing_variants if variant}
    candidates = _deterministic_query_variants(query, settings.MULTI_QUERY_COUNT + 4)
    candidates.extend(
        [
            f"{query} case study evidence",
            f"{query} brand positioning differentiation",
            f"{query} pricing trust audience",
        ]
    )

    expanded: list[str] = []
    for candidate in candidates:
        normalized = " ".join((candidate or "").split())
        key = normalized.casefold()
        if not normalized or key in existing:
            continue
        existing.add(key)
        expanded.append(normalized)

    return expanded[: max(settings.MULTI_QUERY_COUNT, 2)]


def _retrieve_for_query(query: str, top_k: int) -> list[RetrievedDoc]:
    collection = get_collection()
    query_vector = embed_query(query)

    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    retrieved: list[RetrievedDoc] = []
    for chunk_id, text, metadata, distance in zip(ids, docs, metadatas, distances):
        meta = metadata or {}
        similarity = 1.0 - float(distance)
        parent_index = _to_int(meta.get("parent_index"))
        child_index = _to_int(meta.get("child_index"))
        retrieved.append(
            RetrievedDoc(
                text=text,
                source=str(meta.get("source", "unknown")),
                similarity_score=similarity,
                doc_type=str(meta.get("doc_type", "local")),
                chunk_id=chunk_id,
                parent_id=str(meta.get("parent_id")) if meta.get("parent_id") else None,
                parent_index=parent_index,
                child_index=child_index,
                chunk_type=str(meta.get("chunk_type", "chunk")),
                section=str(meta.get("section", "")),
                pages=str(meta.get("pages", "")),
                has_table=bool(meta.get("has_table", False)),
                topic=str(meta.get("topic", "")),
                subtopic=str(meta.get("subtopic", "")),
                source_rel=str(meta.get("source_rel", "")),
            )
        )

    return retrieved


def _reconstruct_parent_context(docs: list[RetrievedDoc], max_parents: int, top_k: int) -> list[RetrievedDoc]:
    if not docs:
        return docs

    top_children = docs[: max(top_k, 1)]
    parent_candidates: dict[str, RetrievedDoc] = {}

    for doc in top_children:
        if not doc.parent_id:
            continue

        parent_payload = get_parent_chunk(doc.parent_id)
        if not parent_payload:
            continue

        parent_text = str(parent_payload.get("text", "")).strip()
        if not parent_text:
            continue

        current = parent_candidates.get(doc.parent_id)
        candidate = RetrievedDoc(
            text=parent_text,
            source=str(parent_payload.get("source", doc.source)),
            similarity_score=doc.similarity_score,
            doc_type=doc.doc_type,
            chunk_id=f"parent::{doc.parent_id}",
            parent_id=doc.parent_id,
            parent_index=_to_int(parent_payload.get("parent_index")),
            chunk_type="parent",
            section=str(parent_payload.get("section", "")),
            pages=str(parent_payload.get("pages", "")),
            has_table=bool(parent_payload.get("has_table", False)),
            topic=str(parent_payload.get("topic", doc.topic)),
            subtopic=str(parent_payload.get("subtopic", doc.subtopic)),
            source_rel=str(parent_payload.get("source_rel", doc.source_rel)),
        )

        if current is None or candidate.similarity_score > current.similarity_score:
            parent_candidates[doc.parent_id] = candidate

    ranked_parents = sorted(
        parent_candidates.values(), key=lambda item: item.similarity_score, reverse=True
    )[:max_parents]

    combined: list[RetrievedDoc] = []
    seen: set[str] = set()
    for doc in top_children + ranked_parents:
        key = doc.chunk_id or f"{doc.source}:{hash(doc.text)}:{doc.chunk_type}"
        if key in seen:
            continue
        seen.add(key)
        combined.append(doc)

    return combined


def _reciprocal_rank_fusion(ranked_lists: list[list[RetrievedDoc]], top_k: int) -> list[RetrievedDoc]:
    if not ranked_lists:
        return []

    fused_scores: dict[str, float] = {}
    docs_by_id: dict[str, RetrievedDoc] = {}

    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = doc.chunk_id or f"{doc.source}:{hash(doc.text)}"
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (settings.RRF_K + rank)

            existing = docs_by_id.get(key)
            if existing is None or doc.similarity_score > existing.similarity_score:
                docs_by_id[key] = doc

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [docs_by_id[key] for key, _ in ranked[:top_k]]


def _grade_relevance(question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
    if settings.BYPASS_LLM_CALLS or not settings.ENABLE_RELEVANCE_GRADING:
        return docs

    if not docs:
        return docs

    if _retrieval_llm_in_cooldown():
        return docs

    docs_blob = "\n\n".join(
        f"[{idx}] source={doc.source}\n{doc.text[:1200]}" for idx, doc in enumerate(docs, start=1)
    )

    prompt = (
        "You are a strict relevance grader for RAG.\n"
        "Task: select which document excerpts are useful to answer the user question.\n"
        "Return ONLY JSON with schema: {\"relevant_indices\": [1,2,...]}.\n"
        "Only include indices that are clearly relevant.\n\n"
        f"Question:\n{question}\n\n"
        f"Documents:\n{docs_blob}"
    )

    try:
        response = _generate_json_response(prompt, temperature=0)
        payload = _parse_json_payload(_extract_text(response))
        raw_indices = payload.get("relevant_indices", []) if isinstance(payload, dict) else []

        relevant_indices = sorted(
            {
                int(i)
                for i in raw_indices
                if isinstance(i, int) and 1 <= i <= len(docs)
            }
        )
    except Exception as exc:
        _set_retrieval_llm_cooldown(exc)
        logger.info("Relevance grading fallback: %s", exc)
        return docs

    filtered = [docs[i - 1] for i in relevant_indices]
    if len(filtered) < min(settings.CRAG_MIN_RELEVANT_DOCS, len(docs)):
        return docs
    return filtered


def retrieve_local(query: str, top_k: int) -> tuple[list[RetrievedDoc], bool, dict]:
    query_variants = _generate_query_variants(query)

    ranked_lists: list[list[RetrievedDoc]] = []
    for variant in query_variants:
        try:
            ranked_lists.append(_retrieve_for_query(variant, top_k=top_k))
        except Exception as exc:
            logger.warning("Retrieval failed for variant '%s': %s", variant[:80], exc)

    fused_docs = _reciprocal_rank_fusion(ranked_lists, top_k=top_k)
    graded_docs = _grade_relevance(query, fused_docs)
    initial_retrieved_before_crag = len(fused_docs)
    initial_retrieved_after_crag = len(graded_docs)

    corrective_pass_used = False
    corrective_variants: list[str] = []
    min_relevant_target = max(1, min(settings.CRAG_MIN_RELEVANT_DOCS, top_k))
    if len(graded_docs) < min_relevant_target:
        corrective_variants = _build_corrective_query_variants(query, query_variants)
        corrective_lists: list[list[RetrievedDoc]] = []
        for variant in corrective_variants:
            try:
                corrective_lists.append(_retrieve_for_query(variant, top_k=top_k))
            except Exception as exc:
                logger.warning("Corrective retrieval failed for variant '%s': %s", variant[:80], exc)

        if corrective_lists:
            widened_top_k = max(top_k + settings.MAX_PARENT_CONTEXT_CHUNKS, top_k)
            fused_docs = _reciprocal_rank_fusion([fused_docs, *corrective_lists], top_k=widened_top_k)
            graded_docs = _grade_relevance(query, fused_docs)
            corrective_pass_used = True

    parent_docs = _reconstruct_parent_context(
        graded_docs,
        max_parents=max(settings.MAX_PARENT_CONTEXT_CHUNKS, 1),
        top_k=top_k,
    )

    max_similarity = max((doc.similarity_score for doc in parent_docs), default=0.0)
    low_confidence = max_similarity < settings.CONFIDENCE_THRESHOLD

    diagnostics = {
        "query_variants": query_variants,
        "corrective_query_variants": corrective_variants,
        "corrective_pass_used": corrective_pass_used,
        "retrieved_before_crag": initial_retrieved_before_crag,
        "retrieved_after_initial_crag": initial_retrieved_after_crag,
        "retrieved_after_crag": len(graded_docs),
        "child_context_docs": len([doc for doc in parent_docs if doc.chunk_type != "parent"]),
        "parent_context_docs": len([doc for doc in parent_docs if doc.chunk_type == "parent"]),
    }

    return parent_docs, low_confidence, diagnostics