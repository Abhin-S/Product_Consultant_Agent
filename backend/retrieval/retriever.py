from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import logging
import math
import re
import time

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from config import settings
from ingestion.embedder import embed_query
from retrieval.parent_store import get_parent_chunk
from retrieval.vector_store import get_collection


logger = logging.getLogger(__name__)


RETRIEVAL_OVERSAMPLE_FACTOR = 6
RETRIEVAL_OVERSAMPLE_MAX = 60
MAX_CHUNKS_PER_SOURCE = 2
LEXICAL_TOP_K_MULTIPLIER = 4

_DEFAULT_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "how", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "what",
    "when", "where", "which", "who", "why", "with", "would", "should", "could", "can",
}

_ENTITY_ALIAS_RULES: tuple[tuple[str, str], ...] = (
    (r"\btata communications?\b", "tata communication"),
    (r"\btata comms?\b", "tata communication"),
    (r"\bbusiness to business\b", "b2b"),
)


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
    rerank_score: float | None = None


genai.configure(api_key=settings.GOOGLE_API_KEY)
_retrieval_model_cache: dict[str, genai.GenerativeModel] = {}
_retrieval_llm_cooldown_until: float = 0.0
_cross_encoder = None
_cross_encoder_unavailable = False
_lexical_index_cache: dict[str, object] = {}


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


def _is_valid_query_variant(candidate: str, original_query: str) -> bool:
    normalized = " ".join((candidate or "").split())
    if not normalized:
        return False

    lowered = normalized.casefold()
    if lowered in {"...", "..", ".", "n/a", "none", "null", "unknown"}:
        return False

    if lowered == " ".join((original_query or "").split()).casefold():
        return False

    if len(normalized) < 8:
        return False

    if not re.search(r"[a-z0-9]", normalized, flags=re.IGNORECASE):
        return False

    token_count = len(re.findall(r"[A-Za-z0-9]+", normalized))
    if token_count < 2:
        return False

    return True


def _normalize_query_text(query: str) -> str:
    return " ".join((query or "").split()).strip()


def _classify_query_intent(query: str) -> str:
    text = _normalize_query_text(query).lower()
    if not text:
        return "analytical"

    factual_markers = (
        "what is", "what are", "who is", "when did", "where is", "define ",
    )
    inferential_markers = (
        "implication", "infer", "scenario", "if ", "could", "might", "trade-off", "tradeoff",
    )
    analytical_markers = (
        "how should", "principle", "guidance", "strategy", "balance", "compare", "evaluate",
        "recommend", "position", "pricing", "trust",
    )

    if any(marker in text for marker in factual_markers):
        return "factual"
    if any(marker in text for marker in inferential_markers):
        return "inferential"
    if any(marker in text for marker in analytical_markers):
        return "analytical"
    return "analytical"


def _extract_content_terms(text: str, max_terms: int = 12) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    seen: set[str] = set()
    terms: list[str] = []

    for token in tokens:
        if len(token) <= 2 or token in _DEFAULT_QUERY_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break

    return terms


def _extract_entity_hints(query: str) -> list[str]:
    normalized = _normalize_query_text(query)
    lowered = normalized.lower()
    entities: list[str] = []

    for pattern, canonical in _ENTITY_ALIAS_RULES:
        if re.search(pattern, lowered):
            entities.append(canonical)

    title_case_entities = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", normalized)
    for entity in title_case_entities:
        compact = " ".join(entity.split()).strip().lower()
        if compact and compact not in entities:
            entities.append(compact)

    return entities[:6]


def _entity_normalization_variants(query: str) -> list[str]:
    normalized = _normalize_query_text(query)
    if not normalized:
        return []

    lowered = normalized.lower()
    variants: list[str] = []
    for pattern, canonical in _ENTITY_ALIAS_RULES:
        if re.search(pattern, lowered):
            replaced = re.sub(pattern, canonical, lowered)
            if replaced and replaced != lowered:
                variants.append(replaced)

    return variants[:3]


def _expand_related_concepts(query: str, query_intent: str, entities: list[str]) -> list[str]:
    normalized = _normalize_query_text(query)
    lowered = normalized.lower()
    expansions: list[str] = []

    if "pricing" in lowered:
        expansions.extend(
            [
                f"{normalized} value-based pricing",
                f"{normalized} discount governance",
                f"{normalized} price signaling trust",
            ]
        )

    if "b2b" in lowered or "business" in lowered:
        expansions.extend(
            [
                f"{normalized} enterprise buyer value quantification",
                f"{normalized} procurement decision criteria",
            ]
        )

    if query_intent == "inferential":
        expansions.append(f"{normalized} implications and trade-offs")
    elif query_intent == "factual":
        expansions.append(f"{normalized} key facts and evidence")
    else:
        expansions.append(f"{normalized} strategic principles and practical guidance")

    for entity in entities[:2]:
        expansions.append(f"{entity} pricing strategy trust")

    return expansions[:6]


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
    normalized_query = _normalize_query_text(query)
    query_intent = _classify_query_intent(normalized_query)
    entity_hints = _extract_entity_hints(normalized_query)

    if settings.BYPASS_LLM_CALLS or not settings.ENABLE_QUERY_EXPANSION:
        fallback = [normalized_query]
        fallback.extend(_entity_normalization_variants(normalized_query))
        unique: list[str] = []
        seen: set[str] = set()
        for variant in fallback:
            key = variant.casefold()
            if variant and key not in seen:
                seen.add(key)
                unique.append(variant)
        return unique[:2] or [normalized_query]

    if settings.MULTI_QUERY_COUNT <= 1:
        return [normalized_query]

    if _retrieval_llm_in_cooldown():
        deterministic = _deterministic_query_variants(normalized_query, settings.MULTI_QUERY_COUNT)
        deterministic.extend(_entity_normalization_variants(normalized_query))
        deterministic.extend(_expand_related_concepts(normalized_query, query_intent, entity_hints))
        unique: list[str] = []
        seen: set[str] = set()
        for variant in deterministic:
            normalized = _normalize_query_text(variant)
            key = normalized.casefold()
            if normalized and key not in seen and _is_valid_query_variant(normalized, normalized_query):
                seen.add(key)
                unique.append(normalized)
        return [normalized_query, *unique][: max(settings.MULTI_QUERY_COUNT + 2, 3)]

    prompt = (
        "You are a retrieval query rewriter for product strategy case studies.\n"
        "Generate diverse paraphrases to improve semantic retrieval coverage.\n"
        "Return ONLY JSON with schema: {\"queries\": [\"...\"]}.\n"
        f"Generate {settings.MULTI_QUERY_COUNT} alternatives for this query:\n"
        f"{normalized_query}"
    )

    try:
        response = _generate_json_response(prompt, temperature=0.2)
        payload = _parse_json_payload(_extract_text(response))
        generated = payload.get("queries", []) if isinstance(payload, dict) else []
        clean = [q.strip() for q in generated if isinstance(q, str) and q.strip()]
    except Exception as exc:
        logger.info("Query expansion fallback: %s", exc)
        deterministic = _deterministic_query_variants(normalized_query, settings.MULTI_QUERY_COUNT)
        clean = deterministic[1:] if len(deterministic) > 1 else []

    filtered_clean = [
        " ".join(candidate.split())
        for candidate in clean
        if _is_valid_query_variant(candidate, normalized_query)
    ]

    filtered_clean.extend(_entity_normalization_variants(normalized_query))
    filtered_clean.extend(_expand_related_concepts(normalized_query, query_intent, entity_hints))

    if not filtered_clean and settings.MULTI_QUERY_COUNT > 1:
        deterministic = _deterministic_query_variants(normalized_query, settings.MULTI_QUERY_COUNT)
        filtered_clean = [
            variant
            for variant in deterministic[1:]
            if _is_valid_query_variant(variant, normalized_query)
        ]

    unique: list[str] = []
    seen: set[str] = set()
    for q in [normalized_query] + filtered_clean:
        q = _normalize_query_text(q)
        if not q:
            continue
        key = q.casefold()
        if key not in seen and (q == normalized_query or _is_valid_query_variant(q, normalized_query)):
            seen.add(key)
            unique.append(q)

    return unique[: max(settings.MULTI_QUERY_COUNT + 2, 3)]


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


def _tokenize_for_bm25(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    return [token for token in tokens if len(token) > 2 and token not in _DEFAULT_QUERY_STOPWORDS]


def _flatten_get_field(value: object) -> list:
    if not isinstance(value, list):
        return []
    if value and isinstance(value[0], list):
        return value[0]
    return value


def _build_retrieved_doc(
    *,
    chunk_id: str,
    text: str,
    meta: dict,
    similarity: float,
    rerank_score: float | None = None,
) -> RetrievedDoc:
    parent_index = _to_int(meta.get("parent_index"))
    child_index = _to_int(meta.get("child_index"))
    return RetrievedDoc(
        text=text,
        source=str(meta.get("source", "unknown")),
        similarity_score=float(max(0.0, min(1.0, similarity))),
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
        rerank_score=rerank_score,
    )


def _ensure_lexical_index() -> None:
    global _lexical_index_cache

    collection = get_collection()
    current_count = int(collection.count() or 0)
    cached_count = int(_lexical_index_cache.get("count", -1)) if _lexical_index_cache else -1
    if _lexical_index_cache and cached_count == current_count:
        return

    try:
        payload = collection.get(where={"doc_type": "local"}, include=["documents", "metadatas"])
    except Exception:
        payload = collection.get(include=["documents", "metadatas"])

    ids = _flatten_get_field(payload.get("ids", []))
    documents = _flatten_get_field(payload.get("documents", []))
    metadatas = _flatten_get_field(payload.get("metadatas", []))

    total = min(len(ids), len(documents), len(metadatas))
    ids = ids[:total]
    documents = documents[:total]
    metadatas = metadatas[:total]

    filtered_ids: list[str] = []
    filtered_docs: list[str] = []
    filtered_metas: list[dict] = []
    for chunk_id, doc_text, meta in zip(ids, documents, metadatas):
        meta_dict = meta if isinstance(meta, dict) else {}
        doc_type = str(meta_dict.get("doc_type", "local"))
        if doc_type != "local":
            continue
        filtered_ids.append(str(chunk_id))
        filtered_docs.append(str(doc_text or ""))
        filtered_metas.append(meta_dict)

    ids = filtered_ids
    documents = filtered_docs
    metadatas = filtered_metas

    tokenized_docs: list[list[str]] = []
    term_frequencies: list[Counter[str]] = []
    document_frequencies: Counter[str] = Counter()
    doc_lengths: list[int] = []

    for text in documents:
        tokens = _tokenize_for_bm25(str(text or ""))
        tf = Counter(tokens)
        tokenized_docs.append(tokens)
        term_frequencies.append(tf)
        doc_lengths.append(max(len(tokens), 1))
        for term in tf.keys():
            document_frequencies[term] += 1

    avg_doc_len = float(sum(doc_lengths) / max(len(doc_lengths), 1))

    _lexical_index_cache = {
        "count": current_count,
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "tokenized_docs": tokenized_docs,
        "term_frequencies": term_frequencies,
        "document_frequencies": document_frequencies,
        "doc_lengths": doc_lengths,
        "avg_doc_len": avg_doc_len,
    }

    if settings.RAG_DEBUG_MODE:
        logger.info(
            "lexical_index_rebuilt doc_count=%d avg_doc_len=%.2f unique_terms=%d",
            len(documents),
            avg_doc_len,
            len(document_frequencies),
        )


def _bm25_score(query_terms: list[str], tf: Counter[str], doc_len: int, n_docs: int, avg_doc_len: float, df: Counter[str]) -> float:
    if not query_terms or n_docs <= 0:
        return 0.0

    k1 = float(settings.BM25_K1)
    b = float(settings.BM25_B)
    doc_len_f = float(max(doc_len, 1))
    avg_len_f = float(max(avg_doc_len, 1.0))

    score = 0.0
    for term in query_terms:
        term_df = int(df.get(term, 0))
        if term_df <= 0:
            continue

        idf = math.log(1.0 + ((n_docs - term_df + 0.5) / (term_df + 0.5)))
        term_tf = float(tf.get(term, 0))
        if term_tf <= 0:
            continue

        denom = term_tf + k1 * (1.0 - b + b * (doc_len_f / avg_len_f))
        if denom <= 0:
            continue
        score += idf * ((term_tf * (k1 + 1.0)) / denom)

    return score


def _retrieve_lexical_for_query(query: str, top_k: int) -> list[RetrievedDoc]:
    if not settings.ENABLE_HYBRID_RETRIEVAL:
        return []

    _ensure_lexical_index()
    if not _lexical_index_cache:
        return []

    query_terms = _extract_content_terms(query, max_terms=20)
    if not query_terms:
        return []

    ids = _lexical_index_cache.get("ids", [])
    documents = _lexical_index_cache.get("documents", [])
    metadatas = _lexical_index_cache.get("metadatas", [])
    term_frequencies = _lexical_index_cache.get("term_frequencies", [])
    doc_lengths = _lexical_index_cache.get("doc_lengths", [])
    document_frequencies = _lexical_index_cache.get("document_frequencies", Counter())
    avg_doc_len = float(_lexical_index_cache.get("avg_doc_len", 1.0) or 1.0)

    n_docs = len(ids)
    scored: list[tuple[int, float]] = []
    for idx, tf in enumerate(term_frequencies):
        score = _bm25_score(
            query_terms,
            tf,
            int(doc_lengths[idx]) if idx < len(doc_lengths) else 1,
            n_docs,
            avg_doc_len,
            document_frequencies,
        )
        if score > 0:
            scored.append((idx, score))

    if not scored:
        return []

    scored.sort(key=lambda item: item[1], reverse=True)
    candidate_k = max(
        top_k,
        min(max(top_k * LEXICAL_TOP_K_MULTIPLIER, top_k), int(settings.LEXICAL_OVERSAMPLE_CAP)),
    )
    top_scored = scored[:candidate_k]

    max_score = top_scored[0][1]
    min_score = top_scored[-1][1]
    spread = max(max_score - min_score, 1e-9)

    lexical_docs: list[RetrievedDoc] = []
    for rank, (idx, score) in enumerate(top_scored, start=1):
        meta = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
        chunk_id = str(ids[idx]) if idx < len(ids) else f"lexical::{idx}"
        text = str(documents[idx]) if idx < len(documents) else ""

        normalized = (score - min_score) / spread
        rank_bias = 1.0 - ((rank - 1) / max(len(top_scored), 1))
        similarity = 0.2 + 0.6 * normalized + 0.2 * rank_bias

        lexical_docs.append(
            _build_retrieved_doc(
                chunk_id=chunk_id,
                text=text,
                meta=meta,
                similarity=similarity,
            )
        )

    return lexical_docs


def _retrieve_dense_for_query(query: str, top_k: int) -> list[RetrievedDoc]:
    collection = get_collection()
    query_vector = embed_query(query)
    candidate_k = max(top_k, min(max(top_k * RETRIEVAL_OVERSAMPLE_FACTOR, top_k), RETRIEVAL_OVERSAMPLE_MAX))

    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=candidate_k,
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
        retrieved.append(
            _build_retrieved_doc(
                chunk_id=str(chunk_id),
                text=str(text),
                meta=meta,
                similarity=similarity,
            )
        )

    return retrieved


def _retrieve_for_query(query: str, top_k: int) -> list[RetrievedDoc]:
    return _retrieve_dense_for_query(query, top_k=top_k)


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
    min_target = min(settings.CRAG_MIN_RELEVANT_DOCS, len(docs))
    if not filtered:
        return docs[:min_target]

    if len(filtered) < min_target:
        selected_ids = {doc.chunk_id for doc in filtered}
        for doc in docs:
            if doc.chunk_id in selected_ids:
                continue
            filtered.append(doc)
            if len(filtered) >= min_target:
                break

    return filtered


def _text_overlap_score(query: str, doc_text: str) -> float:
    query_terms = set(_extract_content_terms(query, max_terms=14))
    if not query_terms:
        return 0.0

    doc_terms = set(_extract_content_terms(doc_text, max_terms=180))
    if not doc_terms:
        return 0.0

    overlap = len(query_terms.intersection(doc_terms))
    return float(overlap / max(len(query_terms), 1))


def _metadata_alignment_score(query: str, doc: RetrievedDoc) -> float:
    query_terms = _extract_content_terms(query, max_terms=14)
    if not query_terms:
        return 0.0

    meta_blob = " ".join(
        [
            (doc.source_rel or ""),
            (doc.topic or ""),
            (doc.subtopic or ""),
            (doc.section or ""),
        ]
    ).lower()

    entity_hits = 0
    for entity in _extract_entity_hints(query):
        if entity in meta_blob:
            entity_hits += 1

    term_hits = sum(1 for term in query_terms if term in meta_blob)
    term_component = float(term_hits / max(len(query_terms), 1))
    entity_component = float(min(entity_hits, 2) / 2.0)
    return max(term_component, entity_component)


def _get_cross_encoder():
    global _cross_encoder, _cross_encoder_unavailable

    if not settings.ENABLE_CROSS_ENCODER_RERANK:
        return None
    if _cross_encoder_unavailable:
        return None
    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL_NAME)
        return _cross_encoder
    except Exception as exc:
        _cross_encoder_unavailable = True
        logger.warning("Cross-encoder reranker unavailable: %s", exc)
        return None


def _cross_encoder_scores(query: str, docs: list[RetrievedDoc]) -> list[float] | None:
    model = _get_cross_encoder()
    if model is None or not docs:
        return None

    pairs = [[query, doc.text[:1800]] for doc in docs]
    try:
        raw_scores = model.predict(pairs)
    except Exception as exc:
        logger.warning("Cross-encoder rerank failed; falling back to heuristic rerank: %s", exc)
        return None

    values = [float(score) for score in raw_scores]
    max_value = max(values, default=0.0)
    min_value = min(values, default=0.0)
    spread = max(max_value - min_value, 1e-9)
    return [float((value - min_value) / spread) for value in values]


def _rerank_docs(query: str, docs: list[RetrievedDoc], top_n: int) -> list[RetrievedDoc]:
    if not docs:
        return []

    candidates = docs[: max(1, min(int(settings.RERANK_MAX_CANDIDATES), len(docs)))]
    ce_scores = _cross_encoder_scores(query, candidates)

    scored_docs: list[tuple[float, RetrievedDoc]] = []
    for idx, doc in enumerate(candidates):
        lexical_overlap = _text_overlap_score(query, doc.text)
        metadata_match = _metadata_alignment_score(query, doc)
        base_similarity = float(max(0.0, min(1.0, doc.similarity_score)))

        if ce_scores is not None and idx < len(ce_scores):
            rerank_score = (0.55 * ce_scores[idx]) + (0.30 * base_similarity) + (0.15 * lexical_overlap)
        else:
            rerank_score = (0.60 * base_similarity) + (0.25 * lexical_overlap) + (0.15 * metadata_match)

        doc.rerank_score = float(max(0.0, min(1.0, rerank_score)))
        scored_docs.append((doc.rerank_score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    reranked = [doc for _, doc in scored_docs[:top_n]]

    # Keep remaining tail documents in original order so downstream fallbacks still have coverage.
    reranked_ids = {doc.chunk_id for doc in reranked}
    tail = [doc for doc in docs if doc.chunk_id not in reranked_ids]
    return reranked + tail


def _diversify_by_source(
    docs: list[RetrievedDoc],
    *,
    top_k: int,
    max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
) -> list[RetrievedDoc]:
    if not docs:
        return []

    if max_chunks_per_source <= 0:
        return docs[:top_k]

    selected: list[RetrievedDoc] = []
    per_source_counts: dict[str, int] = {}

    for doc in docs:
        source_key = (doc.source_rel or doc.source or "").strip().casefold()
        if not source_key:
            source_key = f"unknown:{id(doc)}"

        used = per_source_counts.get(source_key, 0)
        if used >= max_chunks_per_source:
            continue

        per_source_counts[source_key] = used + 1
        selected.append(doc)

        if len(selected) >= top_k:
            break

    if selected:
        return selected

    return docs[:top_k]


def retrieve_local(query: str, top_k: int) -> tuple[list[RetrievedDoc], bool, dict]:
    normalized_query = _normalize_query_text(query)
    query_intent = _classify_query_intent(normalized_query)
    entity_hints = _extract_entity_hints(normalized_query)
    query_variants = _generate_query_variants(normalized_query)

    ranked_lists: list[list[RetrievedDoc]] = []
    dense_candidates = 0
    lexical_candidates = 0

    for variant in query_variants:
        try:
            dense_docs = _retrieve_dense_for_query(variant, top_k=top_k)
            if dense_docs:
                ranked_lists.append(dense_docs)
                dense_candidates += len(dense_docs)

            lexical_docs = _retrieve_lexical_for_query(variant, top_k=top_k)
            if lexical_docs:
                ranked_lists.append(lexical_docs)
                lexical_candidates += len(lexical_docs)
        except Exception as exc:
            logger.warning("Retrieval failed for variant '%s': %s", variant[:120], exc)

    fused_top_k = max(top_k + settings.MAX_PARENT_CONTEXT_CHUNKS, top_k)
    fused_docs = _reciprocal_rank_fusion(ranked_lists, top_k=fused_top_k)
    graded_docs = _grade_relevance(normalized_query, fused_docs)
    reranked_docs = _rerank_docs(normalized_query, graded_docs, top_n=max(top_k * 2, top_k))

    initial_retrieved_before_crag = len(fused_docs)
    initial_retrieved_after_crag = len(graded_docs)

    corrective_pass_used = False
    corrective_variants: list[str] = []
    min_relevant_target = max(1, min(settings.CRAG_MIN_RELEVANT_DOCS, top_k))
    if len(reranked_docs) < min_relevant_target:
        corrective_variants = _build_corrective_query_variants(normalized_query, query_variants)
        corrective_lists: list[list[RetrievedDoc]] = []

        for variant in corrective_variants:
            try:
                dense_docs = _retrieve_dense_for_query(variant, top_k=top_k)
                if dense_docs:
                    corrective_lists.append(dense_docs)
                    dense_candidates += len(dense_docs)

                lexical_docs = _retrieve_lexical_for_query(variant, top_k=top_k)
                if lexical_docs:
                    corrective_lists.append(lexical_docs)
                    lexical_candidates += len(lexical_docs)
            except Exception as exc:
                logger.warning("Corrective retrieval failed for variant '%s': %s", variant[:120], exc)

        if corrective_lists:
            widened_top_k = max(top_k + settings.MAX_PARENT_CONTEXT_CHUNKS + 2, fused_top_k)
            fused_docs = _reciprocal_rank_fusion([fused_docs, *corrective_lists], top_k=widened_top_k)
            graded_docs = _grade_relevance(normalized_query, fused_docs)
            reranked_docs = _rerank_docs(normalized_query, graded_docs, top_n=max(top_k * 2, top_k))
            corrective_pass_used = True

    diversified_docs = _diversify_by_source(
        reranked_docs,
        top_k=max(top_k * 2, top_k),
        max_chunks_per_source=MAX_CHUNKS_PER_SOURCE,
    )
    docs_for_parent_context = diversified_docs if diversified_docs else reranked_docs

    parent_docs = _reconstruct_parent_context(
        docs_for_parent_context,
        max_parents=max(settings.MAX_PARENT_CONTEXT_CHUNKS, 1),
        top_k=top_k,
    )

    confidence_proxies = [
        (0.70 * float(doc.similarity_score))
        + (0.30 * float(doc.rerank_score if doc.rerank_score is not None else doc.similarity_score))
        for doc in parent_docs
    ]
    max_retrieval_score = max(confidence_proxies, default=0.0)
    low_confidence = max_retrieval_score < settings.CONFIDENCE_THRESHOLD

    if settings.RAG_DEBUG_MODE:
        logger.info(
            "retrieve_local intent=%s variants=%d fused=%d graded=%d reranked=%d diversified=%d low_conf=%s",
            query_intent,
            len(query_variants),
            len(fused_docs),
            len(graded_docs),
            len(reranked_docs),
            len(diversified_docs),
            low_confidence,
        )

    diagnostics = {
        "query_intent": query_intent,
        "entity_hints": entity_hints,
        "query_variants": query_variants,
        "corrective_query_variants": corrective_variants,
        "corrective_pass_used": corrective_pass_used,
        "hybrid_retrieval_enabled": bool(settings.ENABLE_HYBRID_RETRIEVAL),
        "dense_candidates": dense_candidates,
        "lexical_candidates": lexical_candidates,
        "retrieved_before_crag": initial_retrieved_before_crag,
        "retrieved_after_initial_crag": initial_retrieved_after_crag,
        "retrieved_after_crag": len(graded_docs),
        "retrieved_after_rerank": len(reranked_docs),
        "retrieved_after_diversification": len(diversified_docs),
        "child_context_docs": len([doc for doc in parent_docs if doc.chunk_type != "parent"]),
        "parent_context_docs": len([doc for doc in parent_docs if doc.chunk_type == "parent"]),
        "max_retrieval_score": round(float(max_retrieval_score), 4),
    }

    return parent_docs, low_confidence, diagnostics