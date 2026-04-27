from __future__ import annotations

from collections import defaultdict
import json
import logging
import re
import time
from statistics import mean

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import tiktoken
from fastapi import HTTPException
from pydantic import ValidationError

from config import settings
from reasoning.prompts import FALLBACK_NOTE, NO_FALLBACK_NOTE, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from reasoning.schema import InsightOutput, NotionDatabaseMetadata
from retrieval.fallback.context_builder import ContextBundle


logger = logging.getLogger(__name__)
ENCODING = tiktoken.get_encoding("cl100k_base")


genai.configure(api_key=settings.GOOGLE_API_KEY)
_model_cache: dict[str, genai.GenerativeModel] = {}


def _candidate_model_names() -> list[str]:
    names = [settings.GEMMA_MODEL_NAME]
    fallback = settings.LLM_FALLBACK_MODEL_NAME.strip()
    if fallback and fallback not in names:
        names.append(fallback)
    return names


def _get_model(model_name: str) -> genai.GenerativeModel:
    model = _model_cache.get(model_name)
    if model is None:
        model = genai.GenerativeModel(model_name)
        _model_cache[model_name] = model
    return model


def _token_len(text: str) -> int:
    return len(ENCODING.encode(text or ""))


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

    raise ValueError("No response text returned from model")


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


_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "what", "when",
    "where", "which", "who", "why", "with", "would", "should", "could", "can",
}


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _extract_terms(value: str, max_terms: int = 14) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", (value or "").lower())
    seen: set[str] = set()
    terms: list[str] = []
    for token in tokens:
        if len(token) <= 2 or token in _QUERY_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def _classify_query_intent(query: str) -> str:
    text = _normalize_text(query).lower()
    if not text:
        return "analytical"

    if any(marker in text for marker in ("what is", "what are", "who is", "when did", "where is", "define ")):
        return "factual"
    if any(marker in text for marker in ("implication", "scenario", "if ", "might", "could", "trade-off", "tradeoff")):
        return "inferential"
    return "analytical"


def _extract_entity_hints(query: str) -> list[str]:
    normalized = _normalize_text(query)
    entities: list[str] = []

    lowered = normalized.lower()
    if re.search(r"\btata communications?\b", lowered):
        entities.append("tata communication")
    if re.search(r"\btata comms?\b", lowered):
        entities.append("tata communication")

    title_case_entities = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", normalized)
    for entity in title_case_entities:
        compact = " ".join(entity.split()).strip().lower()
        if compact and compact not in entities:
            entities.append(compact)

    return entities[:6]


def _build_context_text(bundle: ContextBundle) -> str:
    grouped: dict[str, list] = defaultdict(list)
    group_order: list[str] = []

    for doc in bundle.docs:
        source_key = (doc.source_rel or doc.source or "unknown").strip()
        if source_key not in grouped:
            group_order.append(source_key)
        grouped[source_key].append(doc)

    lines: list[str] = []
    source_idx = 0
    for source_key in group_order:
        docs = grouped[source_key]
        if not docs:
            continue

        source_idx += 1
        lead = docs[0]
        doc_header_parts = [
            f"source={lead.source}",
            f"source_rel={lead.source_rel or source_key}",
            f"topic={lead.topic or 'n/a'}",
            f"subtopic={lead.subtopic or 'n/a'}",
            f"doc_type={lead.doc_type}",
        ]
        lines.append(f"[Document {source_idx}] {' '.join(doc_header_parts)}")

        ranked_docs = sorted(
            docs,
            key=lambda item: (float(item.similarity_score), float(item.rerank_score or 0.0)),
            reverse=True,
        )

        for chunk_idx, doc in enumerate(ranked_docs, start=1):
            chunk_meta = [
                f"chunk_type={doc.chunk_type}",
                f"similarity={doc.similarity_score:.4f}",
            ]
            if doc.rerank_score is not None:
                chunk_meta.append(f"rerank={doc.rerank_score:.4f}")
            if doc.section:
                chunk_meta.append(f"section={doc.section}")
            if doc.pages:
                chunk_meta.append(f"pages={doc.pages}")
            if doc.has_table:
                chunk_meta.append("has_table=true")

            lines.append(f"  [Chunk {source_idx}.{chunk_idx}] {' '.join(chunk_meta)}\n{doc.text}")

    return "\n\n".join(lines)


def _retrieval_confidence_signal(context_bundle: ContextBundle) -> float:
    similarities = [max(0.0, min(1.0, float(doc.similarity_score))) for doc in context_bundle.docs]
    if not similarities:
        return 0.0

    top_n = min(4, len(similarities))
    top_mean = mean(sorted(similarities, reverse=True)[:top_n])
    strong_docs = len([score for score in similarities if score >= settings.CONFIDENCE_THRESHOLD])
    coverage = min(1.0, strong_docs / max(top_n, 1))
    return float(max(0.0, min(1.0, (0.7 * top_mean) + (0.3 * coverage))))


def _generation_confidence_signal(insight: InsightOutput) -> float:
    if _clean_text(insight.abstention_message):
        return 0.1

    primary_fields = [
        insight.brand_diagnosis,
        insight.market_insight,
        insight.final_positioning,
        insight.target_audience,
        insight.chosen_strategy,
    ]
    filled = len([field for field in primary_fields if _clean_text(field)])
    fill_ratio = filled / len(primary_fields)

    evidence_blob = " ".join(
        [
            _clean_text(insight.brand_diagnosis),
            _clean_text(insight.market_insight),
            " ".join(_clean_text_list(insight.suggested_positioning, max_items=6, max_len=180)),
            " ".join(_clean_text_list(insight.risks, max_items=6, max_len=180)),
            " ".join(_clean_text_list(insight.opportunities, max_items=6, max_len=180)),
        ]
    )
    evidence_tags = len(re.findall(r"\[source=.*?\]", evidence_blob, flags=re.IGNORECASE))
    evidence_signal = min(1.0, evidence_tags / 3.0)

    return float(max(0.0, min(1.0, (0.7 * fill_ratio) + (0.3 * evidence_signal))))


def _calibrate_confidence(
    context_bundle: ContextBundle,
    insight: InsightOutput,
    *,
    grounding_verdict: str | None = None,
) -> float:
    retrieval_signal = _retrieval_confidence_signal(context_bundle)

    if _clean_text(insight.abstention_message):
        return float(max(0.05, min(0.25, retrieval_signal * 0.45)))

    generation_signal = _generation_confidence_signal(insight)
    confidence = (0.75 * retrieval_signal) + (0.25 * generation_signal)

    if grounding_verdict == "partial":
        confidence *= 0.9
    elif grounding_verdict == "not_grounded":
        confidence *= 0.35

    return float(max(0.0, min(1.0, confidence)))


def _should_retry_after_abstention(idea: str, context_bundle: ContextBundle, insight: InsightOutput) -> bool:
    if not _clean_text(insight.abstention_message):
        return False

    metrics = _context_coverage_metrics(context_bundle)
    doc_count = int(metrics["doc_count"])
    strong_doc_count = int(metrics["strong_doc_count"])
    max_similarity = float(metrics["max_similarity"])

    if doc_count < 2:
        return False
    if strong_doc_count <= 0 and max_similarity < settings.CONFIDENCE_THRESHOLD:
        return False

    entity_hints = _extract_entity_hints(idea)
    if not entity_hints:
        return True

    context_head = " ".join(
        [
            (doc.source_rel or doc.source or "")
            + " "
            + (doc.topic or "")
            + " "
            + (doc.subtopic or "")
            + " "
            + (doc.text[:300] if doc.text else "")
            for doc in context_bundle.docs
        ]
    ).lower()
    return any(entity in context_head for entity in entity_hints)


def _context_coverage_metrics(context_bundle: ContextBundle) -> dict[str, float | int]:
    similarities = [max(0.0, float(doc.similarity_score)) for doc in context_bundle.docs]
    max_similarity = max(similarities, default=0.0)
    avg_similarity = mean(similarities) if similarities else 0.0
    strong_doc_count = len([score for score in similarities if score >= settings.CONFIDENCE_THRESHOLD])

    return {
        "doc_count": len(similarities),
        "strong_doc_count": strong_doc_count,
        "max_similarity": round(max_similarity, 4),
        "avg_similarity": round(avg_similarity, 4),
        "local_tokens": int(context_bundle.local_tokens),
        "dynamic_tokens": int(context_bundle.dynamic_tokens),
        "total_tokens": int(context_bundle.total_tokens),
        "confidence_threshold": float(settings.CONFIDENCE_THRESHOLD),
    }


def should_abstain_for_coverage(
    context_bundle: ContextBundle,
    *,
    low_confidence: bool,
    fallback_requested: bool,
) -> tuple[bool, str, dict[str, float | int]]:
    metrics = _context_coverage_metrics(context_bundle)

    doc_count = int(metrics["doc_count"])
    strong_doc_count = int(metrics["strong_doc_count"])
    max_similarity = float(metrics["max_similarity"])
    dynamic_tokens = int(metrics["dynamic_tokens"])

    if doc_count == 0:
        return True, "No retrievable context was found for this question.", metrics

    if low_confidence and strong_doc_count == 0:
        if fallback_requested and dynamic_tokens == 0:
            return (
                True,
                (
                    "Local retrieval confidence is below threshold, and fallback retrieval did not produce "
                    "usable context."
                ),
                metrics,
            )
        return True, "Retrieved context confidence is below threshold for a reliable answer.", metrics

    if max_similarity < (settings.CONFIDENCE_THRESHOLD * 0.75) and doc_count < settings.CRAG_MIN_RELEVANT_DOCS:
        return (
            True,
            "Too few relevant context chunks are available to support a grounded answer.",
            metrics,
        )

    return False, "", metrics


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def _truncate_text(value: str | None, max_len: int) -> str:
    clean = _clean_text(value)
    if len(clean) <= max_len:
        return clean
    return f"{clean[: max_len - 1].rstrip()}..."


def _clean_text_list(items: list[str] | None, max_items: int = 6, max_len: int = 220) -> list[str]:
    if not items:
        return []

    cleaned: list[str] = []
    for item in items[:max_items]:
        text = _truncate_text(str(item), max_len)
        if text:
            cleaned.append(text)
    return cleaned


def _short_title_from_idea(idea: str) -> str:
    cleaned = _clean_text(idea)
    if not cleaned:
        return "Brand Strategy Decision"

    lowered = cleaned.lower()
    for prefix in ("build ", "create ", "develop ", "launch ", "design ", "start "):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    candidate = _truncate_text(cleaned, 80).strip(" .,:;-")
    if not candidate:
        return "Brand Strategy Decision"
    return candidate[0].upper() + candidate[1:]


def _infer_risk_level(insight: InsightOutput) -> str | None:
    risks = _clean_text_list(insight.risks)
    risk_text = " ".join(risks).lower()
    severe_keywords = (
        "regulat",
        "legal",
        "compliance",
        "security",
        "privacy",
        "safety",
        "liability",
        "high risk",
        "failure",
    )

    has_high_priority = any(action.priority == "high" for action in insight.actions)
    has_severe_signal = any(keyword in risk_text for keyword in severe_keywords)

    if not risks and not has_high_priority:
        return None

    if has_high_priority or has_severe_signal:
        return "High"
    if insight.confidence_score >= 0.72 and len(risks) <= 3:
        return "Low"
    return "Medium"


def _derive_tags(idea: str, insight: InsightOutput) -> list[str]:
    opportunities = _clean_text_list(insight.opportunities, max_items=6, max_len=120)
    suggested = _clean_text_list(insight.suggested_positioning, max_items=6, max_len=120)
    trade_offs = _clean_text_list(insight.trade_offs, max_items=6, max_len=120)

    source = " ".join(
        [
            idea,
            _clean_text(insight.brand_diagnosis),
            _clean_text(insight.market_insight),
            _clean_text(insight.final_positioning),
            _clean_text(insight.target_audience),
            _clean_text(insight.chosen_strategy),
            " ".join(opportunities),
            " ".join(suggested),
            " ".join(trade_offs),
        ]
    ).lower()

    keyword_tags = [
        ("ai", "ai"),
        ("automation", "automation"),
        ("saas", "saas"),
        ("market", "market"),
        ("fin", "fintech"),
        ("health", "healthtech"),
        ("edu", "edtech"),
        ("supply", "supply-chain"),
        ("b2b", "b2b"),
        ("consumer", "b2c"),
        ("mobile", "mobile"),
        ("platform", "platform"),
        ("position", "positioning"),
        ("message", "messaging"),
        ("trust", "trust"),
        ("brand", "brand"),
    ]

    tags: list[str] = []
    for keyword, tag in keyword_tags:
        if keyword in source and tag not in tags:
            tags.append(tag)

    if "brand-strategy" not in tags:
        tags.append("brand-strategy")
    if "execution" not in tags:
        tags.append("execution")

    return tags[:8]


def _default_abstention_message() -> str:
    return (
        "Sorry, I do not have enough grounded evidence in the available documents to answer that question reliably. "
        "Please ask a question tied to the ingested brand materials or add more relevant evidence and try again."
    )


def build_insufficient_context_insight(idea: str, context_bundle: ContextBundle, reason: str) -> InsightOutput:
    metrics = _context_coverage_metrics(context_bundle)
    confidence = min(float(metrics["avg_similarity"]), 0.2)

    insight = InsightOutput(
        abstention_message=_default_abstention_message(),
        confidence_score=float(max(0.0, min(0.2, confidence))),
    )
    return _ensure_notion_format_outputs(idea, insight)


def _build_notion_page_content(idea: str, insight: InsightOutput) -> str:
    del idea  # Content is derived from the structured insight payload.

    suggested = _clean_text_list(insight.suggested_positioning, max_items=6, max_len=320)
    risks = _clean_text_list(insight.risks)
    opportunities = _clean_text_list(insight.opportunities)
    rejected = _clean_text_list(insight.rejected_directions)
    trade_offs = _clean_text_list(insight.trade_offs)

    tasks: list[str] = []
    for idx, action in enumerate(insight.actions[:5], start=1):
        task_text = _clean_text(action.title)
        if action.description:
            task_text = f"{task_text}: {_clean_text(action.description)}"
        decision_label = str(action.decision_type).replace("_", " ").title()
        meta = [f"Decision Type: {decision_label}", f"Impact: {action.impact.capitalize()}"]
        task_text = f"{task_text} ({'; '.join(meta)})"
        tasks.append(f"* Task {idx}: {task_text}")

    lines: list[str] = []

    def add_text_section(title: str, value: str | None, *, max_len: int = 320) -> None:
        section_text = _truncate_text(value, max_len)
        if not section_text:
            return
        lines.extend([title, section_text, ""])

    def add_list_section(title: str, values: list[str]) -> None:
        if not values:
            return
        lines.append(title)
        lines.extend(f"- {item}" for item in values)
        lines.append("")

    add_text_section("🎯 Target Audience", insight.target_audience)
    add_text_section("💡 Positioning", insight.final_positioning or (suggested[0] if suggested else None))
    add_text_section("⚡ Differentiation", suggested[1] if len(suggested) > 1 else None)
    add_text_section("🧠 Brand Narrative", insight.chosen_strategy)
    add_text_section("📊 Market Insight", insight.market_insight or insight.brand_diagnosis, max_len=450)
    add_list_section("⚠️ Risks", risks)
    add_list_section("📈 Opportunities", opportunities)
    add_text_section("✅ Final Positioning", insight.final_positioning)
    add_list_section("❌ Rejected Directions", rejected)
    add_list_section("⚖️ Trade-offs", trade_offs)
    add_list_section("🛠 Action Items", tasks)

    if not lines:
        return ""

    # Remove trailing blank line while preserving section separation.
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines).strip()


def _build_database_metadata(idea: str, insight: InsightOutput) -> NotionDatabaseMetadata:
    suggested = _clean_text_list(insight.suggested_positioning, max_items=6, max_len=240)
    primary_positioning = (
        insight.final_positioning
        or (suggested[0] if suggested else "")
        or insight.market_insight
        or insight.brand_diagnosis
        or idea
    )

    tags = _derive_tags(idea, insight)
    confidence_score = int(round(max(0.0, min(1.0, insight.confidence_score)) * 100))

    return NotionDatabaseMetadata(
        name=_short_title_from_idea(idea),
        brand_positioning=_truncate_text(primary_positioning, 240) or None,
        brand_risk_level=_infer_risk_level(insight),
        confidence_score=confidence_score,
        tags=tags or None,
    )


def _ensure_notion_format_outputs(idea: str, insight: InsightOutput) -> InsightOutput:
    abstention_message = _truncate_text(insight.abstention_message, 320) or None
    if abstention_message:
        insight.abstention_message = abstention_message
        insight.brand_diagnosis = None
        insight.market_insight = None
        insight.final_positioning = None
        insight.target_audience = None
        insight.chosen_strategy = None
        insight.suggested_positioning = None
        insight.risks = None
        insight.opportunities = None
        insight.rejected_directions = None
        insight.trade_offs = None
        insight.actions = []
        insight.notion_page_content = None
        insight.database_metadata = None
        return insight

    insight.brand_diagnosis = _truncate_text(insight.brand_diagnosis, 500) or None
    insight.market_insight = _truncate_text(insight.market_insight, 450) or None
    insight.final_positioning = _truncate_text(insight.final_positioning, 320) or None
    insight.target_audience = _truncate_text(insight.target_audience, 320) or None
    insight.chosen_strategy = _truncate_text(insight.chosen_strategy, 320) or None

    insight.suggested_positioning = _clean_text_list(insight.suggested_positioning, max_items=6, max_len=320) or None
    insight.risks = _clean_text_list(insight.risks) or None
    insight.opportunities = _clean_text_list(insight.opportunities) or None
    insight.rejected_directions = _clean_text_list(insight.rejected_directions) or None
    insight.trade_offs = _clean_text_list(insight.trade_offs) or None

    if not _clean_text(insight.notion_page_content):
        insight.notion_page_content = _build_notion_page_content(idea, insight) or None
    else:
        insight.notion_page_content = insight.notion_page_content.strip() or None

    if insight.database_metadata is None:
        insight.database_metadata = _build_database_metadata(idea, insight)
    else:
        metadata = insight.database_metadata
        metadata.name = _truncate_text(metadata.name, 120) or None
        metadata.brand_positioning = _truncate_text(
            metadata.brand_positioning,
            300,
        ) or None
        metadata.confidence_score = (
            int(max(0, min(100, metadata.confidence_score)))
            if metadata.confidence_score is not None
            else int(round(max(0.0, min(1.0, insight.confidence_score)) * 100))
        )
        tags = [
            _truncate_text(tag, 40)
            for tag in (metadata.tags or [])
            if _clean_text(tag)
        ][:8]
        metadata.tags = tags or (_derive_tags(idea, insight) or None)

        if metadata.brand_risk_level is None:
            metadata.brand_risk_level = _infer_risk_level(insight)

        if not metadata.name:
            metadata.name = _short_title_from_idea(idea)

        if not metadata.brand_positioning:
            metadata.brand_positioning = _truncate_text(
                insight.final_positioning or insight.market_insight or insight.brand_diagnosis or idea,
                240,
            ) or None

    return insight


def _request_options(timeout_override_seconds: int | None = None) -> dict | None:
    timeout = (
        int(timeout_override_seconds)
        if timeout_override_seconds is not None
        else int(settings.MODEL_REQUEST_TIMEOUT_SECONDS)
    )
    if timeout > 0:
        return {"timeout": timeout}
    return None


def _generate_with_options(
    model: genai.GenerativeModel,
    prompt: str,
    generation_config: dict,
    *,
    timeout_override_seconds: int | None = None,
) -> object:
    options = _request_options(timeout_override_seconds=timeout_override_seconds)
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


def _call_gemma(prompt: str, *, timeout_override_seconds: int | None = None) -> tuple[object, str]:
    generation_config = {
        "temperature": 0,
        "response_mime_type": "application/json",
    }
    last_error: Exception | None = None

    for model_name in _candidate_model_names():
        model = _get_model(model_name)
        try:
            response = _generate_with_options(
                model,
                prompt,
                generation_config,
                timeout_override_seconds=timeout_override_seconds,
            )
            return response, model_name
        except Exception as exc:
            try:
                if _json_mode_not_supported(exc):
                    response = _generate_with_options(
                        model,
                        prompt,
                        {"temperature": 0},
                        timeout_override_seconds=timeout_override_seconds,
                    )
                    return response, model_name
            except Exception as inner_exc:
                exc = inner_exc

            last_error = exc
            logger.warning("llm_model_attempt_failed model=%s reason=%s", model_name, exc)
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("No LLM model names configured")


def _is_retryable_model_error(exc: Exception) -> bool:
    retryable = (
        google_exceptions.DeadlineExceeded,
        google_exceptions.ServiceUnavailable,
        google_exceptions.ResourceExhausted,
        google_exceptions.InternalServerError,
        google_exceptions.Aborted,
        google_exceptions.Unknown,
    )
    if isinstance(exc, retryable):
        return True

    message = str(exc).lower()
    return (
        "deadline" in message
        or "timed out" in message
        or "timeout" in message
        or "temporarily unavailable" in message
        or "resource exhausted" in message
    )


async def generate_insight(
    idea: str,
    context_bundle: ContextBundle,
    *,
    timeout_override_seconds: int | None = None,
) -> tuple[InsightOutput, float, int]:
    schema_json = json.dumps(InsightOutput.model_json_schema(), indent=2)
    system_prompt = SYSTEM_PROMPT.format(schema=schema_json)
    context_text = _build_context_text(context_bundle)
    query_intent = _classify_query_intent(idea)
    entity_hints = _extract_entity_hints(idea)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        idea=idea,
        query_intent=query_intent,
        entity_hints=(", ".join(entity_hints) if entity_hints else "none"),
        doc_count=len(context_bundle.docs),
        context=context_text,
        fallback_note=FALLBACK_NOTE if context_bundle.used_fallback else NO_FALLBACK_NOTE,
    )

    prompt = f"{system_prompt}\n\n{user_prompt}"
    corrections: list[str] = []

    start = time.perf_counter()
    last_validation_error: Exception | None = None
    last_provider_error: Exception | None = None
    over_abstention_retry_used = False

    for attempt in range(settings.LLM_MAX_RETRIES):
        composed_prompt = prompt
        if corrections:
            composed_prompt = f"{composed_prompt}\n\n" + "\n\n".join(corrections)

        try:
            response, model_used = _call_gemma(
                composed_prompt,
                timeout_override_seconds=timeout_override_seconds,
            )
            response_text = _extract_text(response)
            insight = InsightOutput.model_validate(_parse_json_payload(response_text))

            if (not over_abstention_retry_used) and _should_retry_after_abstention(idea, context_bundle, insight):
                corrections.append(
                    "Your previous response abstained despite retrievable context. "
                    "Regenerate a PARTIAL grounded answer using related evidence from the retrieved sources. "
                    "Keep abstention_message null unless there is truly zero relevant evidence."
                )
                over_abstention_retry_used = True
                continue

            insight.confidence_score = _calibrate_confidence(context_bundle, insight)
            insight = _ensure_notion_format_outputs(idea, insight)

            latency_ms = (time.perf_counter() - start) * 1000

            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", _token_len(composed_prompt))
            output_tokens = getattr(usage, "candidates_token_count", _token_len(response_text))

            logger.info(
                "llm_call model=%s latency_ms=%.2f input_tokens=%s output_tokens=%s retry_count=%d",
                model_used,
                latency_ms,
                input_tokens,
                output_tokens,
                attempt,
            )

            return insight, float(latency_ms), attempt
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_validation_error = exc
            corrections.append(
                "Your previous response failed JSON validation.\n"
                f"Error: {exc}\n"
                "Respond with ONLY valid JSON matching this schema:\n"
                f"{schema_json}"
            )
        except Exception as exc:
            if _is_retryable_model_error(exc):
                last_provider_error = exc
                logger.warning(
                    "llm_provider_retry attempt=%d/%d reason=%s",
                    attempt + 1,
                    settings.LLM_MAX_RETRIES,
                    exc,
                )
                continue
            raise HTTPException(
                status_code=502,
                detail="LLM provider error while generating insight. Please try again.",
            )

    if last_provider_error is not None:
        raise HTTPException(
            status_code=504,
            detail=(
                "LLM request timed out while generating insight. "
                "Please retry, lower Top K, or shorten the idea text."
            ),
        )

    raise HTTPException(
        status_code=422,
        detail={
            "error": "LLM failed to produce valid structured output after retries",
            "last_error": str(last_validation_error),
        },
    )


async def check_grounding(context_bundle: ContextBundle, insight: InsightOutput) -> str:
    """
    Self-RAG style grounding check.
    Returns one of: grounded, partial, not_grounded, unknown.
    """
    if not context_bundle.docs:
        return "unknown"

    context_text = _build_context_text(context_bundle)
    answer_json = insight.model_dump_json()

    prompt = (
        "You are a grounding grader for RAG outputs.\n"
        "Decide whether the answer is supported by the provided context.\n"
        "Return ONLY JSON with schema: {\"verdict\": \"grounded|partial|not_grounded\"}.\n\n"
        f"Context:\n{context_text[:12000]}\n\n"
        f"Answer JSON:\n{answer_json}"
    )

    try:
        response, _ = _call_gemma(prompt)
        payload = _parse_json_payload(_extract_text(response))
        verdict = str(payload.get("verdict", "unknown")).strip().lower()
        if verdict in {"grounded", "partial", "not_grounded"}:
            return verdict
    except Exception as exc:
        logger.info("Grounding check fallback: %s", exc)

    return "unknown"


async def regenerate_grounded_insight(
    idea: str,
    context_bundle: ContextBundle,
    prior_insight: InsightOutput,
) -> InsightOutput | None:
    """
    Attempts one corrective rewrite when the grounding grader flags hallucination risk.
    Returns None if a valid corrected JSON output cannot be produced.
    """
    schema_json = json.dumps(InsightOutput.model_json_schema(), indent=2)
    context_text = _build_context_text(context_bundle)

    prompt = (
        "You are repairing a RAG answer that was flagged as not grounded.\n"
        "Rewrite the answer using ONLY information supported by the provided context excerpts.\n"
        "If context is missing details, set `abstention_message` and avoid fabricated specifics.\n"
        "Return ONLY valid JSON for this schema:\n"
        f"{schema_json}\n\n"
        f"Brand Decision Question:\n{idea}\n\n"
        f"Context Excerpts:\n{context_text}\n\n"
        f"Previous Answer JSON:\n{prior_insight.model_dump_json()}"
    )

    corrections: list[str] = []
    for _ in range(2):
        composed_prompt = f"{prompt}\n\n" + "\n\n".join(corrections) if corrections else prompt
        try:
            response, _ = _call_gemma(composed_prompt)
            response_text = _extract_text(response)
            repaired = InsightOutput.model_validate(_parse_json_payload(response_text))

            repaired.confidence_score = _calibrate_confidence(context_bundle, repaired)
            repaired = _ensure_notion_format_outputs(idea, repaired)
            return repaired
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            corrections.append(
                "Your previous repair attempt failed JSON validation.\n"
                f"Error: {exc}\n"
                "Respond with ONLY valid JSON matching the schema."
            )

    return None


def build_conservative_insight(idea: str, context_bundle: ContextBundle) -> InsightOutput:
    """
    Produces an explicit abstention response when a grounded answer cannot be produced.
    """
    return build_insufficient_context_insight(
        idea,
        context_bundle,
        "The generated answer could not be grounded to retrieved evidence.",
    )


async def enforce_faithfulness(
    idea: str,
    context_bundle: ContextBundle,
    insight: InsightOutput,
) -> tuple[InsightOutput, str, bool]:
    """
    Dedicated hallucination-control pass:
    1) grade grounding
    2) repair once when ungrounded
    3) fallback to conservative output if still ungrounded
    """
    verdict = await check_grounding(context_bundle, insight)
    if verdict != "not_grounded":
        insight.confidence_score = _calibrate_confidence(context_bundle, insight, grounding_verdict=verdict)
        return insight, verdict, False

    repaired = await regenerate_grounded_insight(idea, context_bundle, insight)
    if repaired is not None:
        repaired_verdict = await check_grounding(context_bundle, repaired)
        if repaired_verdict in {"grounded", "partial"}:
            repaired.confidence_score = _calibrate_confidence(
                context_bundle,
                repaired,
                grounding_verdict=repaired_verdict,
            )
            return repaired, repaired_verdict, True

    fallback = build_conservative_insight(idea, context_bundle)
    return fallback, "not_grounded", True
