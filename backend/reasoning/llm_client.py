from __future__ import annotations

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


def _build_context_text(bundle: ContextBundle) -> str:
    lines: list[str] = []
    for idx, doc in enumerate(bundle.docs, start=1):
        lines.append(
            f"[{idx}] source={doc.source} doc_type={doc.doc_type} similarity={doc.similarity_score:.4f}\n{doc.text}"
        )
    return "\n\n".join(lines)


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


def _clean_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _truncate_text(value: str, max_len: int) -> str:
    clean = _clean_text(value)
    if len(clean) <= max_len:
        return clean
    return f"{clean[: max_len - 1].rstrip()}..."


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


def _infer_risk_level(insight: InsightOutput) -> str:
    risk_text = " ".join(insight.risks).lower()
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

    if has_high_priority or has_severe_signal:
        return "High"
    if insight.confidence_score >= 0.72 and len(insight.risks) <= 3:
        return "Low"
    return "Medium"


def _derive_tags(idea: str, insight: InsightOutput) -> list[str]:
    source = " ".join(
        [
            idea,
            insight.brand_diagnosis,
            insight.market_insight,
            insight.final_positioning,
            insight.target_audience,
            insight.chosen_strategy,
            " ".join(insight.opportunities),
            " ".join(insight.suggested_positioning),
            " ".join(insight.trade_offs),
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


def _ensure_decision_defaults(insight: InsightOutput) -> None:
    if not _clean_text(insight.market_insight):
        insight.market_insight = _truncate_text(insight.brand_diagnosis, 450)

    if not insight.suggested_positioning:
        insight.suggested_positioning = [
            "Clarify a narrow promise for a specific audience before scaling messaging.",
            "Lead with a concrete proof point instead of broad category language.",
        ]

    if not _clean_text(insight.target_audience):
        insight.target_audience = "TODO: Define the first audience segment to prioritize."

    if not _clean_text(insight.final_positioning):
        insight.final_positioning = "TODO: Choose and refine one positioning statement."

    if not _clean_text(insight.chosen_strategy):
        insight.chosen_strategy = "TODO: Select one strategy path and commit to it for the next sprint."

    if not insight.rejected_directions:
        insight.rejected_directions = [
            "Generic positioning for all users without a clear differentiator.",
        ]

    if not insight.trade_offs:
        insight.trade_offs = [
            "Narrow positioning improves clarity and conversion, but reduces initial reach.",
        ]


def build_insufficient_context_insight(idea: str, context_bundle: ContextBundle, reason: str) -> InsightOutput:
    metrics = _context_coverage_metrics(context_bundle)
    confidence = min(float(metrics["avg_similarity"]), 0.2)

    insight = InsightOutput(
        brand_diagnosis=(
            f"Unable to answer reliably from the available corpus for this question: {idea[:220]}"
        ),
        market_insight=(
            "I cannot provide a grounded recommendation because supporting evidence is insufficient. "
            f"Reason: {reason}"
        ),
        suggested_positioning=[
            "No strategic recommendation generated due to insufficient grounded evidence.",
            "Please add domain-specific documents for this question and retry.",
        ],
        risks=[
            "Any detailed recommendation now would likely be speculative or hallucinated.",
            "Current context does not provide enough support for a confident answer.",
        ],
        opportunities=[
            "Ingest additional documents directly related to this decision question.",
            "Retry after corpus expansion or with a narrower, evidence-linked query.",
        ],
        final_positioning="Not provided due to insufficient evidence.",
        target_audience="Not provided due to insufficient evidence.",
        chosen_strategy="Not provided due to insufficient evidence.",
        rejected_directions=["Proceeding with unsupported strategic decisions from weak context."],
        trade_offs=["Abstaining avoids confident but ungrounded recommendations."],
        actions=[
            {
                "type": "task",
                "title": "Expand corpus coverage for this question",
                "description": "Add and ingest domain-specific evidence, then rerun the analysis.",
                "priority": "high",
                "decision_type": "other",
                "impact": "high",
            }
        ],
        confidence_score=float(max(0.0, min(0.2, confidence))),
    )
    return _ensure_notion_format_outputs(idea, insight)


def _build_notion_page_content(idea: str, insight: InsightOutput) -> str:
    _ensure_decision_defaults(insight)

    market_insight = _truncate_text(
        insight.market_insight or insight.brand_diagnosis,
        450,
    ) or "No market insight available."

    positioning = _truncate_text(
        insight.suggested_positioning[0] if insight.suggested_positioning else insight.final_positioning,
        320,
    ) or "TODO: Define how this brand should be positioned."

    differentiation = _truncate_text(
        insight.suggested_positioning[1] if len(insight.suggested_positioning) > 1 else insight.chosen_strategy,
        320,
    ) or "TODO: Clarify the strongest reason users should choose this brand."

    brand_narrative = _truncate_text(insight.chosen_strategy, 320)

    risks = insight.risks or ["No major brand risks identified from retrieved context."]
    opportunities = insight.opportunities or ["No clear opportunity areas identified from retrieved context."]

    tasks: list[str] = []
    for idx, action in enumerate(insight.actions[:5], start=1):
        task_text = _clean_text(action.title)
        if action.description:
            task_text = f"{task_text}: {_clean_text(action.description)}"
        decision_label = str(action.decision_type).replace("_", " ").title()
        meta = [f"Decision Type: {decision_label}", f"Impact: {action.impact.capitalize()}"]
        task_text = f"{task_text} ({'; '.join(meta)})"
        tasks.append(f"* Task {idx}: {task_text}")
    if not tasks:
        tasks.append("* Task 1: Define immediate next-step actions from selected decisions")

    lines: list[str] = [
        "🎯 Target Audience",
        _truncate_text(insight.target_audience, 320),
        "",
        "💡 Positioning",
        positioning,
        "",
        "⚡ Differentiation",
        differentiation,
        "",
        "🧠 Brand Narrative",
        brand_narrative,
        "",
        "📊 Market Insight",
        market_insight,
        "",
        "⚠️ Risks",
    ]

    lines.extend(f"- {_truncate_text(item, 220)}" for item in risks[:6])
    lines.extend(["", "📈 Opportunities"])
    lines.extend(f"- {_truncate_text(item, 220)}" for item in opportunities[:6])
    lines.extend(["", "✅ Final Positioning", _truncate_text(insight.final_positioning, 320)])
    lines.extend(["", "❌ Rejected Directions"])
    lines.extend(f"- {_truncate_text(item, 220)}" for item in insight.rejected_directions[:6])
    lines.extend(["", "⚖️ Trade-offs"])
    lines.extend(f"- {_truncate_text(item, 220)}" for item in insight.trade_offs[:6])
    lines.extend(["", "🛠 Action Items"])
    lines.extend(tasks)

    return "\n".join(lines).strip()


def _build_database_metadata(idea: str, insight: InsightOutput) -> NotionDatabaseMetadata:
    _ensure_decision_defaults(insight)
    primary_positioning = (
        insight.final_positioning
        or (insight.suggested_positioning[0] if insight.suggested_positioning else "")
        or insight.brand_diagnosis
        or idea
    )

    return NotionDatabaseMetadata(
        name=_short_title_from_idea(idea),
        brand_positioning=_truncate_text(primary_positioning, 240),
        brand_risk_level=_infer_risk_level(insight),
        confidence_score=int(round(max(0.0, min(1.0, insight.confidence_score)) * 100)),
        tags=_derive_tags(idea, insight),
    )


def _ensure_notion_format_outputs(idea: str, insight: InsightOutput) -> InsightOutput:
    insight.brand_diagnosis = _truncate_text(insight.brand_diagnosis, 500) or "No brand diagnosis available."
    _ensure_decision_defaults(insight)

    if not _clean_text(insight.notion_page_content):
        insight.notion_page_content = _build_notion_page_content(idea, insight)
    else:
        insight.notion_page_content = insight.notion_page_content.strip()

    if insight.database_metadata is None:
        insight.database_metadata = _build_database_metadata(idea, insight)
    else:
        metadata = insight.database_metadata
        metadata.name = _truncate_text(metadata.name, 120) or _short_title_from_idea(idea)
        metadata.brand_positioning = _truncate_text(
            metadata.brand_positioning,
            300,
        ) or _truncate_text(insight.final_positioning or insight.brand_diagnosis or idea, 240)
        metadata.confidence_score = int(max(0, min(100, metadata.confidence_score)))
        metadata.tags = [
            _truncate_text(tag, 40)
            for tag in metadata.tags
            if _clean_text(tag)
        ][:8]
        if not metadata.tags:
            metadata.tags = _derive_tags(idea, insight)

    return insight


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


def _call_gemma(prompt: str) -> tuple[object, str]:
    generation_config = {
        "temperature": 0,
        "response_mime_type": "application/json",
    }
    last_error: Exception | None = None

    for model_name in _candidate_model_names():
        model = _get_model(model_name)
        try:
            response = _generate_with_options(model, prompt, generation_config)
            return response, model_name
        except Exception as exc:
            try:
                if _json_mode_not_supported(exc):
                    response = _generate_with_options(model, prompt, {"temperature": 0})
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


async def generate_insight(idea: str, context_bundle: ContextBundle) -> tuple[InsightOutput, float, int]:
    schema_json = json.dumps(InsightOutput.model_json_schema(), indent=2)
    system_prompt = SYSTEM_PROMPT.format(schema=schema_json)
    context_text = _build_context_text(context_bundle)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        idea=idea,
        doc_count=len(context_bundle.docs),
        context=context_text,
        fallback_note=FALLBACK_NOTE if context_bundle.used_fallback else NO_FALLBACK_NOTE,
    )

    prompt = f"{system_prompt}\n\n{user_prompt}"
    corrections: list[str] = []

    start = time.perf_counter()
    last_validation_error: Exception | None = None
    last_provider_error: Exception | None = None

    for attempt in range(settings.LLM_MAX_RETRIES):
        composed_prompt = prompt
        if corrections:
            composed_prompt = f"{composed_prompt}\n\n" + "\n\n".join(corrections)

        try:
            response, model_used = _call_gemma(composed_prompt)
            response_text = _extract_text(response)
            insight = InsightOutput.model_validate(_parse_json_payload(response_text))

            confidence = mean([doc.similarity_score for doc in context_bundle.docs]) if context_bundle.docs else 0.0
            insight.confidence_score = float(max(0.0, min(1.0, confidence)))
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
        "If context is missing details, use cautious language and avoid fabricated specifics.\n"
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

            confidence = mean([doc.similarity_score for doc in context_bundle.docs]) if context_bundle.docs else 0.0
            repaired.confidence_score = float(max(0.0, min(1.0, confidence)))
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
        return insight, verdict, False

    repaired = await regenerate_grounded_insight(idea, context_bundle, insight)
    if repaired is not None:
        repaired_verdict = await check_grounding(context_bundle, repaired)
        if repaired_verdict in {"grounded", "partial"}:
            return repaired, repaired_verdict, True

    fallback = build_conservative_insight(idea, context_bundle)
    return fallback, "not_grounded", True