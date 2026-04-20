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
from reasoning.schema import InsightOutput
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
                if "response_mime_type" in str(exc):
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
        f"Startup Idea:\n{idea}\n\n"
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
    Produces a low-risk fallback response when the model remains ungrounded after repair.
    """
    confidence = mean([doc.similarity_score for doc in context_bundle.docs]) if context_bundle.docs else 0.0
    conservative_confidence = float(max(0.0, min(1.0, confidence * 0.6)))

    return InsightOutput(
        idea_summary=f"Initial review for: {idea[:220]}",
        risks=[
            "Retrieved evidence is insufficient to fully validate detailed risk claims.",
            "Some high-impact assumptions may still require explicit supporting data.",
        ],
        opportunities=[
            "Context indicates demand for deeper user and segment-specific validation.",
            "Operational improvements can be prioritized once stronger evidence is collected.",
        ],
        recommendations=[
            "Prioritize evidence collection from additional trusted product or market sources.",
            "Re-run analysis after adding more domain-specific and recent context documents.",
        ],
        actions=[
            {
                "type": "task",
                "title": "Collect supporting evidence",
                "description": "Add more relevant documents and rerun grounded analysis.",
                "priority": "high",
            }
        ],
        confidence_score=conservative_confidence,
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