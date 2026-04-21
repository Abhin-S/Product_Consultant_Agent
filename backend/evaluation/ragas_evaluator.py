from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import typing as t
from uuid import UUID

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper, is_multiple_completion_supported
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from evaluation.models import EvaluationLog


logger = logging.getLogger(__name__)


def _candidate_model_names() -> list[str]:
    names = [settings.GEMMA_MODEL_NAME]
    fallback = settings.LLM_FALLBACK_MODEL_NAME.strip()
    if fallback and fallback not in names:
        names.append(fallback)
    return names


class TemperatureSafeLangchainLLMWrapper(LangchainLLMWrapper):
    """
    RAGAS may pass `temperature` at call-time, but current Google GenAI async client
    rejects it as a top-level kwarg. Force temperature=None here so the wrapped
    LangChain model does not forward incompatible kwargs.
    """

    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks=None,
    ):
        if temperature is not None:
            logger.debug("Ignoring runtime RAGAS temperature=%s for Google GenAI compatibility", temperature)

        if is_multiple_completion_supported(self.langchain_llm):
            return self.langchain_llm.generate_prompt(
                prompts=[prompt],
                n=n,
                stop=stop,
                callbacks=callbacks,
            )

        result = self.langchain_llm.generate_prompt(
            prompts=[prompt] * n,
            stop=stop,
            callbacks=callbacks,
        )
        generations = [[g[0] for g in result.generations]]
        result.generations = generations
        return result

    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks=None,
    ):
        if temperature is not None:
            logger.debug("Ignoring runtime RAGAS temperature=%s for Google GenAI compatibility", temperature)

        if is_multiple_completion_supported(self.langchain_llm):
            return await self.langchain_llm.agenerate_prompt(
                prompts=[prompt],
                n=n,
                stop=stop,
                callbacks=callbacks,
            )

        result = await self.langchain_llm.agenerate_prompt(
            prompts=[prompt] * n,
            stop=stop,
            callbacks=callbacks,
        )
        generations = [[g[0] for g in result.generations]]
        result.generations = generations
        return result


def _ragas_timeout_seconds() -> int:
    timeout = int(settings.RAGAS_TIMEOUT_SECONDS)
    return timeout if timeout > 0 else 0


def _to_metric_float(value: t.Any) -> float | None:
    if value is None:
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(parsed):
        return None

    return parsed


def log_ragas_version() -> None:
    try:
        import ragas

        logger.info("RAGAS version: %s", ragas.__version__)
    except Exception as exc:
        logger.warning("Unable to read RAGAS version: %s", exc)


def should_run_ragas() -> bool:
    """
    Returns True with probability 1/EVAL_SAMPLE_RATE.
    Uses random sampling to avoid deterministic bias.
    """
    rate = max(settings.EVAL_SAMPLE_RATE, 1)
    return random.random() < (1.0 / rate)


def build_ragas_components(model_name: str) -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """
    Configure RAGAS with a candidate LLM and Google text-embedding-004.
    """
    timeout_seconds = _ragas_timeout_seconds()
    llm_kwargs = {
        "model": model_name,
        "google_api_key": settings.GOOGLE_API_KEY,
    }
    max_output_tokens = int(settings.RAGAS_MAX_OUTPUT_TOKENS)
    if max_output_tokens > 0:
        llm_kwargs["max_output_tokens"] = max_output_tokens
    if timeout_seconds > 0:
        llm_kwargs["request_timeout"] = timeout_seconds

    llm = ChatGoogleGenerativeAI(**llm_kwargs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.GOOGLE_API_KEY,
    )
    return TemperatureSafeLangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def _configure_metric(metric_obj, ragas_llm, ragas_embeddings):
    try:
        return metric_obj.copy(update={"llm": ragas_llm, "embeddings": ragas_embeddings})
    except Exception:
        try:
            metric_cls = metric_obj.__class__
            try:
                return metric_cls(llm=ragas_llm, embeddings=ragas_embeddings)
            except TypeError:
                return metric_cls(llm=ragas_llm)
        except Exception:
            return metric_obj


def _evaluate_metrics_fail_fast(dataset: Dataset, ragas_llm, ragas_embeddings) -> dict[str, float | None]:
    """
    Evaluate each metric one-by-one and fail immediately on the first metric exception.
    This avoids long-running partial evaluations when one metric is already unrecoverable.
    """
    metric_specs = [
        ("context_precision", context_precision),
        ("context_recall", context_recall),
        ("faithfulness", faithfulness),
        ("answer_relevancy", answer_relevancy),
    ]

    values: dict[str, float | None] = {
        "context_precision": None,
        "context_recall": None,
        "faithfulness": None,
        "answer_relevancy": None,
    }

    for metric_name, metric_obj in metric_specs:
        configured_metric = _configure_metric(metric_obj, ragas_llm, ragas_embeddings)
        result = evaluate(
            dataset,
            metrics=[configured_metric],
            raise_exceptions=True,
        )
        row = result.to_pandas().iloc[0].to_dict()
        value = _to_metric_float(row.get(metric_name))
        if value is None:
            raise ValueError(f"RAGAS metric '{metric_name}' returned no usable value")
        values[metric_name] = value

    return values


def _truncate_text(value: str, limit: int) -> str:
    text = (value or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _build_eval_answer_text(generated_output: str) -> str:
    answer_limit = int(settings.RAGAS_ANSWER_CHAR_LIMIT)

    try:
        payload = json.loads(generated_output)
    except Exception:
        return _truncate_text(generated_output, answer_limit)

    if not isinstance(payload, dict):
        return _truncate_text(generated_output, answer_limit)

    lines: list[str] = []
    for key in ("brand_diagnosis", "market_insight", "final_positioning", "target_audience", "chosen_strategy"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value.strip()}")

    suggestions = payload.get("suggested_positioning")
    if isinstance(suggestions, list) and suggestions:
        lines.append("suggested_positioning: " + "; ".join(str(item).strip() for item in suggestions[:4]))

    risks = payload.get("risks")
    if isinstance(risks, list) and risks:
        lines.append("risks: " + "; ".join(str(item).strip() for item in risks[:4]))

    opportunities = payload.get("opportunities")
    if isinstance(opportunities, list) and opportunities:
        lines.append("opportunities: " + "; ".join(str(item).strip() for item in opportunities[:4]))

    if not lines:
        return _truncate_text(generated_output, answer_limit)

    return _truncate_text("\n".join(lines), answer_limit)


def _build_eval_contexts(retrieved_docs: list[str]) -> list[str]:
    max_docs = max(int(settings.RAGAS_MAX_CONTEXT_DOCS), 1)
    char_limit = max(int(settings.RAGAS_CONTEXT_DOC_CHAR_LIMIT), 200)
    contexts: list[str] = []
    for doc in retrieved_docs[:max_docs]:
        contexts.append(_truncate_text(str(doc), char_limit))
    return contexts


async def _set_status(db: AsyncSession, session_id: UUID, status_value: str) -> EvaluationLog | None:
    result = await db.execute(select(EvaluationLog).where(EvaluationLog.session_id == session_id))
    row = result.scalar_one_or_none()
    if row is None:
        return None
    row.ragas_eval_status = status_value
    await db.commit()
    return row


async def run_ragas_evaluation(
    session_id: UUID,
    query: str,
    retrieved_docs: list[str],
    generated_output: str,
    db: AsyncSession,
) -> None:
    """
    Runs as a FastAPI BackgroundTask. Never raises.
    Updates the evaluation_logs row for this session_id.

    Evaluation is for developer monitoring only.
    It does NOT affect user-facing responses.
    """
    await _set_status(db, session_id, "pending")

    try:
        contexts = _build_eval_contexts(retrieved_docs)
        answer_text = _build_eval_answer_text(generated_output)

        dataset = Dataset.from_dict(
            {
                "question": [query],
                "contexts": [contexts],
                "answer": [answer_text],
                "ground_truth": [answer_text],
            }
        )

        timeout_seconds = _ragas_timeout_seconds()
        loop = asyncio.get_event_loop()
        metric_values: dict[str, float | None] | None = None
        selected_model: str | None = None
        last_exception: Exception | None = None

        for model_name in _candidate_model_names():
            try:
                ragas_llm, ragas_embeddings = build_ragas_components(model_name)
                evaluate_job = loop.run_in_executor(
                    None,
                    lambda llm=ragas_llm, embeddings=ragas_embeddings: _evaluate_metrics_fail_fast(
                        dataset,
                        llm,
                        embeddings,
                    ),
                )

                if timeout_seconds > 0:
                    metric_values = await asyncio.wait_for(evaluate_job, timeout=timeout_seconds)
                else:
                    logger.info(
                        "RAGAS timeout disabled (RAGAS_TIMEOUT_SECONDS<=0); waiting for evaluation completion"
                    )
                    metric_values = await evaluate_job

                selected_model = model_name
                break

            except asyncio.TimeoutError as exc:
                last_exception = exc
                logger.warning(
                    "RAGAS evaluation timed out after %ss for session %s on model %s",
                    timeout_seconds,
                    session_id,
                    model_name,
                )
                continue

            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "RAGAS evaluation attempt failed for session %s on model %s: %s: %s",
                    session_id,
                    model_name,
                    type(exc).__name__,
                    exc,
                )
                continue

        if metric_values is None:
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("RAGAS evaluation failed for all configured models")

        eval_row_result = await db.execute(select(EvaluationLog).where(EvaluationLog.session_id == session_id))
        eval_row = eval_row_result.scalar_one_or_none()
        if eval_row is None:
            return

        eval_row.context_precision = metric_values["context_precision"]
        eval_row.context_recall = metric_values["context_recall"]
        eval_row.faithfulness = metric_values["faithfulness"]
        eval_row.answer_relevance = metric_values["answer_relevancy"]

        eval_row.ragas_eval_status = "completed"
        await db.commit()

        if selected_model is not None:
            logger.info(
                "RAGAS evaluation completed for session %s using model %s",
                session_id,
                selected_model,
            )

    except Exception as exc:
        logger.error(
            "RAGAS evaluation failed for session %s: %s: %s",
            session_id,
            type(exc).__name__,
            exc,
        )
        await _set_status(db, session_id, "failed")