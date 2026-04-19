from __future__ import annotations

import asyncio
import logging
import random
from uuid import UUID

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from evaluation.models import EvaluationLog


logger = logging.getLogger(__name__)


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


def build_ragas_components() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """
    Configure RAGAS with Gemma 4 and Google text-embedding-004.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMMA_MODEL_NAME,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0,
        request_timeout=settings.RAGAS_TIMEOUT_SECONDS,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.GOOGLE_API_KEY,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


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
        ragas_llm, ragas_embeddings = build_ragas_components()

        dataset = Dataset.from_dict(
            {
                "question": [query],
                "contexts": [retrieved_docs],
                "answer": [generated_output],
                "ground_truth": [generated_output],
            }
        )

        metrics = [
            _configure_metric(context_precision, ragas_llm, ragas_embeddings),
            _configure_metric(context_recall, ragas_llm, ragas_embeddings),
            _configure_metric(faithfulness, ragas_llm, ragas_embeddings),
            _configure_metric(answer_relevancy, ragas_llm, ragas_embeddings),
        ]

        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: evaluate(dataset, metrics=metrics)),
            timeout=settings.RAGAS_TIMEOUT_SECONDS,
        )

        scores = result.to_pandas().iloc[0].to_dict()

        eval_row_result = await db.execute(select(EvaluationLog).where(EvaluationLog.session_id == session_id))
        eval_row = eval_row_result.scalar_one_or_none()
        if eval_row is None:
            return

        eval_row.context_precision = scores.get("context_precision")
        eval_row.context_recall = scores.get("context_recall")
        eval_row.faithfulness = scores.get("faithfulness")
        eval_row.answer_relevance = scores.get("answer_relevancy")
        eval_row.ragas_eval_status = "completed"
        await db.commit()

    except asyncio.TimeoutError:
        logger.warning(
            "RAGAS evaluation timed out after %ss for session %s",
            settings.RAGAS_TIMEOUT_SECONDS,
            session_id,
        )
        await _set_status(db, session_id, "failed")

    except Exception as exc:
        logger.error(
            "RAGAS evaluation failed for session %s: %s: %s",
            session_id,
            type(exc).__name__,
            exc,
        )
        await _set_status(db, session_id, "failed")