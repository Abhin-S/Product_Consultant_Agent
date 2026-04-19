from __future__ import annotations

import json
from statistics import mean
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from auth.models import User
from config import settings
from database import AsyncSessionLocal, get_db
from evaluation.lightweight_metrics import (
    compute_avg_similarity,
    compute_context_token_ratio,
    compute_fallback_stats,
    compute_generation_stats,
    compute_similarity_distribution,
)
from evaluation.models import AnalysisSession, EvaluationLog
from evaluation.ragas_evaluator import run_ragas_evaluation, should_run_ragas
from ingestion.embedder import get_embedder_model
from reasoning.llm_client import enforce_faithfulness, generate_insight
from retrieval.fallback.context_builder import build_context_bundle
from retrieval.fallback.dynamic_retriever import retrieve_dynamic_chunks
from retrieval.retriever import retrieve_local


router = APIRouter(tags=["analysis"])


class AnalyzeRequest(BaseModel):
    idea: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=settings.TOP_K_DEFAULT, ge=1, le=10)
    use_fallback: bool = True
    run_evaluation: bool = True


async def _run_ragas_in_background(
    session_id: UUID,
    query: str,
    retrieved_docs: list[str],
    generated_output: str,
) -> None:
    async with AsyncSessionLocal() as bg_db:
        await run_ragas_evaluation(
            session_id=session_id,
            query=query,
            retrieved_docs=retrieved_docs,
            generated_output=generated_output,
            db=bg_db,
        )


@router.post("/analyze")
async def analyze_idea(
    payload: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    local_docs, low_confidence, retrieval_diagnostics = retrieve_local(payload.idea, payload.top_k)

    dynamic_chunks = []
    filter_stats = {
        "fetched": 0,
        "after_language": 0,
        "after_length": 0,
        "after_gibberish": 0,
        "after_relevance": 0,
        "after_dedup": 0,
        "after_recency": 0,
    }

    if low_confidence and payload.use_fallback:
        dynamic_chunks, filter_stats = await retrieve_dynamic_chunks(
            query=payload.idea,
            local_docs=local_docs,
            embedder=get_embedder_model(),
        )

    context_bundle = build_context_bundle(local_docs=local_docs, dynamic_chunks=dynamic_chunks)
    insight, latency_ms, retry_count = await generate_insight(payload.idea, context_bundle)
    faithfulness_corrected = False
    if settings.ENABLE_GROUNDING_CHECK:
        insight, grounding_status, faithfulness_corrected = await enforce_faithfulness(
            payload.idea,
            context_bundle,
            insight,
        )
    else:
        grounding_status = "not_requested"

    session = AnalysisSession(
        user_id=current_user.id,
        idea_text=payload.idea,
        raw_output={
            **insight.model_dump(),
            "_grounding_status": grounding_status,
            "_faithfulness_corrected": faithfulness_corrected,
        },
        confidence_score=insight.confidence_score,
        used_fallback=context_bundle.used_fallback,
    )
    db.add(session)
    await db.flush()

    similarity_mean = compute_avg_similarity(context_bundle.docs)
    similarity_dist = compute_similarity_distribution(context_bundle.docs)
    token_ratio = compute_context_token_ratio(context_bundle)

    fallback_relevance = [chunk.relevance_score for chunk in dynamic_chunks if chunk.relevance_score is not None]
    fallback_stats = compute_fallback_stats(
        used_fallback=context_bundle.used_fallback,
        articles_fetched=filter_stats.get("fetched", 0),
        articles_surviving=filter_stats.get("after_recency", 0),
        avg_fallback_relevance=(mean(fallback_relevance) if fallback_relevance else None),
    )

    generation_stats = compute_generation_stats(
        latency_ms=latency_ms,
        retry_count=retry_count,
        validation_passed=grounding_status != "not_grounded",
    )

    tier1_metrics = {
        "avg_similarity_score": similarity_mean,
        "docs_above_threshold": similarity_dist["above_threshold"],
        "total_docs_retrieved": similarity_dist["total"],
        "context_total_tokens": token_ratio["total_tokens"],
        "context_local_ratio": token_ratio["local_ratio"],
        "context_dynamic_ratio": token_ratio["dynamic_ratio"],
        "used_fallback": fallback_stats["used_fallback"],
        "articles_fetched": fallback_stats["articles_fetched"],
        "articles_surviving": fallback_stats["articles_surviving"],
        "llm_latency_ms": generation_stats["latency_ms"],
        "llm_retry_count": generation_stats["retry_count"],
    }

    evaluation_status = "not_requested"

    if payload.run_evaluation:
        evaluation_status = "skipped"
        run_ragas = should_run_ragas()
        if run_ragas:
            evaluation_status = "pending"

        eval_log = EvaluationLog(
            session_id=session.id,
            avg_similarity_score=similarity_mean,
            min_similarity_score=similarity_dist["min"],
            max_similarity_score=similarity_dist["max"],
            docs_above_threshold=similarity_dist["above_threshold"],
            total_docs_retrieved=similarity_dist["total"],
            context_total_tokens=token_ratio["total_tokens"],
            context_local_ratio=token_ratio["local_ratio"],
            context_dynamic_ratio=token_ratio["dynamic_ratio"],
            used_fallback=fallback_stats["used_fallback"],
            articles_fetched=fallback_stats["articles_fetched"],
            articles_surviving=fallback_stats["articles_surviving"],
            avg_fallback_relevance=fallback_stats["avg_fallback_relevance"],
            llm_latency_ms=generation_stats["latency_ms"],
            llm_retry_count=generation_stats["retry_count"],
            llm_validation_passed=generation_stats["validation_passed"],
            ragas_eval_status=evaluation_status,
            query=payload.idea,
            retrieved_docs=[
                {
                    "source": doc.source,
                    "doc_type": doc.doc_type,
                    "similarity_score": doc.similarity_score,
                }
                for doc in context_bundle.docs
            ],
            generated_output=json.dumps(insight.model_dump(), ensure_ascii=True),
        )
        db.add(eval_log)

        if run_ragas:
            background_tasks.add_task(
                _run_ragas_in_background,
                session.id,
                payload.idea,
                [doc.text for doc in context_bundle.docs],
                json.dumps(insight.model_dump(), ensure_ascii=True),
            )

    await db.commit()

    return {
        "session_id": str(session.id),
        "insights": insight,
        "grounding_status": grounding_status,
        "faithfulness_corrected": faithfulness_corrected,
        "used_fallback": context_bundle.used_fallback,
        "retrieved_sources": sorted({doc.source for doc in context_bundle.docs}),
        "retrieval_diagnostics": retrieval_diagnostics,
        "evaluation_status": evaluation_status,
        "tier1_metrics": tier1_metrics,
    }