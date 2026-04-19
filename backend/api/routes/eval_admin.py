from __future__ import annotations

from datetime import date, datetime, time, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from auth.models import User
from database import get_db
from evaluation.models import EvaluationLog


router = APIRouter(prefix="/eval", tags=["evaluation-admin"])


def _require_admin(user: User) -> None:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")


@router.get("/summary")
async def get_eval_summary(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    _require_admin(current_user)

    totals = await db.execute(
        select(
            func.count(EvaluationLog.id),
            func.avg(EvaluationLog.avg_similarity_score),
            func.avg(func.cast(EvaluationLog.used_fallback, func.Float)) * 100.0,
            func.avg(EvaluationLog.articles_surviving),
            func.avg(EvaluationLog.llm_latency_ms),
            func.avg(EvaluationLog.llm_retry_count),
        )
    )
    (
        total_sessions,
        avg_similarity_score,
        fallback_trigger_rate,
        avg_articles_surviving,
        avg_latency,
        avg_retry,
    ) = totals.one()

    ragas_count_q = await db.execute(
        select(func.count(EvaluationLog.id)).where(EvaluationLog.ragas_eval_status == "completed")
    )
    sessions_with_ragas = int(ragas_count_q.scalar_one() or 0)

    ragas_avgs_q = await db.execute(
        select(
            func.avg(EvaluationLog.context_precision),
            func.avg(EvaluationLog.context_recall),
            func.avg(EvaluationLog.faithfulness),
            func.avg(EvaluationLog.answer_relevance),
        )
    )
    avg_context_precision, avg_context_recall, avg_faithfulness, avg_answer_relevance = ragas_avgs_q.one()

    return {
        "total_sessions": int(total_sessions or 0),
        "sessions_with_ragas": sessions_with_ragas,
        "avg_similarity_score": float(avg_similarity_score or 0.0),
        "fallback_trigger_rate": float(fallback_trigger_rate or 0.0),
        "avg_articles_surviving_filter": float(avg_articles_surviving or 0.0),
        "avg_llm_latency_ms": float(avg_latency or 0.0),
        "avg_retry_count": float(avg_retry or 0.0),
        "ragas_scores": {
            "avg_context_precision": float(avg_context_precision) if avg_context_precision is not None else None,
            "avg_context_recall": float(avg_context_recall) if avg_context_recall is not None else None,
            "avg_faithfulness": float(avg_faithfulness) if avg_faithfulness is not None else None,
            "avg_answer_relevance": float(avg_answer_relevance) if avg_answer_relevance is not None else None,
        },
    }


@router.get("/sessions")
async def list_eval_sessions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=50),
    ragas_eval_status: str | None = None,
    used_fallback: bool | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    _require_admin(current_user)

    filters = []
    if ragas_eval_status:
        filters.append(EvaluationLog.ragas_eval_status == ragas_eval_status)
    if used_fallback is not None:
        filters.append(EvaluationLog.used_fallback == used_fallback)
    if date_from is not None:
        filters.append(EvaluationLog.created_at >= datetime.combine(date_from, time.min, tzinfo=timezone.utc))
    if date_to is not None:
        filters.append(EvaluationLog.created_at <= datetime.combine(date_to, time.max, tzinfo=timezone.utc))

    where_clause = and_(*filters) if filters else None

    total_stmt = select(func.count()).select_from(EvaluationLog)
    if where_clause is not None:
        total_stmt = total_stmt.where(where_clause)

    total_q = await db.execute(total_stmt)
    total = int(total_q.scalar_one() or 0)

    stmt = select(EvaluationLog).order_by(EvaluationLog.created_at.desc())
    if where_clause is not None:
        stmt = stmt.where(where_clause)

    stmt = stmt.offset((page - 1) * page_size).limit(page_size)

    rows_q = await db.execute(stmt)
    rows = rows_q.scalars().all()

    items = [
        {
            "id": str(row.id),
            "session_id": str(row.session_id),
            "avg_similarity_score": row.avg_similarity_score,
            "min_similarity_score": row.min_similarity_score,
            "max_similarity_score": row.max_similarity_score,
            "docs_above_threshold": row.docs_above_threshold,
            "total_docs_retrieved": row.total_docs_retrieved,
            "context_total_tokens": row.context_total_tokens,
            "context_local_ratio": row.context_local_ratio,
            "context_dynamic_ratio": row.context_dynamic_ratio,
            "used_fallback": row.used_fallback,
            "articles_fetched": row.articles_fetched,
            "articles_surviving": row.articles_surviving,
            "avg_fallback_relevance": row.avg_fallback_relevance,
            "llm_latency_ms": row.llm_latency_ms,
            "llm_retry_count": row.llm_retry_count,
            "llm_validation_passed": row.llm_validation_passed,
            "context_precision": row.context_precision,
            "context_recall": row.context_recall,
            "faithfulness": row.faithfulness,
            "answer_relevance": row.answer_relevance,
            "ragas_eval_status": row.ragas_eval_status,
            "query": row.query,
            "retrieved_docs": row.retrieved_docs,
            "generated_output": row.generated_output,
            "created_at": row.created_at.isoformat() if isinstance(row.created_at, datetime) else None,
        }
        for row in rows
    ]

    return {"page": page, "page_size": page_size, "total": total, "items": items}