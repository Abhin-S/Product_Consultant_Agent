from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from auth.models import User
from database import get_db
from evaluation.models import ActionLog, AnalysisSession, EvaluationLog


router = APIRouter(tags=["sessions"])


@router.get("/sessions")
async def list_sessions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    offset = (page - 1) * page_size

    total_q = await db.execute(
        select(func.count()).select_from(AnalysisSession).where(AnalysisSession.user_id == current_user.id)
    )
    total = int(total_q.scalar_one() or 0)

    sessions_q = await db.execute(
        select(AnalysisSession)
        .where(AnalysisSession.user_id == current_user.id)
        .order_by(AnalysisSession.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    sessions = sessions_q.scalars().all()

    items: list[dict] = []
    for session in sessions:
        eval_q = await db.execute(select(EvaluationLog).where(EvaluationLog.session_id == session.id))
        eval_row = eval_q.scalar_one_or_none()

        actions_count_q = await db.execute(
            select(func.count()).select_from(ActionLog).where(ActionLog.session_id == session.id)
        )
        actions_count = int(actions_count_q.scalar_one() or 0)

        items.append(
            {
                "id": str(session.id),
                "created_at": session.created_at.isoformat() if isinstance(session.created_at, datetime) else None,
                "idea_text": session.idea_text,
                "confidence_score": session.confidence_score,
                "used_fallback": session.used_fallback,
                "actions_taken": actions_count,
                "tier1_metrics": (
                    {
                        "avg_similarity_score": eval_row.avg_similarity_score,
                        "docs_above_threshold": eval_row.docs_above_threshold,
                        "total_docs_retrieved": eval_row.total_docs_retrieved,
                        "context_total_tokens": eval_row.context_total_tokens,
                        "context_local_ratio": eval_row.context_local_ratio,
                        "context_dynamic_ratio": eval_row.context_dynamic_ratio,
                        "used_fallback": eval_row.used_fallback,
                        "articles_fetched": eval_row.articles_fetched,
                        "articles_surviving": eval_row.articles_surviving,
                        "llm_latency_ms": eval_row.llm_latency_ms,
                        "llm_retry_count": eval_row.llm_retry_count,
                    }
                    if eval_row
                    else None
                ),
                "ragas": (
                    {
                        "status": eval_row.ragas_eval_status,
                        "context_precision": eval_row.context_precision,
                        "context_recall": eval_row.context_recall,
                        "faithfulness": eval_row.faithfulness,
                        "answer_relevance": eval_row.answer_relevance,
                    }
                    if eval_row
                    else None
                ),
            }
        )

    return {"page": page, "page_size": page_size, "total": total, "items": items}


@router.get("/sessions/{session_id}")
async def get_session_detail(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    session_q = await db.execute(select(AnalysisSession).where(AnalysisSession.id == session_id))
    session = session_q.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this session")

    eval_q = await db.execute(select(EvaluationLog).where(EvaluationLog.session_id == session.id))
    eval_row = eval_q.scalar_one_or_none()

    action_q = await db.execute(
        select(ActionLog).where(ActionLog.session_id == session.id).order_by(ActionLog.created_at.desc())
    )
    action_rows = action_q.scalars().all()

    return {
        "id": str(session.id),
        "idea_text": session.idea_text,
        "raw_output": session.raw_output,
        "confidence_score": session.confidence_score,
        "used_fallback": session.used_fallback,
        "created_at": session.created_at.isoformat() if isinstance(session.created_at, datetime) else None,
        "evaluation_log": (
            {
                "avg_similarity_score": eval_row.avg_similarity_score,
                "min_similarity_score": eval_row.min_similarity_score,
                "max_similarity_score": eval_row.max_similarity_score,
                "docs_above_threshold": eval_row.docs_above_threshold,
                "total_docs_retrieved": eval_row.total_docs_retrieved,
                "context_total_tokens": eval_row.context_total_tokens,
                "context_local_ratio": eval_row.context_local_ratio,
                "context_dynamic_ratio": eval_row.context_dynamic_ratio,
                "used_fallback": eval_row.used_fallback,
                "articles_fetched": eval_row.articles_fetched,
                "articles_surviving": eval_row.articles_surviving,
                "avg_fallback_relevance": eval_row.avg_fallback_relevance,
                "llm_latency_ms": eval_row.llm_latency_ms,
                "llm_retry_count": eval_row.llm_retry_count,
                "llm_validation_passed": eval_row.llm_validation_passed,
                "context_precision": eval_row.context_precision,
                "context_recall": eval_row.context_recall,
                "faithfulness": eval_row.faithfulness,
                "answer_relevance": eval_row.answer_relevance,
                "ragas_eval_status": eval_row.ragas_eval_status,
                "query": eval_row.query,
                "retrieved_docs": eval_row.retrieved_docs,
                "generated_output": eval_row.generated_output,
                "created_at": (
                    eval_row.created_at.isoformat() if isinstance(eval_row.created_at, datetime) else None
                ),
            }
            if eval_row
            else None
        ),
        "action_logs": [
            {
                "id": str(action.id),
                "action_type": action.action_type,
                "title": action.title,
                "description": action.description,
                "priority": action.priority,
                "target_provider": action.target_provider,
                "status": action.status,
                "external_id": action.external_id,
                "error_message": action.error_message,
                "created_at": (
                    action.created_at.isoformat() if isinstance(action.created_at, datetime) else None
                ),
            }
            for action in action_rows
        ],
    }