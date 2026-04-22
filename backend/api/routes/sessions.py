from __future__ import annotations

import asyncio
import json
from datetime import datetime
from statistics import mean
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from sqlalchemy import func, select
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
from evaluation.models import ActionLog, AnalysisSession, EvaluationLog, SessionChatTurn
from evaluation.ragas_evaluator import run_ragas_evaluation, should_run_ragas
from ingestion.embedder import get_embedder_model
from reasoning.llm_client import (
    build_conservative_insight,
    build_insufficient_context_insight,
    enforce_faithfulness,
    generate_insight,
    should_abstain_for_coverage,
)
from reasoning.schema import InsightOutput
from retrieval.fallback.context_builder import build_context_bundle
from retrieval.fallback.dynamic_retriever import retrieve_dynamic_chunks
from retrieval.retriever import retrieve_local


router = APIRouter(tags=["sessions"])


class SessionChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    message: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=settings.TOP_K_DEFAULT, ge=1, le=10, validation_alias=AliasChoices("top_k", "topK"))
    use_fallback: bool | None = Field(default=True, validation_alias=AliasChoices("use_fallback", "useFallback"))
    run_evaluation: bool | None = Field(
        default=True,
        validation_alias=AliasChoices("run_evaluation", "runEvaluation"),
    )


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


def _serialize_insight_output(raw_insight: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw_insight, dict):
        return {}
    try:
        return InsightOutput.model_validate(raw_insight).model_dump()
    except Exception:
        # Keep compatibility with any legacy payload shape stored before schema updates.
        return raw_insight


def _normalize_ragas_status(status: str | None) -> str:
    normalized = str(status or "").strip()
    if normalized == "fallback_completed":
        return "failed"
    return normalized


def _serialize_evaluation_log(eval_row: EvaluationLog | None) -> dict[str, Any] | None:
    if eval_row is None:
        return None

    status = _normalize_ragas_status(eval_row.ragas_eval_status)
    expose_ragas_metrics = status == "completed"

    return {
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
        "context_precision": eval_row.context_precision if expose_ragas_metrics else None,
        "context_recall": eval_row.context_recall if expose_ragas_metrics else None,
        "faithfulness": eval_row.faithfulness if expose_ragas_metrics else None,
        "answer_relevance": eval_row.answer_relevance if expose_ragas_metrics else None,
        "ragas_eval_status": status,
        "evaluation_mode": "ragas",
        "query": eval_row.query,
        "retrieved_docs": eval_row.retrieved_docs,
        "generated_output": eval_row.generated_output,
        "created_at": eval_row.created_at.isoformat() if isinstance(eval_row.created_at, datetime) else None,
    }


def _serialize_action_log(action: ActionLog) -> dict[str, Any]:
    return {
        "id": str(action.id),
        "action_type": action.action_type,
        "title": action.title,
        "description": action.description,
        "priority": action.priority,
        "target_provider": action.target_provider,
        "status": action.status,
        "external_id": action.external_id,
        "error_message": action.error_message,
        "created_at": action.created_at.isoformat() if isinstance(action.created_at, datetime) else None,
    }


def _serialize_chat_turn(turn: SessionChatTurn) -> dict[str, Any]:
    return {
        "id": str(turn.id),
        "user_message": turn.user_message,
        "assistant_message": turn.assistant_message,
        "insights": _serialize_insight_output(turn.insight_output),
        "grounding_status": turn.grounding_status,
        "faithfulness_corrected": turn.faithfulness_corrected,
        "used_fallback": turn.used_fallback,
        "retrieval_diagnostics": turn.retrieval_diagnostics if isinstance(turn.retrieval_diagnostics, dict) else None,
        "created_at": turn.created_at.isoformat() if isinstance(turn.created_at, datetime) else None,
    }


def _conversation_from_chat_turns(chat_turns: list[SessionChatTurn]) -> list[dict[str, str]]:
    conversation: list[dict[str, str]] = []
    for turn in chat_turns:
        user_message = (turn.user_message or "").strip()
        assistant_message = (turn.assistant_message or "").strip()
        if user_message:
            conversation.append({"role": "user", "content": user_message[:2000]})
        if assistant_message:
            conversation.append({"role": "assistant", "content": assistant_message[:2000]})
    return conversation[-20:]


def _conversation_from_legacy_raw_output(raw_output: dict[str, Any] | None) -> list[dict[str, str]]:
    if not isinstance(raw_output, dict):
        return []

    conversation = raw_output.get("_conversation")
    if not isinstance(conversation, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in conversation:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue

        normalized.append({"role": role, "content": content[:2000]})

    return normalized[-20:]


def _assistant_summary_from_raw_output(raw_output: dict[str, Any] | None) -> str:
    if not isinstance(raw_output, dict):
        return "Initial brand strategy analysis generated for this decision session."

    candidates = [
        raw_output.get("abstention_message"),
        raw_output.get("market_insight"),
        raw_output.get("brand_diagnosis"),
        raw_output.get("idea_summary"),
    ]
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text[:2000]

    return "Initial brand strategy analysis generated for this decision session."


def _default_conversation_from_session(idea_text: str, raw_output: dict[str, Any] | None) -> list[dict[str, str]]:
    user_text = str(idea_text or "").strip()
    if not user_text:
        user_text = "Original brand decision question was not captured."

    return [
        {"role": "user", "content": user_text[:2000]},
        {"role": "assistant", "content": _assistant_summary_from_raw_output(raw_output)},
    ]


def _build_chat_reasoning_query(
    idea_text: str,
    raw_output: dict[str, Any] | None,
    conversation: list[dict[str, str]],
    message: str,
) -> str:
    base = [
        "You are continuing an existing Brand Decision session.",
        "Answer the latest follow-up question directly and with fresh reasoning.",
        "If this follow-up changes topic or intent, do NOT paraphrase previous assistant text.",
        "Use prior turns only for continuity, not as a template for repetition.",
        "Unless the user explicitly asks to focus on one decision type, cover positioning, differentiation, messaging, trust, audience, pricing, and narrative impacts.",
        "",
        "Original Brand Decision Question:",
        idea_text.strip(),
        "",
    ]

    if isinstance(raw_output, dict):
        diagnosis = str(raw_output.get("brand_diagnosis") or raw_output.get("idea_summary") or "").strip()
        final_positioning = str(raw_output.get("final_positioning") or "").strip()
        target_audience = str(raw_output.get("target_audience") or "").strip()
        chosen_strategy = str(raw_output.get("chosen_strategy") or "").strip()

        suggested = raw_output.get("suggested_positioning") or raw_output.get("recommendations") or []
        suggested_lines = []
        if isinstance(suggested, list):
            for value in suggested[:3]:
                text = str(value).strip()
                if text:
                    suggested_lines.append(f"- {text}")

        summary_lines = [
            "Baseline Structured Insight Snapshot:",
            f"- Brand Diagnosis: {diagnosis or 'n/a'}",
            f"- Final Positioning: {final_positioning or 'n/a'}",
            f"- Target Audience: {target_audience or 'n/a'}",
            f"- Chosen Strategy: {chosen_strategy or 'n/a'}",
        ]

        if suggested_lines:
            summary_lines.append("- Suggested Positioning:")
            summary_lines.extend(suggested_lines)

        base.extend(summary_lines)
        base.append("")

    if conversation:
        base.append("Conversation History (latest turns):")
        for turn in conversation[-8:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            base.append(f"{role}: {turn['content']}")
        base.append("")

    base.extend(
        [
            "Current Follow-up Question:",
            message.strip(),
            "",
            "Return a complete structured response for this follow-up and highlight what is newly inferred for this question.",
        ]
    )

    return "\n".join(base)


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
                        "status": _normalize_ragas_status(eval_row.ragas_eval_status),
                        "context_precision": (
                            eval_row.context_precision
                            if _normalize_ragas_status(eval_row.ragas_eval_status) == "completed"
                            else None
                        ),
                        "context_recall": (
                            eval_row.context_recall
                            if _normalize_ragas_status(eval_row.ragas_eval_status) == "completed"
                            else None
                        ),
                        "faithfulness": (
                            eval_row.faithfulness
                            if _normalize_ragas_status(eval_row.ragas_eval_status) == "completed"
                            else None
                        ),
                        "answer_relevance": (
                            eval_row.answer_relevance
                            if _normalize_ragas_status(eval_row.ragas_eval_status) == "completed"
                            else None
                        ),
                        "evaluation_mode": "ragas",
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

    chat_q = await db.execute(
        select(SessionChatTurn)
        .where(SessionChatTurn.session_id == session.id)
        .order_by(SessionChatTurn.created_at.asc())
    )
    chat_rows = chat_q.scalars().all()

    raw_output = session.raw_output if isinstance(session.raw_output, dict) else {}
    conversation = _conversation_from_chat_turns(chat_rows)
    if not conversation:
        conversation = _conversation_from_legacy_raw_output(raw_output)
    if not conversation:
        conversation = _default_conversation_from_session(session.idea_text, raw_output)

    evaluation_log = _serialize_evaluation_log(eval_row)
    action_logs = [_serialize_action_log(action) for action in action_rows]
    chat_turns = [_serialize_chat_turn(turn) for turn in chat_rows]

    return {
        "id": str(session.id),
        "idea_text": session.idea_text,
        "raw_output": _serialize_insight_output(raw_output),
        "confidence_score": session.confidence_score,
        "used_fallback": session.used_fallback,
        "created_at": session.created_at.isoformat() if isinstance(session.created_at, datetime) else None,
        "conversation": conversation,
        "chat_turns": chat_turns,
        "evaluation_log": evaluation_log,
        "action_logs": action_logs,
    }


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    session_q = await db.execute(select(AnalysisSession).where(AnalysisSession.id == session_id))
    session = session_q.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this session")

    await db.delete(session)
    await db.commit()
    return Response(status_code=204)


@router.post("/sessions/{session_id}/chat")
async def chat_in_session(
    session_id: UUID,
    payload: SessionChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    session_q = await db.execute(select(AnalysisSession).where(AnalysisSession.id == session_id))
    session = session_q.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this session")

    chat_q = await db.execute(
        select(SessionChatTurn)
        .where(SessionChatTurn.session_id == session.id)
        .order_by(SessionChatTurn.created_at.asc())
    )
    existing_turns = chat_q.scalars().all()

    raw_output = session.raw_output if isinstance(session.raw_output, dict) else {}
    conversation = _conversation_from_chat_turns(existing_turns)
    if not conversation:
        conversation = _conversation_from_legacy_raw_output(raw_output)
    if not conversation:
        conversation = _default_conversation_from_session(session.idea_text, raw_output)

    user_message = payload.message.strip()
    recent_user_turns = [turn.user_message.strip() for turn in existing_turns[-3:] if turn.user_message.strip()]
    retrieval_lines = [f"Original decision question: {session.idea_text.strip()}"]
    if recent_user_turns:
        retrieval_lines.append("Recent follow-up questions:")
        retrieval_lines.extend(f"- {value}" for value in recent_user_turns)
    retrieval_lines.append(f"Current follow-up question: {user_message}")
    retrieval_query = "\n".join(retrieval_lines)

    reasoning_query = _build_chat_reasoning_query(
        idea_text=session.idea_text,
        raw_output=raw_output,
        conversation=conversation,
        message=user_message,
    )

    top_k = payload.top_k if payload.top_k is not None else settings.TOP_K_DEFAULT
    use_fallback = bool(payload.use_fallback) if payload.use_fallback is not None else True
    run_evaluation = bool(payload.run_evaluation) if payload.run_evaluation is not None else True

    local_docs, low_confidence, retrieval_diagnostics = retrieve_local(retrieval_query, top_k)

    dynamic_chunks = []
    filter_stats = {
        "fetched": 0,
        "after_recency": 0,
    }
    if low_confidence and use_fallback:
        dynamic_chunks, filter_stats = await retrieve_dynamic_chunks(
            query=retrieval_query,
            local_docs=local_docs,
            embedder=get_embedder_model(),
        )

    context_bundle = build_context_bundle(local_docs=local_docs, dynamic_chunks=dynamic_chunks)

    should_abstain, abstain_reason, coverage_metrics = should_abstain_for_coverage(
        context_bundle,
        low_confidence=low_confidence,
        fallback_requested=use_fallback,
    )

    if should_abstain:
        insight = build_insufficient_context_insight(user_message, context_bundle, abstain_reason)
        latency_ms = 0.0
        retry_count = 0
        grounding_status = "insufficient_context"
        faithfulness_corrected = False
    elif settings.BYPASS_LLM_CALLS:
        insight = build_conservative_insight(reasoning_query, context_bundle)
        latency_ms = 0.0
        retry_count = 0
        grounding_status = "bypassed"
        faithfulness_corrected = False
    else:
        insight, latency_ms, retry_count = await generate_insight(reasoning_query, context_bundle)
        faithfulness_corrected = False
        if settings.ENABLE_GROUNDING_CHECK:
            insight, grounding_status, faithfulness_corrected = await enforce_faithfulness(
                reasoning_query,
                context_bundle,
                insight,
            )
        else:
            grounding_status = "not_requested"

    assistant_message = (
        (insight.abstention_message or "").strip()
        or (insight.market_insight or "").strip()
        or (insight.brand_diagnosis or "").strip()
        or "Updated recommendation generated for this follow-up question."
    )

    enriched_retrieval_diagnostics = dict(retrieval_diagnostics)
    enriched_retrieval_diagnostics.update(
        {
            "coverage_metrics": coverage_metrics,
            "abstained": should_abstain,
            "abstain_reason": abstain_reason if should_abstain else None,
        }
    )

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

    new_turn = SessionChatTurn(
        session_id=session.id,
        user_message=user_message[:2000],
        assistant_message=assistant_message[:2000],
        insight_output=insight.model_dump(),
        grounding_status=grounding_status,
        faithfulness_corrected=faithfulness_corrected,
        used_fallback=context_bundle.used_fallback,
        retrieval_diagnostics=enriched_retrieval_diagnostics,
    )
    db.add(new_turn)
    await db.flush()

    updated_conversation = [
        *conversation,
        {"role": "user", "content": user_message[:2000]},
        {"role": "assistant", "content": assistant_message[:2000]},
    ][-20:]

    session.confidence_score = insight.confidence_score
    session.used_fallback = context_bundle.used_fallback

    evaluation_status = "not_requested"
    run_ragas = False
    if run_evaluation:
        evaluation_status = "skipped"
        run_ragas = should_run_ragas()
        if run_ragas:
            evaluation_status = "pending"

    eval_q = await db.execute(select(EvaluationLog).where(EvaluationLog.session_id == session.id))
    eval_row = eval_q.scalar_one_or_none()
    if eval_row is None:
        eval_row = EvaluationLog(
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
            query=user_message,
            retrieved_docs=[],
            generated_output="",
        )
        db.add(eval_row)

    eval_row.avg_similarity_score = similarity_mean
    eval_row.min_similarity_score = similarity_dist["min"]
    eval_row.max_similarity_score = similarity_dist["max"]
    eval_row.docs_above_threshold = similarity_dist["above_threshold"]
    eval_row.total_docs_retrieved = similarity_dist["total"]
    eval_row.context_total_tokens = token_ratio["total_tokens"]
    eval_row.context_local_ratio = token_ratio["local_ratio"]
    eval_row.context_dynamic_ratio = token_ratio["dynamic_ratio"]
    eval_row.used_fallback = fallback_stats["used_fallback"]
    eval_row.articles_fetched = fallback_stats["articles_fetched"]
    eval_row.articles_surviving = fallback_stats["articles_surviving"]
    eval_row.avg_fallback_relevance = fallback_stats["avg_fallback_relevance"]
    eval_row.llm_latency_ms = generation_stats["latency_ms"]
    eval_row.llm_retry_count = generation_stats["retry_count"]
    eval_row.llm_validation_passed = generation_stats["validation_passed"]
    eval_row.context_precision = None
    eval_row.context_recall = None
    eval_row.faithfulness = None
    eval_row.answer_relevance = None
    eval_row.ragas_eval_status = evaluation_status
    eval_row.query = user_message
    eval_row.retrieved_docs = [
        {
            "source": doc.source,
            "doc_type": doc.doc_type,
            "similarity_score": doc.similarity_score,
        }
        for doc in context_bundle.docs
    ]
    eval_row.generated_output = json.dumps(insight.model_dump(), ensure_ascii=True)

    await db.commit()

    if run_ragas:
        asyncio.create_task(
            _run_ragas_in_background(
                session.id,
                user_message,
                [doc.text for doc in context_bundle.docs],
                json.dumps(insight.model_dump(), ensure_ascii=True),
            )
        )

    action_q = await db.execute(
        select(ActionLog).where(ActionLog.session_id == session.id).order_by(ActionLog.created_at.desc())
    )
    action_rows = action_q.scalars().all()

    return {
        "session_id": str(session.id),
        "insights": insight,
        "grounding_status": grounding_status,
        "faithfulness_corrected": faithfulness_corrected,
        "used_fallback": context_bundle.used_fallback,
        "retrieval_diagnostics": enriched_retrieval_diagnostics,
        "chat_turn": _serialize_chat_turn(new_turn),
        "evaluation_log": _serialize_evaluation_log(eval_row),
        "action_logs": [_serialize_action_log(action) for action in action_rows],
        "conversation": updated_conversation,
    }
