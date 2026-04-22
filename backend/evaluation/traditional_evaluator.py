from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

from bert_score import score as bert_score
from rouge_score import rouge_scorer

from config import settings
from reasoning.llm_client import (
    build_insufficient_context_insight,
    generate_insight,
    should_abstain_for_coverage,
)
from retrieval.fallback.context_builder import build_context_bundle
from retrieval.retriever import RetrievedDoc, retrieve_local


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkItem:
    query: str
    document_id: str
    reference_answer: str


def _benchmark_path() -> Path:
    configured = Path(settings.TRADITIONAL_EVAL_BENCHMARK_PATH)
    if configured.is_absolute():
        return configured

    backend_root = Path(__file__).resolve().parent.parent
    return (backend_root / configured).resolve()


def _load_benchmark() -> tuple[str, list[BenchmarkItem]]:
    path = _benchmark_path()
    if not path.exists():
        raise FileNotFoundError(f"Traditional evaluation benchmark file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Traditional evaluation benchmark JSON must be an object")

    benchmark_name = str(payload.get("benchmark_name") or "traditional_benchmark")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Traditional evaluation benchmark must include a non-empty 'items' list")

    items: list[BenchmarkItem] = []
    for idx, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            continue

        query = str(raw.get("query") or "").strip()
        document_id = str(raw.get("document_id") or "").strip()
        reference_answer = str(raw.get("reference_answer") or raw.get("gold_answer") or "").strip()
        if not query or not document_id or not reference_answer:
            logger.warning("Skipping invalid benchmark row %d in %s", idx, path)
            continue

        items.append(
            BenchmarkItem(
                query=query,
                document_id=document_id,
                reference_answer=reference_answer,
            )
        )

    if not items:
        raise ValueError("Traditional evaluation benchmark has no valid items")

    return benchmark_name, items


def _normalize_doc_id(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def _candidate_doc_ids(doc: RetrievedDoc) -> list[str]:
    candidates = [doc.source, doc.source_rel]
    if doc.source:
        candidates.append(Path(doc.source).name)

    normalized: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        cleaned = _normalize_doc_id(str(value or ""))
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return normalized


def _is_doc_match(candidate: str, target: str) -> bool:
    if candidate == target:
        return True
    return candidate.endswith(f"/{target}") or target.endswith(f"/{candidate}")


def _target_rank_at_k(retrieved_docs: list[RetrievedDoc], target_document_id: str, k: int) -> int | None:
    target = _normalize_doc_id(target_document_id)
    if not target:
        return None

    ranked_sources: list[str] = []
    seen_sources: set[str] = set()

    for doc in retrieved_docs[:k]:
        candidates = _candidate_doc_ids(doc)
        source_key = candidates[0] if candidates else ""
        if not source_key or source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        ranked_sources.append(source_key)

    for idx, source_id in enumerate(ranked_sources, start=1):
        if _is_doc_match(source_id, target):
            return idx

    return None


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _insight_to_text(insight) -> str:
    lines: list[str] = []

    for value in (
        insight.brand_diagnosis,
        insight.market_insight,
        insight.final_positioning,
        insight.target_audience,
        insight.chosen_strategy,
    ):
        text = str(value or "").strip()
        if text:
            lines.append(text)

    for values in (
        insight.suggested_positioning,
        insight.risks,
        insight.opportunities,
        insight.rejected_directions,
        insight.trade_offs,
    ):
        if not values:
            continue
        lines.extend(str(item).strip() for item in values if str(item).strip())

    if lines:
        return "\n".join(lines)

    fallback = str(insight.notion_page_content or "").strip()
    if fallback:
        return fallback

    return "No generated answer text."


async def run_traditional_benchmark_evaluation() -> dict:
    benchmark_name, items = _load_benchmark()
    top_k = max(int(settings.TRADITIONAL_EVAL_TOP_K), 1)

    recall_scores: list[float] = []
    ap_scores: list[float] = []
    rouge_scores: list[float] = []
    candidate_answers: list[str] = []
    reference_answers: list[str] = []
    per_query: list[dict] = []

    for item in items:
        retrieved_docs, low_confidence, diagnostics = retrieve_local(item.query, top_k=top_k)
        rank = _target_rank_at_k(retrieved_docs, item.document_id, k=top_k)

        recall_at_k = 1.0 if rank is not None else 0.0
        average_precision_at_k = (1.0 / rank) if rank is not None else 0.0
        recall_scores.append(recall_at_k)
        ap_scores.append(average_precision_at_k)

        context_bundle = build_context_bundle(local_docs=retrieved_docs, dynamic_chunks=[])
        should_abstain, abstain_reason, _coverage = should_abstain_for_coverage(
            context_bundle,
            low_confidence=low_confidence,
            fallback_requested=False,
        )

        if should_abstain:
            insight = build_insufficient_context_insight(item.query, context_bundle, abstain_reason)
        else:
            try:
                # Disable model request timeout for benchmark evaluation, as requested.
                insight, _latency_ms, _retry_count = await generate_insight(
                    item.query,
                    context_bundle,
                    timeout_override_seconds=0,
                )
            except Exception as exc:
                logger.warning(
                    "Traditional evaluation generation failed for benchmark query '%s': %s",
                    item.query[:120],
                    exc,
                )
                insight = build_insufficient_context_insight(
                    item.query,
                    context_bundle,
                    "Benchmark generation failed; falling back to conservative response.",
                )

        candidate_answer = _insight_to_text(insight)
        candidate_answers.append(candidate_answer)
        reference_answers.append(item.reference_answer)

        rouge_value = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(
            item.reference_answer,
            candidate_answer,
        )["rougeL"].fmeasure
        rouge_scores.append(float(rouge_value))

        per_query.append(
            {
                "query": item.query,
                "document_id": item.document_id,
                "retrieval_hit": rank is not None,
                "target_rank": rank,
                "recall_at_k": recall_at_k,
                "average_precision_at_k": average_precision_at_k,
                "rouge_l_f1": float(rouge_value),
                "retrieval_diagnostics": diagnostics,
            }
        )

    _, _, bert_f1 = bert_score(
        candidate_answers,
        reference_answers,
        lang="en",
        model_type="distilbert-base-uncased",
        device="cpu",
        verbose=False,
    )
    bert_scores = [float(value) for value in bert_f1.tolist()]

    for idx, value in enumerate(bert_scores):
        per_query[idx]["bertscore_f1"] = value

    return {
        "benchmark_name": benchmark_name,
        "k": top_k,
        "query_count": len(items),
        "recall_at_k": _safe_mean(recall_scores),
        "map_at_k": _safe_mean(ap_scores),
        "rouge_l_f1": _safe_mean(rouge_scores),
        "bertscore_f1": _safe_mean(bert_scores),
        "per_query": per_query,
    }
