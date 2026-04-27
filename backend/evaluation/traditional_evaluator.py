from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio
import logging
import math
from pathlib import Path
import re

from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from config import settings
from reasoning.llm_client import (
    build_insufficient_context_insight,
    generate_insight,
    should_abstain_for_coverage,
)
from retrieval.fallback.context_builder import build_context_bundle
from retrieval.retriever import RetrievedDoc, retrieve_local
from utils.datetime_utils import now_ist


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkItem:
    query: str
    relevant_document_ids: list[str]
    reference_payload: dict | None
    reference_text: str | None


def _benchmark_path() -> Path:
    configured = Path(settings.TRADITIONAL_EVAL_BENCHMARK_PATH)
    if configured.is_absolute():
        return configured

    backend_root = Path(__file__).resolve().parent.parent
    return (backend_root / configured).resolve()


def _results_path() -> Path:
    backend_root = Path(__file__).resolve().parent.parent
    return (backend_root / "evaluation" / "results").resolve()


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
        document_ids = _parse_document_ids(raw)
        reference_payload, reference_text = _parse_reference_answer(
            raw.get("reference_output", raw.get("reference_answer", raw.get("gold_answer"))),
        )

        if not query or not document_ids or (reference_payload is None and not reference_text):
            logger.warning("Skipping invalid benchmark row %d in %s", idx, path)
            continue

        items.append(
            BenchmarkItem(
                query=query,
                relevant_document_ids=document_ids,
                reference_payload=reference_payload,
                reference_text=reference_text,
            )
        )

    if not items:
        raise ValueError("Traditional evaluation benchmark has no valid items")

    return benchmark_name, items


def _normalize_doc_id(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def _parse_document_ids(raw_item: dict) -> list[str]:
    raw_document_ids = raw_item.get("document_ids") or raw_item.get("relevant_document_ids")

    candidates: list[str]
    if isinstance(raw_document_ids, list):
        candidates = [str(value or "").strip() for value in raw_document_ids]
    else:
        candidates = [str(raw_item.get("document_id") or "").strip()]

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = _normalize_doc_id(candidate)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return normalized


def _parse_reference_answer(raw_reference: object) -> tuple[dict | None, str | None]:
    if isinstance(raw_reference, dict):
        return raw_reference, None

    if isinstance(raw_reference, str):
        cleaned = raw_reference.strip()
        if not cleaned:
            return None, None

        try:
            maybe_json = json.loads(cleaned)
            if isinstance(maybe_json, dict):
                return maybe_json, None
        except json.JSONDecodeError:
            pass

        return None, cleaned

    return None, None


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


def _ranked_unique_sources_at_k(retrieved_docs: list[RetrievedDoc], k: int) -> list[str]:
    ranked_sources: list[str] = []
    seen_sources: set[str] = set()

    for doc in retrieved_docs[:k]:
        candidates = _candidate_doc_ids(doc)
        source_key = candidates[0] if candidates else ""
        if not source_key or source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        ranked_sources.append(source_key)

    return ranked_sources


def _matched_relevant_ranks(
    ranked_sources: list[str],
    relevant_document_ids: list[str],
) -> list[int]:
    if not ranked_sources or not relevant_document_ids:
        return []

    unmatched = set(relevant_document_ids)
    matched_ranks: list[int] = []

    for rank, source_id in enumerate(ranked_sources, start=1):
        matched_target: str | None = None
        for target in unmatched:
            if _is_doc_match(source_id, target):
                matched_target = target
                break
        if matched_target is None:
            continue

        matched_ranks.append(rank)
        unmatched.remove(matched_target)
        if not unmatched:
            break

    return matched_ranks


def _compute_retrieval_metrics(
    retrieved_docs: list[RetrievedDoc],
    relevant_document_ids: list[str],
    k: int,
) -> dict:
    ranked_sources = _ranked_unique_sources_at_k(retrieved_docs, k=k)
    matched_ranks = _matched_relevant_ranks(ranked_sources, relevant_document_ids)

    relevant_count = len(relevant_document_ids)
    hits = len(matched_ranks)

    hit_at_k = 1.0 if hits > 0 else 0.0
    recall_at_k = float(hits / relevant_count) if relevant_count > 0 else 0.0
    precision_at_k = float(hits / float(k)) if k > 0 else 0.0
    mrr_at_k = float(1.0 / matched_ranks[0]) if matched_ranks else 0.0

    ap_running = 0.0
    hits_seen = 0
    matched_rank_set = set(matched_ranks)
    for rank in range(1, len(ranked_sources) + 1):
        if rank not in matched_rank_set:
            continue
        hits_seen += 1
        ap_running += hits_seen / rank

    average_precision_at_k = float(ap_running / relevant_count) if relevant_count > 0 else 0.0

    dcg = float(sum(1.0 / math.log2(rank + 1.0) for rank in matched_ranks))
    ideal_hits = min(relevant_count, k)
    idcg = float(sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1)))
    ndcg_at_k = float(dcg / idcg) if idcg > 0 else 0.0

    first_hit_rank = matched_ranks[0] if matched_ranks else None

    return {
        "ranked_sources": ranked_sources,
        "matched_ranks": matched_ranks,
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "average_precision_at_k": average_precision_at_k,
        "mrr_at_k": mrr_at_k,
        "ndcg_at_k": ndcg_at_k,
        "first_hit_rank": first_hit_rank,
    }


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _write_results_file(result: dict) -> str | None:
    try:
        output_path = _results_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "generated_at": now_ist().isoformat(),
            **result,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return str(output_path)
    except Exception as exc:
        logger.warning("Failed to write traditional evaluation results file: %s", exc)
        return None


def _compact_whitespace(value: object) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def _list_items_to_text(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [_compact_whitespace(item) for item in value]
    return [item for item in items if item]


def _action_item_to_text(action: object) -> str:
    if not isinstance(action, dict):
        return _compact_whitespace(action)

    parts = []
    for key in ("title", "description", "priority", "decision_type", "impact"):
        value = _compact_whitespace(action.get(key))
        if value:
            parts.append(f"{key}={value}")
    return " ; ".join(parts)


def _database_metadata_to_text(metadata: object) -> str:
    if not isinstance(metadata, dict):
        return _compact_whitespace(metadata)

    parts = []
    for key in ("name", "brand_positioning", "brand_risk_level", "confidence_score", "tags"):
        if key not in metadata:
            continue
        value = metadata.get(key)
        if isinstance(value, list):
            list_items = _list_items_to_text(value)
            if list_items:
                parts.append(f"{key}={ ' | '.join(list_items)}")
            continue
        cleaned = _compact_whitespace(value)
        if cleaned:
            parts.append(f"{key}={cleaned}")

    return " ; ".join(parts)


def _normalize_output_payload(payload: dict) -> str:
    lines: list[str] = []
    ordered_fields = (
        "abstention_message",
        "brand_diagnosis",
        "market_insight",
        "suggested_positioning",
        "risks",
        "opportunities",
        "final_positioning",
        "target_audience",
        "chosen_strategy",
        "rejected_directions",
        "trade_offs",
        "actions",
        "confidence_score",
        "notion_page_content",
        "database_metadata",
    )

    seen_fields: set[str] = set()

    for field in ordered_fields:
        if field not in payload:
            continue
        seen_fields.add(field)
        raw_value = payload.get(field)
        if raw_value is None:
            continue

        if field == "actions":
            action_lines = []
            if isinstance(raw_value, list):
                action_lines = [_action_item_to_text(action) for action in raw_value]
            action_lines = [line for line in action_lines if line]
            if action_lines:
                lines.append(f"{field}: {' || '.join(action_lines)}")
            continue

        if field == "database_metadata":
            metadata_text = _database_metadata_to_text(raw_value)
            if metadata_text:
                lines.append(f"{field}: {metadata_text}")
            continue

        if isinstance(raw_value, list):
            list_items = _list_items_to_text(raw_value)
            if list_items:
                lines.append(f"{field}: {' | '.join(list_items)}")
            continue

        cleaned = _compact_whitespace(raw_value)
        if cleaned:
            lines.append(f"{field}: {cleaned}")

    for field in sorted(key for key in payload.keys() if key not in seen_fields):
        raw_value = payload.get(field)
        if raw_value is None:
            continue
        cleaned = _compact_whitespace(raw_value)
        if cleaned:
            lines.append(f"{field}: {cleaned}")

    return "\n".join(lines).strip()


def _insight_to_payload(insight: object) -> dict:
    if hasattr(insight, "model_dump"):
        payload = insight.model_dump(mode="json", exclude_none=True)
        if isinstance(payload, dict):
            return payload

    if isinstance(insight, dict):
        return insight

    return {}


def _reference_to_text(item: BenchmarkItem) -> tuple[str, str]:
    if item.reference_payload is not None:
        normalized = _normalize_output_payload(item.reference_payload)
        if normalized:
            return normalized, "structured"

    fallback = _compact_whitespace(item.reference_text)
    return fallback, "plain_text"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _token_f1(reference_text: str, candidate_text: str) -> float:
    ref_tokens = _tokenize(reference_text)
    cand_tokens = _tokenize(candidate_text)
    if not ref_tokens or not cand_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    overlap = sum(min(ref_counts[token], cand_counts[token]) for token in ref_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def _exact_match(reference_text: str, candidate_text: str) -> float:
    normalized_reference = " ".join(_tokenize(reference_text))
    normalized_candidate = " ".join(_tokenize(candidate_text))
    if not normalized_reference and not normalized_candidate:
        return 1.0
    return 1.0 if normalized_reference == normalized_candidate else 0.0


def _bleu_score(reference_text: str, candidate_text: str, weights: tuple[float, ...]) -> float:
    ref_tokens = _tokenize(reference_text)
    cand_tokens = _tokenize(candidate_text)
    if not ref_tokens or not cand_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return float(
        sentence_bleu(
            [ref_tokens],
            cand_tokens,
            weights=weights,
            smoothing_function=smoothing,
        )
    )


async def run_traditional_benchmark_evaluation() -> dict:
    benchmark_name, items = _load_benchmark()
    top_k = max(int(settings.TRADITIONAL_EVAL_TOP_K), 1)

    hit_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    ap_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []

    rouge1_scores: list[float] = []
    rouge2_scores: list[float] = []
    rougeL_scores: list[float] = []
    bleu1_scores: list[float] = []
    bleu4_scores: list[float] = []
    token_f1_scores: list[float] = []
    exact_match_scores: list[float] = []

    candidate_answers: list[str] = []
    reference_answers: list[str] = []
    per_query: list[dict] = []
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    for item in items:
        retrieved_docs, low_confidence, diagnostics = retrieve_local(item.query, top_k=top_k)
        retrieval_metrics = _compute_retrieval_metrics(
            retrieved_docs,
            relevant_document_ids=item.relevant_document_ids,
            k=top_k,
        )

        hit_scores.append(float(retrieval_metrics["hit_at_k"]))
        precision_scores.append(float(retrieval_metrics["precision_at_k"]))
        recall_scores.append(float(retrieval_metrics["recall_at_k"]))
        ap_scores.append(float(retrieval_metrics["average_precision_at_k"]))
        mrr_scores.append(float(retrieval_metrics["mrr_at_k"]))
        ndcg_scores.append(float(retrieval_metrics["ndcg_at_k"]))

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

        candidate_payload = _insight_to_payload(insight)
        candidate_answer = _normalize_output_payload(candidate_payload)
        if not candidate_answer:
            candidate_answer = "No generated answer text."

        reference_answer, reference_format = _reference_to_text(item)
        if not reference_answer:
            reference_answer = "No reference answer text."

        candidate_answers.append(candidate_answer)
        reference_answers.append(reference_answer)

        rouge_values = rouge.score(
            reference_answer,
            candidate_answer,
        )
        rouge1_value = float(rouge_values["rouge1"].fmeasure)
        rouge2_value = float(rouge_values["rouge2"].fmeasure)
        rougeL_value = float(rouge_values["rougeL"].fmeasure)

        bleu1_value = _bleu_score(reference_answer, candidate_answer, weights=(1.0,))
        bleu4_value = _bleu_score(reference_answer, candidate_answer, weights=(0.25, 0.25, 0.25, 0.25))
        token_f1_value = _token_f1(reference_answer, candidate_answer)
        exact_match_value = _exact_match(reference_answer, candidate_answer)

        rouge1_scores.append(rouge1_value)
        rouge2_scores.append(rouge2_value)
        rougeL_scores.append(rougeL_value)
        bleu1_scores.append(bleu1_value)
        bleu4_scores.append(bleu4_value)
        token_f1_scores.append(token_f1_value)
        exact_match_scores.append(exact_match_value)

        per_query.append(
            {
                "query": item.query,
                "relevant_document_ids": item.relevant_document_ids,
                "reference_format": reference_format,
                "retrieval_hit": bool(retrieval_metrics["hit_at_k"] > 0.0),
                "target_rank": retrieval_metrics["first_hit_rank"],
                "matched_ranks": retrieval_metrics["matched_ranks"],
                "precision_at_k": retrieval_metrics["precision_at_k"],
                "recall_at_k": retrieval_metrics["recall_at_k"],
                "average_precision_at_k": retrieval_metrics["average_precision_at_k"],
                "mrr_at_k": retrieval_metrics["mrr_at_k"],
                "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
                "rouge_1_f1": rouge1_value,
                "rouge_2_f1": rouge2_value,
                "rouge_l_f1": rougeL_value,
                "bleu_1": bleu1_value,
                "bleu_4": bleu4_value,
                "token_f1": token_f1_value,
                "exact_match": exact_match_value,
                "retrieval_diagnostics": diagnostics,
            }
        )

    try:
        _, _, bert_f1 = bert_score(
            candidate_answers,
            reference_answers,
            lang="en",
            model_type="distilbert-base-uncased",
            device="cpu",
            verbose=False,
        )
        bert_scores = [float(value) for value in bert_f1.tolist()]
    except Exception as exc:
        logger.warning("Traditional evaluation BERTScore failed; defaulting to 0.0: %s", exc)
        bert_scores = [0.0 for _ in candidate_answers]

    for idx, value in enumerate(bert_scores):
        per_query[idx]["bertscore_f1"] = value

    result = {
        "benchmark_name": benchmark_name,
        "k": top_k,
        "query_count": len(items),
        "hit_rate_at_k": _safe_mean(hit_scores),
        "precision_at_k": _safe_mean(precision_scores),
        "recall_at_k": _safe_mean(recall_scores),
        "map_at_k": _safe_mean(ap_scores),
        "mrr_at_k": _safe_mean(mrr_scores),
        "ndcg_at_k": _safe_mean(ndcg_scores),
        "rouge_1_f1": _safe_mean(rouge1_scores),
        "rouge_2_f1": _safe_mean(rouge2_scores),
        "rouge_l_f1": _safe_mean(rougeL_scores),
        "bleu_1": _safe_mean(bleu1_scores),
        "bleu_4": _safe_mean(bleu4_scores),
        "token_f1": _safe_mean(token_f1_scores),
        "exact_match_rate": _safe_mean(exact_match_scores),
        "bertscore_f1": _safe_mean(bert_scores),
        "per_query": per_query,
    }

    results_file = _write_results_file(result)
    if results_file:
        result["results_file"] = results_file

    return result


if __name__ == "__main__":
    print("Running traditional evaluation...\n")

    result = asyncio.run(run_traditional_benchmark_evaluation())

    print("\nFinal Results:\n")
    print(result)