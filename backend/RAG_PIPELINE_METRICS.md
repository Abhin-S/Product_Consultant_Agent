# RAG Pipeline and Metric Calculations

This document explains how retrieval and evaluation work in this project, how each metric is computed, and what thresholds control behavior.

## 1) End-to-end pipeline

1. User query arrives at `/analyze`.
2. Local retrieval runs in `retrieval/retriever.py`.
3. Optional query expansion generates alternate queries (LLM-assisted).
4. For each query variant, Chroma retrieval returns chunk distances.
5. Distance is converted to similarity as:

   `similarity_score = 1.0 - distance`

6. Ranked results from all variants are merged with Reciprocal Rank Fusion (RRF):

   `fused_score(doc) = sum(1 / (RRF_K + rank_i))`

7. Optional CRAG relevance grading keeps only LLM-judged relevant chunks.
8. Parent reconstruction replaces child hits with parent chunks (when available).
9. Low-confidence check runs using `CONFIDENCE_THRESHOLD`.
10. If low confidence and fallback is enabled, dynamic/news fallback runs.
11. Local + dynamic chunks are merged and clipped to `MAX_CONTEXT_TOKENS`.
12. Generation runs in `reasoning/llm_client.py`.
13. Grounding check runs (`grounded|partial|not_grounded|unknown`) and can trigger one repair attempt.
14. If context coverage is insufficient, the system abstains with an explicit "cannot answer reliably" response.
15. Tier-1 metrics are stored immediately. RAGAS evaluation is launched in background when sampled.

## 2) Tier-1 metrics shown in UI and logs

These are computed in `evaluation/lightweight_metrics.py` and `api/routes/analyze.py`.

### Avg similarity

Mean of final retrieved docs used in context:

`avg_similarity = sum(similarity_score_i) / N`

If no docs: `0.0`.

### Docs above threshold

Count of docs where similarity is strictly greater than threshold:

`docs_above_threshold = count(similarity_score > CONFIDENCE_THRESHOLD)`

Note this uses `>` (not `>=`).

### Total docs retrieved

`total_docs_retrieved = len(context_bundle.docs)`

This is after fusion/grading/parent reconstruction and after token-capping.

### Context tokens

- `context_total_tokens`: total tokens in merged context
- `context_local_ratio`: local_tokens / total_tokens
- `context_dynamic_ratio`: dynamic_tokens / total_tokens

### LLM latency / retry

- `llm_latency_ms`: end-to-end generation latency in milliseconds
- `llm_retry_count`: retry attempts used by generation call

### Query variants

`len(query_variants)` generated for retrieval. If query expansion fails, this may be `1`.

### CRAG docs kept

`retrieved_after_crag / retrieved_before_crag`

### Parent context docs

How many parent chunks are present after parent reconstruction.

## 3) Thresholds and controls

Configured in `.env` / `config.py`.

### `CONFIDENCE_THRESHOLD` (default 0.45)

Used in multiple places:

1. `low_confidence = (max_similarity < CONFIDENCE_THRESHOLD)`
2. `docs_above_threshold` metric
3. `strong_doc_count` for abstention coverage checks

Role: retrieval quality gate before trusting local context.

### `NEWS_RELEVANCE_THRESHOLD` (default 0.35)

Used in fallback filtering:

`relevance_score = 0.6 * cosine(article, query) + 0.4 * cosine(article, kb_centroid)`

Article survives only if score >= threshold.

### `CRAG_MIN_RELEVANT_DOCS` (default 2)

If CRAG keeps fewer than this minimum, system falls back to unfiltered docs (safety against over-filtering).

### `MAX_CONTEXT_TOKENS` (default 3000)

Hard cap for merged context. Dynamic chunks are dropped first, then local chunks.

### `RRF_K` (default 60)

Smoothing constant in reciprocal rank fusion. Higher values reduce dominance of top ranks.

## 4) Retrieval confidence: what it is in this codebase

In generation output:

`insight.confidence_score = mean(similarity_score of context_bundle.docs)`

This is a retrieval-derived confidence proxy in `[0,1]`.

If the system abstains due to insufficient evidence:

`confidence_score = min(avg_similarity, 0.2)`

So abstained outputs are intentionally low-confidence.

## 5) Why you saw "1 doc above threshold" with avg similarity ~0.502

This is plausible and not automatically fake:

1. Query expansion failed in your logs due model issues, so retrieval ran with fewer variants.
2. With only one final retained doc, avg similarity equals that doc score (~0.502).
3. Since threshold is 0.45, that one doc is counted above threshold.

Given your question domain should be well-covered, the likely issue was query-expansion degradation (model fallback + JSON mode handling), not necessarily missing corpus content.

## 6) RAGAS metrics and where they apply

RAGAS runs in `evaluation/ragas_evaluator.py`.

### Retrieval-stage quality

- `context_precision`: how much provided context is relevant to answer.
- `context_recall`: how much needed evidence was captured by retrieved context.

### Generation-stage quality

- `faithfulness`: whether answer statements are supported by context.
- `answer_relevancy`: whether answer addresses the question.

## 7) Practical target ranges (heuristics)

These are practical engineering targets, not absolute rules:

### Retrieval targets

- `avg_similarity_score`: >= 0.50 good, 0.40-0.50 watch, < 0.40 weak
- `docs_above_threshold`: >= 2 preferred for strategy queries
- `context_precision`: >= 0.70 good, >= 0.80 strong
- `context_recall`: >= 0.60 good, >= 0.75 strong

### Generation targets

- `faithfulness`: >= 0.80 good, >= 0.90 strong
- `answer_relevancy`: >= 0.70 good, >= 0.85 strong

### Operational targets

- `llm_retry_count`: 0-1 normal, repeated spikes indicate provider instability
- `llm_latency_ms`: monitor trend by model and prompt size

## 8) Important caveat in current evaluation setup

Current RAGAS dataset uses:

- `ground_truth = generated_output`

This is acceptable for lightweight drift monitoring, but not ideal for true quality benchmarking. For stronger evaluation, use human-verified references or curated expected answers.

## 9) Recent reliability hardening implemented

1. Retrieval and reasoning now detect JSON-mode unsupported errors (for Gemma 3 variants) and retry without `response_mime_type`.
2. RAGAS now evaluates metrics fail-fast (sequential), so evaluation stops at first metric failure instead of running all metrics to completion.
3. RAGAS now tries fallback model candidates (`GEMMA_MODEL_NAME` then `LLM_FALLBACK_MODEL_NAME`) before marking evaluation as failed.
4. Added `RAGAS_MAX_OUTPUT_TOKENS` as an optional cap. Set it to `0` to remove explicit output-token capping and use provider defaults.
5. Added RAGAS payload shaping (`RAGAS_MAX_CONTEXT_DOCS`, `RAGAS_CONTEXT_DOC_CHAR_LIMIT`, `RAGAS_ANSWER_CHAR_LIMIT`) so evaluation prompts stay bounded and avoid `LLMDidNotFinishException` caused by oversized context/answer payloads.

## 10) Failed Feature Attempt: Intent-Aware Query Routing (Rolled Back)

### Goal of the attempt

We attempted to extend the assistant from a single diagnostic response mode into four routed response types:

1. `diagnosis`: brand strategy diagnosis and positioning decisions
2. `planning`: weekly/phase launch execution plans
3. `informational`: concept and framework explanations
4. `comparative`: side-by-side option analysis

Routing was designed to classify each user query first, then validate generation against a type-specific schema and prompt.

### What we implemented during the attempt

- Added classifier-based routing logic before generation.
- Added route map for the four query types.
- Added multiple output schemas and prompt variants for each type.
- Updated `/analyze` response shape to support type-dependent outputs.

### Why the attempt failed in this codebase

Observed runtime failures included repeated fallback to diagnostic mode and occasional `422` responses.
Representative logs:

- `query_classification_failed; defaulting to diagnosis: Invalid query_type returned: <type>`
- Routed generation executed as `query_type=diagnosis` even when planning intent was explicit.

Primary failure reasons:

1. **Classifier instability under strict JSON contract**
   - The model intermittently returned placeholder or malformed classification values (for example `<type>`), which failed validation and forced fallback.

2. **Increased failure surface area**
   - Multi-stage flow (classify -> route -> generate -> per-schema validate) created additional points of failure versus the original single-schema path.

3. **Operational reliability regression**
   - The complexity added retries, validation edge cases, and route mismatch behavior that reduced predictability for production usage.

### Final decision

This feature attempt was **rolled back**.
The assistant remains in **diagnostic-only mode** (single schema, single prompt family) for reliability and simpler observability.