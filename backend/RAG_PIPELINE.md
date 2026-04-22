# RAG Pipeline and Metric Calculations

This document describes how the RAG pipeline is implemented in this project, what each component does, and why it matters at each stage.

The same core pipeline powers:

- `/analyze` for the initial brand-decision response
- `/sessions/{id}/chat` for follow-up questions inside an existing session

The follow-up route reuses the same retrieval and generation stack, but builds a richer retrieval query from session history before calling the shared RAG components.

## 1. Pipeline Overview

High-level flow:

1. Documents are loaded, cleaned, chunked hierarchically, embedded, and indexed.
2. A user query arrives.
3. Local retrieval runs over child chunks in Chroma.
4. Query expansion generates multiple query variants when enabled.
5. Results from multiple variants are fused with Reciprocal Rank Fusion (RRF).
6. CRAG-style relevance grading filters noisy hits.
7. If retrieval is still weak, a corrective retrieval pass expands the search.
8. Parent reconstruction adds broader parent chunks for the strongest child hits.
9. If local retrieval remains weak, optional dynamic web fallback is fetched and filtered.
10. Local and dynamic context are merged and clipped to a token budget.
11. The final answer is generated from the merged context.
12. A grounding pass checks whether the answer is supported by the evidence.
13. Tier-1 metrics are logged immediately, and sampled RAGAS evaluation runs in the background.

## 2. Ingestion Pipeline

Relevant files:

- `ingestion/loader.py`
- `ingestion/preprocessor.py`
- `ingestion/chunker.py`
- `ingestion/embedder.py`
- `retrieval/parent_store.py`
- `retrieval/vector_store.py`
- `api/routes/ingest.py`
- `ingest_local.py`

### 2.1 Document loading

Supported sources:

- `pdf`
- `docx`
- `txt`
- `md`

Important implementation details:

- PDF pages are read with `pdfplumber`.
- Tables are extracted and converted to Markdown so table structure survives retrieval.
- Page markers such as `[Page X]` are preserved.
- File-path-derived metadata like `topic`, `subtopic`, `source_rel`, `source_name`, and `source_ext` are attached.
- DOCX tables are also converted to Markdown.

Why this matters:

- Preserving tables helps answer comparison-heavy or structured business questions.
- Page and section metadata make retrieved evidence easier to interpret and debug.
- Topic and subtopic metadata create better traceability for source provenance.

### 2.2 Preprocessing

The preprocessing step normalizes and cleans text before chunking:

- Unicode normalization with `NFKC`
- Removal of control characters
- Collapsing repeated blank lines
- Preservation of informative lines, page markers, and Markdown-style tables

Why this matters:

- Reduces embedding noise
- Avoids storing junk OCR fragments or formatting artifacts
- Preserves structure that later helps both retrieval and generation

### 2.3 Parent-child hierarchical chunking

This project uses hierarchical chunking instead of flat chunking.

Implementation:

- Parent chunks are created with `PARENT_CHUNK_SIZE` and `PARENT_CHUNK_OVERLAP`
- Child chunks are created inside each parent with `CHILD_CHUNK_SIZE` and `CHILD_CHUNK_OVERLAP`
- Each child stores links to:
  - `parent_id`
  - `parent_index`
  - `child_index`
  - `section`
  - `pages`
  - `has_table`

Why parent-child chunking is important:

- Child chunks give precise retrieval anchors.
- Parent chunks restore surrounding business context at generation time.
- This avoids a common failure mode where very small chunks retrieve well but are too narrow to answer robustly.
- It improves faithfulness by letting the model see both the exact supporting snippet and the larger narrative context.

### 2.4 Storage design

The storage is split by role:

- Child chunks are embedded and stored in Chroma
- Parent chunks are stored in a lightweight JSON parent store

Implementation details:

- Embeddings use `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings are normalized
- Chroma collection uses cosine distance
- Parent chunk lookup is done through `retrieval/parent_store.py`

Why this matters:

- Vector search stays lightweight by indexing smaller child chunks only.
- Parent reconstruction remains cheap and deterministic.
- This gives a good balance of retrieval precision and answer context richness.

## 3. Retrieval Pipeline

Relevant files:

- `retrieval/retriever.py`
- `retrieval/fallback/context_builder.py`
- `retrieval/fallback/dynamic_retriever.py`
- `retrieval/fallback/news_filter.py`
- `api/routes/analyze.py`
- `api/routes/sessions.py`

### 3.1 Query expansion

When `ENABLE_QUERY_EXPANSION=true`, the system asks an LLM retrieval helper to produce alternate phrasings of the user query.

If the LLM helper fails:

- the code falls back to deterministic lexical query variants
- the system can also enter a temporary retrieval-helper cooldown after rate limiting

Why this matters:

- Improves recall when user wording does not match source wording closely
- Helps surface semantically related case studies
- Makes the system less dependent on one exact phrasing

### 3.2 Multi-query retrieval

For each query variant:

- the query is embedded
- Chroma returns top results
- cosine distance is converted to similarity

Formula used:

`similarity_score = 1.0 - distance`

Why this matters:

- Multiple variants increase coverage without changing the underlying index
- Similarity scores become directly usable in later confidence and logging logic

### 3.3 Reciprocal Rank Fusion (RRF)

Results from all query variants are merged with Reciprocal Rank Fusion.

Formula:

`fused_score(doc) = sum(1 / (RRF_K + rank_i))`

Why RRF is important here:

- It combines evidence from multiple query formulations without requiring score calibration across different retrieval runs.
- It rewards documents that consistently appear near the top across variants.
- It is the main rank aggregation step in the retrieval stage.

### 3.4 CRAG-style relevance grading

After fusion, the project applies a CRAG-inspired relevance grading step.

Implementation:

- `retrieval/retriever.py::_grade_relevance`
- An LLM is asked to return `relevant_indices`
- Only clearly relevant documents are kept
- If too few documents survive, the system falls back to the unfiltered set

Why this matters:

- Vector retrieval can return semantically similar but strategically irrelevant chunks.
- Relevance grading acts as a precision-improving filter after approximate nearest-neighbor retrieval.
- It is especially useful when query expansion broadens recall and introduces noisy candidates.

### 3.5 Corrective retrieval pass

If the graded result set is still too small:

- the system builds corrective query variants
- runs another retrieval pass
- fuses widened results
- grades again

Why this matters:

- This is the "corrective" part of the CRAG behavior.
- It gives the system a second chance before declaring local retrieval weak.
- It helps recover from over-filtering or weak initial reformulations.

### 3.6 Parent reconstruction

After child retrieval and grading:

- the top child hits are examined
- corresponding parents are fetched from the parent store
- parent chunks are appended into the final context set

Why this matters:

- Retrieved child chunks are precise but narrow.
- Parent reconstruction restores broader context for the final answer.
- This improves generation quality without sacrificing retrieval precision.

### 3.7 Confidence gate

Low confidence is defined using the max similarity among final local docs:

`low_confidence = (max_similarity < CONFIDENCE_THRESHOLD)`

Why this matters:

- The system avoids over-trusting weak local retrieval.
- It decides whether dynamic web fallback should be attempted.
- It also feeds the abstention logic used later in generation.

### 3.8 Dynamic fallback retrieval

When local retrieval is weak and `use_fallback=true`:

- the system fetches news articles
- filters them through several quality gates
- chunks the surviving articles into dynamic chunks

Filtering steps:

1. English-language filter
2. Minimum length filter
3. Gibberish filter
4. Relevance scoring against:
   - the query embedding
   - a centroid embedding from the retrieved local docs
5. Deduplication by title embedding similarity
6. Recency filter

Why this matters:

- Adds coverage when the local corpus is thin
- Keeps fallback evidence constrained and quality-controlled
- Uses the local corpus centroid to prefer articles aligned with the current knowledge base, not just the query wording

## 4. What This Project Uses Instead of a Classic Reranker

There is no dedicated cross-encoder reranker in this codebase.

Instead, reranking behavior is approximated through a combination of:

1. Multi-query retrieval plus RRF
2. CRAG-style LLM relevance grading
3. Corrective retrieval when the graded set is too thin
4. Parent reconstruction after precise child retrieval

So the closest substitute for reranking is:

- `RRF` as the rank aggregation mechanism
- `LLM relevance grading` as the precision filter that sits where a reranker often would

This is not the same as a classic cross-encoder reranker that rescored every candidate with a learned relevance model, but functionally it serves a similar role:

- remove weak hits
- improve ordering quality indirectly
- increase precision before generation

## 5. Context Assembly

Relevant file:

- `retrieval/fallback/context_builder.py`

The final context bundle merges:

- local docs
- dynamic fallback docs

The bundle tracks:

- `total_tokens`
- `local_tokens`
- `dynamic_tokens`
- `used_fallback`

The merged context is clipped using `MAX_CONTEXT_TOKENS`.

Important implementation detail:

- dynamic chunks are dropped first
- local chunks are dropped second

Why this matters:

- Keeps the prompt bounded for latency and reliability
- Preserves local case-study evidence preferentially over dynamic fallback evidence
- Prevents context bloat from overwhelming generation

## 6. Generation Pipeline

Relevant files:

- `reasoning/prompts.py`
- `reasoning/llm_client.py`
- `reasoning/schema.py`

### 6.1 Coverage-aware abstention before answer generation

Before normal generation, the system checks whether there is enough evidence to proceed.

Signals used:

- `doc_count`
- `strong_doc_count`
- `max_similarity`
- `dynamic_tokens`
- `CONFIDENCE_THRESHOLD`
- `CRAG_MIN_RELEVANT_DOCS`

If coverage is too weak:

- the system does not generate a normal strategic answer
- it returns a controlled abstention response

Why this matters:

- Prevents low-evidence hallucinations
- Makes weak retrieval visible as an explicit product behavior rather than a silent quality failure

### 6.2 Structured answer generation

For grounded cases:

- the final prompt includes the schema
- retrieved context documents are serialized with metadata
- the system asks for JSON only

The schema includes strategic fields such as:

- `brand_diagnosis`
- `market_insight`
- `suggested_positioning`
- `risks`
- `opportunities`
- `final_positioning`
- `target_audience`
- `chosen_strategy`
- `rejected_directions`
- `trade_offs`
- `actions`

Why this matters:

- Gives stable downstream structure for UI rendering and execution workflows
- Separates suggestions, user decisions, and execution tasks

### 6.3 Model fallback

The generation path tries:

1. `GEMMA_MODEL_NAME`
2. `LLM_FALLBACK_MODEL_NAME`

It also handles JSON-mode incompatibility by retrying without `response_mime_type` when needed.

Why this matters:

- Improves robustness when a provider/model rejects a request or JSON mode
- Keeps the pipeline operational without needing manual switching

### 6.4 Self-RAG style grounding check

After generation, the system runs a separate grounding grader that returns:

- `grounded`
- `partial`
- `not_grounded`
- `unknown`

If the answer is `not_grounded`:

1. the system attempts one repair pass
2. if repair still fails, it falls back to abstention

Why this matters:

- This is the main Self-RAG-like behavior in the codebase
- It separates retrieval/generation from post-generation faithfulness checking
- It reduces hallucination risk even when the first answer looks fluent

## 7. Follow-up Chat Pipeline

Relevant route:

- `api/routes/sessions.py`

Follow-up chat reuses the same retrieval and generation stack, but its retrieval query is richer than the initial `/analyze` query.

It includes:

- the original decision question
- recent follow-up questions
- the current follow-up question

It also builds a separate reasoning prompt that includes:

- the original decision question
- a baseline structured snapshot from the original session
- recent conversation history
- the current follow-up question

Why this matters:

- Improves continuity across follow-up turns
- Lets retrieval stay anchored to the session topic instead of only the latest utterance
- Preserves the same CRAG / parent-child / grounding behavior for chat

## 8. Metrics and Diagnostics

Relevant files:

- `evaluation/lightweight_metrics.py`
- `api/routes/analyze.py`
- `api/routes/sessions.py`
- `evaluation/ragas_evaluator.py`

### 8.1 Tier-1 metrics

These are computed immediately from the final context bundle and generation pass.

Main metrics:

- `avg_similarity_score`
- `docs_above_threshold`
- `total_docs_retrieved`
- `context_total_tokens`
- `context_local_ratio`
- `context_dynamic_ratio`
- `used_fallback`
- `articles_fetched`
- `articles_surviving`
- `llm_latency_ms`
- `llm_retry_count`

Meaning:

- `avg_similarity_score`: mean similarity of final retrieved docs
- `docs_above_threshold`: count of docs with similarity `> CONFIDENCE_THRESHOLD`
- `total_docs_retrieved`: number of docs after retrieval, grading, parent reconstruction, and context clipping
- `context_total_tokens`: merged prompt-context token count after clipping
- `context_local_ratio`: share of local evidence in final context
- `context_dynamic_ratio`: share of dynamic fallback evidence in final context
- `llm_latency_ms`: end-to-end answer generation latency
- `llm_retry_count`: retry count inside the generation loop

### 8.2 Retrieval diagnostics

The retrieval layer also exposes more detailed diagnostics:

- `query_variants`
- `corrective_query_variants`
- `corrective_pass_used`
- `retrieved_before_crag`
- `retrieved_after_initial_crag`
- `retrieved_after_crag`
- `child_context_docs`
- `parent_context_docs`

Why these matter:

- They explain whether weak quality comes from recall, precision filtering, or context assembly.
- They make it easier to see if the pipeline is being saved by query expansion, corrective retrieval, or parent reconstruction.

### 8.3 RAGAS evaluation

RAGAS is used as a background monitoring layer for sampled requests.

Tracked dimensions:

- `context_precision`
- `context_recall`
- `faithfulness`
- `answer_relevancy`

Why this matters:

- Gives a second layer of developer-facing quality visibility
- Separates retrieval quality from answer quality
- Does not block the user-facing response path

## 9. Thresholds and Controls

Configured in `.env` / `config.py`.

Important controls:

- `CONFIDENCE_THRESHOLD`
- `TOP_K_DEFAULT`
- `MAX_CONTEXT_TOKENS`
- `LLM_MAX_RETRIES`
- `MODEL_REQUEST_TIMEOUT_SECONDS`
- `ENABLE_QUERY_EXPANSION`
- `ENABLE_RELEVANCE_GRADING`
- `MULTI_QUERY_COUNT`
- `RRF_K`
- `CRAG_MIN_RELEVANT_DOCS`
- `ENABLE_GROUNDING_CHECK`
- `PARENT_CHUNK_SIZE`
- `PARENT_CHUNK_OVERLAP`
- `CHILD_CHUNK_SIZE`
- `CHILD_CHUNK_OVERLAP`
- `MAX_PARENT_CONTEXT_CHUNKS`
- `NEWS_RELEVANCE_THRESHOLD`
- `NEWS_MAX_AGE_DAYS`

These settings control both quality and system behavior.

Examples:

- Increase `TOP_K_DEFAULT` to retrieve more local candidates
- Increase `MAX_CONTEXT_TOKENS` to allow more evidence into the prompt
- Increase `CRAG_MIN_RELEVANT_DOCS` to demand more retained evidence before trusting filtered docs
- Increase `MULTI_QUERY_COUNT` to broaden recall through more query variants
- Increase `MAX_PARENT_CONTEXT_CHUNKS` to expose more broad context around child hits

## 10. Why Each Component Matters by Stage

### Ingestion

- Loader preserves structure and metadata
- Preprocessor reduces noise
- Parent-child chunking balances precision and context
- Embedding plus vector indexing enables fast semantic lookup

### Retrieval

- Query expansion broadens recall
- RRF combines multiple ranked views
- CRAG relevance grading improves precision
- Corrective retrieval recovers from weak initial hits
- Parent reconstruction restores context around precise matches

### Context assembly

- Dynamic fallback improves coverage when the local corpus is weak
- Token clipping keeps prompts bounded and stable
- Local-first retention keeps the knowledge base primary

### Generation

- Structured prompting standardizes outputs
- Model fallback improves runtime resilience
- Abstention prevents unsupported answers
- Self-RAG grounding check protects faithfulness after generation

### Evaluation

- Tier-1 metrics give immediate observability
- Retrieval diagnostics explain where failures originate
- RAGAS adds higher-level developer monitoring without blocking the user path

## 11. Summary

This project is not a plain embed-search-generate stack. It combines:

- hierarchical parent-child chunking
- multi-query retrieval
- RRF fusion
- CRAG-style relevance grading
- corrective retrieval
- optional dynamic web fallback
- context token budgeting
- Self-RAG-like grounding and repair

The most important architectural idea is that retrieval precision and answer faithfulness are enforced in multiple stages:

- before generation through CRAG, confidence gating, and context clipping
- after generation through grounding checks and repair

That layered design is what gives the current pipeline its reliability characteristics.
