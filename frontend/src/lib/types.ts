export interface ActionItem {
  type: "task";
  title: string;
  description: string;
  priority: "low" | "medium" | "high";
  decision_type?:
    | "positioning"
    | "differentiation"
    | "messaging"
    | "trust"
    | "audience"
    | "pricing"
    | "narrative"
    | "other";
  impact?: "low" | "medium" | "high";
}

export interface SessionConversationMessage {
  role: "user" | "assistant";
  content: string;
}

export interface RetrievalDiagnostics {
  query_intent?: "factual" | "analytical" | "inferential" | string;
  entity_hints?: string[];
  query_variants?: string[];
  query_domain?: string | null;
  query_expansion_mode?: "llm_assisted" | "deterministic_fallback" | "disabled" | string;
  relevance_grading_mode?: "llm_assisted" | "heuristic_fallback" | "disabled" | string;
  retrieved_before_crag?: number;
  retrieved_after_initial_crag?: number;
  retrieved_after_crag?: number;
  retrieved_after_rerank?: number;
  retrieved_after_diversification?: number;
  corrective_query_variants?: string[];
  corrective_pass_used?: boolean;
  guardrail_rescues?: number;
  entity_docs_in_fused?: number;
  entity_docs_after_crag?: number;
  entity_docs_after_rerank?: number;
  entity_coverage_required?: boolean;
  entity_coverage_satisfied?: boolean;
  domain_coverage_required?: boolean;
  domain_coverage_satisfied?: boolean;
  domain_aligned_docs_after_rerank?: number;
  domain_mismatched_docs_after_rerank?: number;
  domain_guardrail_triggered?: boolean;
  answerability_score?: number;
  answerability_satisfied?: boolean;
  answerability_threshold?: number;
  fallback_requested?: boolean;
  fallback_attempted?: boolean;
  fallback_used?: boolean;
  news_api_configured?: boolean;
  gnews_api_configured?: boolean;
  fallback_articles_fetched?: number;
  fallback_articles_surviving?: number;
  source_diversity_count_after_rerank?: number;
  dominant_source_after_rerank?: string;
  dominant_source_share_after_rerank?: number;
  child_context_docs?: number;
  parent_context_docs?: number;
  hybrid_retrieval_enabled?: boolean;
  dense_candidates?: number;
  lexical_candidates?: number;
  max_retrieval_score?: number;
  retrieved_sources?: string[];
  retrieval_query_mode?: "current_only" | "contextual" | string;
  coverage_metrics?: {
    doc_count: number;
    strong_doc_count: number;
    max_similarity: number;
    avg_similarity: number;
    local_tokens: number;
    dynamic_tokens: number;
    total_tokens: number;
    confidence_threshold: number;
  };
  abstained?: boolean;
  abstain_reason?: string | null;
  quota_safe_mode_active?: boolean;
  quota_safe_mode_scope?: "retrieval" | "reasoning" | "both" | string | null;
  quota_safe_mode_message?: string | null;
  quota_hit_in_retrieval_helpers?: boolean;
  quota_hit_in_reasoning?: boolean;
  critical_functions_temporarily_degraded?: string[];
  retrieval_quota_retry_after_seconds?: number;
}

export interface SessionChatTurn {
  id: string;
  user_message: string;
  assistant_message: string;
  insights: InsightOutput;
  grounding_status?:
    | "grounded"
    | "partial"
    | "not_grounded"
    | "unknown"
    | "not_requested"
    | "bypassed"
    | "insufficient_context";
  faithfulness_corrected?: boolean;
  used_fallback: boolean;
  retrieval_diagnostics?: {
    query_intent?: "factual" | "analytical" | "inferential" | string;
    entity_hints?: string[];
    query_variants?: string[];
    retrieved_before_crag?: number;
    retrieved_after_crag?: number;
    retrieved_after_rerank?: number;
    retrieved_after_diversification?: number;
    parent_context_docs?: number;
    hybrid_retrieval_enabled?: boolean;
    dense_candidates?: number;
    lexical_candidates?: number;
    max_retrieval_score?: number;
    retrieved_sources?: string[];
    retrieval_query_mode?: "current_only" | "contextual" | string;
  } | null;
  created_at: string | null;
}

export interface InsightOutput {
  abstention_message?: string | null;
  brand_diagnosis?: string | null;
  market_insight?: string | null;
  suggested_positioning?: string[] | null;
  risks?: string[] | null;
  opportunities?: string[] | null;
  final_positioning?: string | null;
  target_audience?: string | null;
  chosen_strategy?: string | null;
  rejected_directions?: string[] | null;
  trade_offs?: string[] | null;

  // Legacy compatibility
  idea_summary?: string | null;
  recommendations?: string[] | null;

  actions: ActionItem[];
  confidence_score: number;
  notion_page_content?: string | null;
  database_metadata?: {
    name?: string | null;
    brand_positioning?: string | null;
    brand_risk_level?: "Low" | "Medium" | "High" | null;

    // Legacy compatibility
    idea_description?: string | null;
    risk_level?: "Low" | "Medium" | "High" | null;

    confidence_score?: number | null;
    tags?: string[] | null;
  } | null;
}

export interface Tier1Metrics {
  avg_similarity_score: number;
  docs_above_threshold: number;
  total_docs_retrieved: number;
  context_total_tokens: number;
  context_local_ratio: number;
  context_dynamic_ratio: number;
  used_fallback: boolean;
  articles_fetched: number;
  articles_surviving: number;
  llm_latency_ms: number;
  llm_retry_count: number;
}

export interface AnalyzeRequest {
  idea: string;
  top_k?: number;
  use_fallback?: boolean;
  run_evaluation?: boolean;
}

export interface SessionChatRequestPayload {
  message: string;
  top_k?: number;
  use_fallback?: boolean;
  run_evaluation?: boolean;
}

export interface AnalyzeResponse {
  session_id: string;
  insights: InsightOutput;
  grounding_status?:
    | "grounded"
    | "partial"
    | "not_grounded"
    | "unknown"
    | "not_requested"
    | "insufficient_context";
  faithfulness_corrected?: boolean;
  used_fallback: boolean;
  retrieved_sources: string[];
  retrieval_diagnostics?: {
    query_intent?: "factual" | "analytical" | "inferential" | string;
    entity_hints?: string[];
    query_variants: string[];
    retrieved_before_crag: number;
    retrieved_after_crag: number;
    retrieved_after_rerank?: number;
    retrieved_after_diversification?: number;
    parent_context_docs?: number;
    hybrid_retrieval_enabled?: boolean;
    dense_candidates?: number;
    lexical_candidates?: number;
    max_retrieval_score?: number;
    retrieved_sources?: string[];
    retrieval_query_mode?: "current_only" | "contextual" | string;
    coverage_metrics?: {
      doc_count: number;
      strong_doc_count: number;
      max_similarity: number;
      avg_similarity: number;
      local_tokens: number;
      dynamic_tokens: number;
      total_tokens: number;
      confidence_threshold: number;
    };
    abstained?: boolean;
    abstain_reason?: string | null;
  };
  evaluation_status: "pending" | "skipped" | "not_requested";
  tier1_metrics: Tier1Metrics;
}

export interface ActionResult {
  action_title: string;
  target_provider: string;
  external_id: string | null;
  status: "executed" | "failed";
  error_message: string | null;
  insight_page_url?: string | null;
}

export interface DatabaseMetadataOverride {
  name?: string | null;
  brand_positioning?: string | null;
  brand_risk_level?: "Low" | "Medium" | "High" | null;
  confidence_score?: number | null;
  tags?: string[] | null;
}

export interface ExecuteRequest {
  session_id: string;
  target: "notion" | "jira" | "both";
  selected_action_indices?: number[];
  notion_page_content_override?: string;
  database_metadata_override?: DatabaseMetadataOverride;
}

export type ExecuteResponse = ActionResult[];

export interface UserIntegrationOut {
  provider: "notion" | "jira" | string;
  workspace_id: string | null;
  database_id: string | null;
  created_at: string;
}

export interface IntegrationConnectRequest {
  provider: "notion" | "jira";
  access_token: string;
  workspace_id?: string | null;
  database_id?: string | null;
}

export interface Session {
  id: string;
  created_at: string;
  idea_text: string;
  confidence_score: number;
  used_fallback: boolean;
  actions_taken: number;
  tier1_metrics?: Tier1Metrics | null;
  ragas?: {
    status: "pending" | "skipped" | "completed" | "failed" | "not_requested" | string;
    context_precision: number | null;
    context_recall: number | null;
    faithfulness: number | null;
    answer_relevance: number | null;
    evaluation_mode?: "ragas";
  } | null;
}

export interface EvaluationLog {
  avg_similarity_score: number;
  min_similarity_score: number;
  max_similarity_score: number;
  docs_above_threshold: number;
  total_docs_retrieved: number;
  context_total_tokens: number;
  context_local_ratio: number;
  context_dynamic_ratio: number;
  used_fallback: boolean;
  articles_fetched: number;
  articles_surviving: number;
  avg_fallback_relevance: number | null;
  llm_latency_ms: number;
  llm_retry_count: number;
  llm_validation_passed: boolean;
  context_precision: number | null;
  context_recall: number | null;
  faithfulness: number | null;
  answer_relevance: number | null;
  ragas_eval_status: "pending" | "skipped" | "completed" | "failed" | "not_requested" | string;
  evaluation_mode?: "ragas";
  query: string;
  retrieved_docs: unknown[];
  generated_output: string;
  created_at: string;
}

export interface SessionActionLog {
  id: string;
  action_type: string;
  title: string;
  description: string;
  priority: string;
  target_provider: string;
  status: string;
  external_id: string | null;
  error_message: string | null;
  created_at: string;
}

export interface SessionDetail {
  id: string;
  idea_text: string;
  raw_output: InsightOutput;
  confidence_score: number;
  used_fallback: boolean;
  created_at: string;
  conversation?: SessionConversationMessage[];
  chat_turns?: SessionChatTurn[];
  tier1_metrics?: Tier1Metrics | null;
  retrieval_diagnostics?: RetrievalDiagnostics;
  retrieved_sources?: string[];
  evaluation_log: EvaluationLog | null;
  action_logs: SessionActionLog[];
}

export interface SessionChatResponse {
  session_id: string;
  insights: InsightOutput;
  grounding_status?:
    | "grounded"
    | "partial"
    | "not_grounded"
    | "unknown"
    | "not_requested"
    | "bypassed"
    | "insufficient_context";
  faithfulness_corrected?: boolean;
  used_fallback: boolean;
  chat_turn?: SessionChatTurn;
  evaluation_log?: EvaluationLog | null;
  action_logs?: SessionActionLog[];
  retrieval_diagnostics?: RetrievalDiagnostics;
  conversation: SessionConversationMessage[];
}
