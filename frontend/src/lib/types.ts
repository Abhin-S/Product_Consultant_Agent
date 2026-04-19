export interface ActionItem {
  type: "task";
  title: string;
  description: string;
  priority: "low" | "medium" | "high";
}

export interface InsightOutput {
  idea_summary: string;
  risks: string[];
  opportunities: string[];
  recommendations: string[];
  actions: ActionItem[];
  confidence_score: number;
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

export interface AnalyzeResponse {
  session_id: string;
  insights: InsightOutput;
  grounding_status?: "grounded" | "partial" | "not_grounded" | "unknown" | "not_requested";
  faithfulness_corrected?: boolean;
  used_fallback: boolean;
  retrieved_sources: string[];
  retrieval_diagnostics?: {
    query_variants: string[];
    retrieved_before_crag: number;
    retrieved_after_crag: number;
    parent_context_docs?: number;
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
}

export interface ExecuteRequest {
  session_id: string;
  target: "notion" | "jira" | "both";
  selected_action_indices?: number[];
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
    status: "pending" | "skipped" | "completed" | "failed" | string;
    context_precision: number | null;
    context_recall: number | null;
    faithfulness: number | null;
    answer_relevance: number | null;
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
  ragas_eval_status: "pending" | "skipped" | "completed" | "failed" | string;
  query: string;
  retrieved_docs: unknown[];
  generated_output: string;
  created_at: string;
}

export interface SessionDetail {
  id: string;
  idea_text: string;
  raw_output: InsightOutput;
  confidence_score: number;
  used_fallback: boolean;
  created_at: string;
  evaluation_log: EvaluationLog | null;
  action_logs: {
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
  }[];
}