"use client";

import { InsightOutput, Tier1Metrics } from "../../lib/types";

type Props = {
  insight: InsightOutput;
  usedFallback: boolean;
  evaluationStatus: "pending" | "skipped" | "not_requested";
  tier1Metrics: Tier1Metrics;
  groundingStatus?: "grounded" | "partial" | "not_grounded" | "unknown" | "not_requested";
  faithfulnessCorrected?: boolean;
  retrievalDiagnostics?: {
    query_variants: string[];
    retrieved_before_crag: number;
    retrieved_after_crag: number;
    parent_context_docs?: number;
  };
  onReview: () => void;
  onStartOver: () => void;
};

function statusLabel(status: "pending" | "skipped" | "not_requested") {
  if (status === "pending") return "Pending evaluation";
  if (status === "skipped") return "Evaluation skipped";
  return "Evaluation not requested";
}

export default function InsightDisplay({
  insight,
  usedFallback,
  evaluationStatus,
  tier1Metrics,
  groundingStatus,
  faithfulnessCorrected,
  retrievalDiagnostics,
  onReview,
  onStartOver
}: Props) {
  return (
    <section className="space-y-4">
      <article className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
        <h2 className="text-xl font-semibold">Idea Summary</h2>
        <p className="mt-2 rounded-xl bg-slate-50 p-4 text-slate-800">{insight.idea_summary}</p>

        <div className="mt-4">
          <p className="mb-1 text-sm font-medium">Retrieval Confidence</p>
          <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200">
            <div
              className="h-full bg-pine transition-all"
              style={{ width: `${Math.max(0, Math.min(1, insight.confidence_score)) * 100}%` }}
            />
          </div>
          <p className="mt-1 text-sm text-slate-600">{insight.confidence_score.toFixed(2)}</p>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
          {usedFallback && (
            <span className="rounded-full bg-amber-100 px-3 py-1 font-medium text-amber-900">
              Web fallback used
            </span>
          )}
          <span className="rounded-full bg-slate-100 px-3 py-1 font-medium text-slate-700">
            {statusLabel(evaluationStatus)}
          </span>
          {groundingStatus && (
            <span className="rounded-full bg-emerald-100 px-3 py-1 font-medium text-emerald-900">
              Grounding: {groundingStatus}
            </span>
          )}
          {faithfulnessCorrected && (
            <span className="rounded-full bg-sky-100 px-3 py-1 font-medium text-sky-900">
              Faithfulness pass applied
            </span>
          )}
        </div>
      </article>

      {insight.notion_page_content && insight.notion_page_content.trim().length > 0 && (
        <article className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
          <h3 className="text-lg font-semibold">Notion Page Content (Primary)</h3>
          <p className="mt-1 text-sm text-slate-600">
            This formatted draft is ready to paste into a Notion page body.
          </p>
          <pre className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-50 p-4 text-sm text-slate-800">
            {insight.notion_page_content}
          </pre>
        </article>
      )}

      {insight.database_metadata && (
        <article className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
          <h3 className="text-lg font-semibold">Database Metadata (Secondary)</h3>
          <div className="mt-3 grid gap-2 text-sm text-slate-700 md:grid-cols-2">
            <p>
              <span className="font-medium">Name:</span> {insight.database_metadata.name}
            </p>
            <p>
              <span className="font-medium">Risk Level:</span> {insight.database_metadata.risk_level}
            </p>
            <p className="md:col-span-2">
              <span className="font-medium">Idea Description:</span> {insight.database_metadata.idea_description}
            </p>
            <p>
              <span className="font-medium">Confidence Score:</span> {insight.database_metadata.confidence_score}
            </p>
            <div className="md:col-span-2">
              <p className="mb-1 font-medium">Tags:</p>
              <div className="flex flex-wrap gap-2">
                {insight.database_metadata.tags.length > 0 ? (
                  insight.database_metadata.tags.map((tag) => (
                    <span key={tag} className="rounded-full bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700">
                      {tag}
                    </span>
                  ))
                ) : (
                  <span className="text-slate-500">No tags</span>
                )}
              </div>
            </div>
          </div>
        </article>
      )}

      <div className="grid gap-4 md:grid-cols-3">
        <article className="rounded-2xl border border-red-200 bg-red-50 p-4">
          <h3 className="mb-2 font-semibold text-red-700">⚠ Risks</h3>
          <ul className="space-y-2 text-sm">
            {insight.risks.map((item, idx) => (
              <li key={idx} className="rounded-md bg-white/70 p-2">
                {item}
              </li>
            ))}
          </ul>
        </article>

        <article className="rounded-2xl border border-green-200 bg-green-50 p-4">
          <h3 className="mb-2 font-semibold text-green-700">✓ Opportunities</h3>
          <ul className="space-y-2 text-sm">
            {insight.opportunities.map((item, idx) => (
              <li key={idx} className="rounded-md bg-white/70 p-2">
                {item}
              </li>
            ))}
          </ul>
        </article>

        <article className="rounded-2xl border border-blue-200 bg-blue-50 p-4">
          <h3 className="mb-2 font-semibold text-blue-700">💡 Recommendations</h3>
          <ul className="space-y-2 text-sm">
            {insight.recommendations.map((item, idx) => (
              <li key={idx} className="rounded-md bg-white/70 p-2">
                {item}
              </li>
            ))}
          </ul>
        </article>
      </div>

      <details className="rounded-2xl border border-slate-200 bg-white/95 p-4 shadow">
        <summary className="cursor-pointer text-sm font-semibold">Tier 1 Metrics Summary</summary>
        <div className="mt-3 grid gap-2 text-sm text-slate-700 md:grid-cols-2">
          <p>Avg similarity: {tier1Metrics.avg_similarity_score.toFixed(3)}</p>
          <p>Docs above threshold: {tier1Metrics.docs_above_threshold}</p>
          <p>Total docs retrieved: {tier1Metrics.total_docs_retrieved}</p>
          <p>Context tokens: {tier1Metrics.context_total_tokens}</p>
          <p>LLM latency: {tier1Metrics.llm_latency_ms.toFixed(1)} ms</p>
          <p>LLM retry count: {tier1Metrics.llm_retry_count}</p>
          {retrievalDiagnostics && (
            <>
              <p>Query variants: {retrievalDiagnostics.query_variants.length}</p>
              <p>
                CRAG docs kept: {retrievalDiagnostics.retrieved_after_crag}/
                {retrievalDiagnostics.retrieved_before_crag}
              </p>
              <p>Parent context docs: {retrievalDiagnostics.parent_context_docs ?? 0}</p>
            </>
          )}
          {tier1Metrics.used_fallback && (
            <>
              <p>Articles fetched: {tier1Metrics.articles_fetched}</p>
              <p>Articles surviving: {tier1Metrics.articles_surviving}</p>
            </>
          )}
        </div>
      </details>

      <article className="rounded-2xl border border-slate-200 bg-white/95 p-4 shadow">
        <h3 className="font-semibold">Actions Preview</h3>
        <ul className="mt-2 space-y-2 text-sm">
          {insight.actions.map((action, idx) => (
            <li key={idx} className="flex items-center justify-between rounded-lg border border-slate-200 p-2">
              <span>{action.title}</span>
              <span className="rounded-full bg-slate-100 px-2 py-1 text-xs font-medium">
                {action.priority}
              </span>
            </li>
          ))}
        </ul>
      </article>

      <div className="flex flex-wrap gap-3">
        <button
          onClick={onReview}
          className="rounded-lg bg-ink px-4 py-2 font-semibold text-white hover:bg-slate-800"
        >
          Review & Execute Actions
        </button>
        <button
          onClick={onStartOver}
          className="rounded-lg border border-slate-300 px-4 py-2 font-semibold text-slate-700 hover:bg-slate-100"
        >
          Start Over
        </button>
      </div>
    </section>
  );
}