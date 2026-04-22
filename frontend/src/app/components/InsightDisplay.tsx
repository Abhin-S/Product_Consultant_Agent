"use client";

import { InsightOutput, Tier1Metrics } from "../../lib/types";

type Props = {
  insight: InsightOutput;
  usedFallback: boolean;
  evaluationStatus: "pending" | "skipped" | "not_requested";
  tier1Metrics: Tier1Metrics;
  groundingStatus?:
    | "grounded"
    | "partial"
    | "not_grounded"
    | "unknown"
    | "not_requested"
    | "bypassed"
    | "insufficient_context";
  faithfulnessCorrected?: boolean;
  retrievalDiagnostics?: {
    query_variants: string[];
    retrieved_before_crag: number;
    retrieved_after_crag: number;
    parent_context_docs?: number;
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
  onReview: () => void;
  onStartOver: () => void;
};

function statusLabel(status: "pending" | "skipped" | "not_requested") {
  if (status === "pending") return "Pending evaluation";
  if (status === "skipped") return "Evaluation skipped";
  return "Evaluation not requested";
}

function hasText(value: string | null | undefined): boolean {
  return typeof value === "string" && value.trim().length > 0;
}

function compactList(values: Array<string | null | undefined> | null | undefined): string[] {
  if (!Array.isArray(values)) return [];
  return values
    .map((item) => (typeof item === "string" ? item.trim() : ""))
    .filter((item) => item.length > 0);
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
  const diagnosis = (insight.brand_diagnosis || insight.idea_summary || "").trim();
  const marketInsight = (insight.market_insight || "").trim();
  const suggestedPositioning =
    (insight.suggested_positioning && insight.suggested_positioning.length > 0
      ? insight.suggested_positioning
      : insight.recommendations) || [];
  const finalPositioning = (insight.final_positioning || "").trim();
  const targetAudience = (insight.target_audience || "").trim();
  const chosenStrategy = (insight.chosen_strategy || "").trim();
  const rejectedDirections = compactList(insight.rejected_directions);
  const tradeOffs = compactList(insight.trade_offs);
  const risks = compactList(insight.risks);
  const opportunities = compactList(insight.opportunities);

  const metadataName = (insight.database_metadata?.name || "").trim();
  const metadataPositioning =
    (insight.database_metadata?.brand_positioning || insight.database_metadata?.idea_description || "").trim();
  const metadataRisk =
    (insight.database_metadata?.brand_risk_level || insight.database_metadata?.risk_level || "").trim();
  const metadataTags = compactList(insight.database_metadata?.tags);

  const hasSuggestionsSection = hasText(marketInsight) || suggestedPositioning.length > 0 || risks.length > 0 || opportunities.length > 0;
  const hasDecisionSection =
    hasText(finalPositioning) ||
    hasText(targetAudience) ||
    hasText(chosenStrategy) ||
    rejectedDirections.length > 0 ||
    tradeOffs.length > 0;
  const hasDatabaseMetadata =
    !!insight.database_metadata &&
    (hasText(metadataName) ||
      hasText(metadataPositioning) ||
      hasText(metadataRisk) ||
      typeof insight.database_metadata?.confidence_score === "number" ||
      metadataTags.length > 0);

  return (
    <section className="space-y-4">
      <article className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
        <h2 className="text-xl font-semibold">Brand Diagnosis</h2>
        {hasText(diagnosis) ? (
          <p className="mt-2 rounded-xl bg-slate-50 p-4 text-slate-800">{diagnosis}</p>
        ) : (
          <p className="mt-2 rounded-xl bg-slate-50 p-4 text-slate-600">
            No diagnosis was generated for this query.
          </p>
        )}

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

      {hasSuggestionsSection && (
        <article className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
          <h3 className="text-lg font-semibold">Suggestions (AI)</h3>
          <p className="mt-1 text-sm text-slate-600">
            AI suggestions are grounded in retrieved case studies. Use these to make explicit decisions.
          </p>

          <div className="mt-3 space-y-3">
            {hasText(marketInsight) && (
              <div className="rounded-xl bg-slate-50 p-4">
                <p className="text-sm font-semibold text-slate-700">Market Insight</p>
                <p className="mt-1 text-sm text-slate-800">{marketInsight}</p>
              </div>
            )}

            {suggestedPositioning.length > 0 && (
              <div className="rounded-xl bg-slate-50 p-4">
                <p className="text-sm font-semibold text-slate-700">Suggested Positioning</p>
                <ul className="mt-2 space-y-2 text-sm text-slate-800">
                  {suggestedPositioning.map((item, idx) => (
                    <li key={idx} className="rounded-md bg-white/80 p-2">
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {(risks.length > 0 || opportunities.length > 0) && (
              <div className="grid gap-4 md:grid-cols-2">
                {risks.length > 0 && (
                  <div className="rounded-xl border border-red-200 bg-red-50 p-4">
                    <h4 className="mb-2 font-semibold text-red-700">Risks</h4>
                    <ul className="space-y-2 text-sm">
                      {risks.map((item, idx) => (
                        <li key={idx} className="rounded-md bg-white/70 p-2">
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {opportunities.length > 0 && (
                  <div className="rounded-xl border border-green-200 bg-green-50 p-4">
                    <h4 className="mb-2 font-semibold text-green-700">Opportunities</h4>
                    <ul className="space-y-2 text-sm">
                      {opportunities.map((item, idx) => (
                        <li key={idx} className="rounded-md bg-white/70 p-2">
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </article>
      )}

      {hasDecisionSection && (
        <article className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
          <h3 className="text-lg font-semibold">Decisions (User)</h3>
          <p className="mt-1 text-sm text-slate-600">
            These fields should be finalized by you in Notion. Tasks should follow these decisions.
          </p>

          {(hasText(finalPositioning) || hasText(targetAudience) || hasText(chosenStrategy)) && (
            <div className="mt-3 grid gap-4 md:grid-cols-3">
              {hasText(finalPositioning) && (
                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Final Positioning</p>
                  <p className="mt-2 text-sm text-slate-800">{finalPositioning}</p>
                </div>
              )}
              {hasText(targetAudience) && (
                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Target Audience</p>
                  <p className="mt-2 text-sm text-slate-800">{targetAudience}</p>
                </div>
              )}
              {hasText(chosenStrategy) && (
                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Chosen Strategy</p>
                  <p className="mt-2 text-sm text-slate-800">{chosenStrategy}</p>
                </div>
              )}
            </div>
          )}

          {(rejectedDirections.length > 0 || tradeOffs.length > 0) && (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              {rejectedDirections.length > 0 && (
                <div className="rounded-xl border border-slate-200 bg-white p-4">
                  <p className="text-sm font-semibold">Rejected Directions</p>
                  <ul className="mt-2 space-y-2 text-sm text-slate-700">
                    {rejectedDirections.map((item, idx) => (
                      <li key={idx} className="rounded-md bg-slate-50 p-2">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {tradeOffs.length > 0 && (
                <div className="rounded-xl border border-slate-200 bg-white p-4">
                  <p className="text-sm font-semibold">Trade-offs</p>
                  <ul className="mt-2 space-y-2 text-sm text-slate-700">
                    {tradeOffs.map((item, idx) => (
                      <li key={idx} className="rounded-md bg-slate-50 p-2">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </article>
      )}

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

      {hasDatabaseMetadata && insight.database_metadata && (
        <article className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
          <h3 className="text-lg font-semibold">Database Metadata (Secondary)</h3>
          <div className="mt-3 grid gap-2 text-sm text-slate-700 md:grid-cols-2">
            {hasText(metadataName) && (
              <p>
                <span className="font-medium">Name:</span> {metadataName}
              </p>
            )}
            {hasText(metadataRisk) && (
              <p>
                <span className="font-medium">Brand Risk Level:</span> {metadataRisk}
              </p>
            )}
            {hasText(metadataPositioning) && (
              <p className="md:col-span-2">
                <span className="font-medium">Brand Positioning:</span> {metadataPositioning}
              </p>
            )}
            {typeof insight.database_metadata.confidence_score === "number" && (
              <p>
                <span className="font-medium">Confidence Score:</span> {insight.database_metadata.confidence_score}
              </p>
            )}
            <div className="md:col-span-2">
              <p className="mb-1 font-medium">Tags:</p>
              <div className="flex flex-wrap gap-2">
                {metadataTags.length > 0 ? (
                  metadataTags.map((tag) => (
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
              {retrievalDiagnostics.abstain_reason && (
                <p className="md:col-span-2">Coverage warning: {retrievalDiagnostics.abstain_reason}</p>
              )}
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
        <h3 className="font-semibold">Execution (Actionable Tasks)</h3>
        <ul className="mt-2 space-y-2 text-sm">
          {insight.actions.map((action, idx) => (
            <li key={idx} className="flex items-center justify-between rounded-lg border border-slate-200 p-2">
              <div>
                <p>{action.title}</p>
                <p className="text-xs text-slate-500">
                  {(action.decision_type || "other").replace("_", " ")} · impact {(action.impact || "medium")}
                </p>
              </div>
              <span className="rounded-full bg-slate-100 px-2 py-1 text-xs font-medium">{action.priority}</span>
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