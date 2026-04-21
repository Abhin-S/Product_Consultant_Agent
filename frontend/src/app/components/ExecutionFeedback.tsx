"use client";

import { ActionResult } from "../../lib/types";

type Props = {
  results: ActionResult[];
  onAnalyzeAnother: () => void;
};

export default function ExecutionFeedback({ results, onAnalyzeAnother }: Props) {
  return (
    <section className="space-y-4 rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
      <h2 className="text-xl font-semibold">Execution Feedback</h2>

      <div className="space-y-3">
        {results.length === 0 ? (
          <p className="text-sm text-slate-600">No execution results available.</p>
        ) : (
          results.map((result, idx) => (
            <article
              key={`${result.action_title}-${idx}`}
              className={`rounded-xl border p-3 ${
                result.status === "executed"
                  ? "border-green-200 bg-green-50 text-green-900"
                  : "border-red-200 bg-red-50 text-red-900"
              }`}
            >
              <p className="font-semibold">
                {result.status === "executed" ? "✓ Success" : "✗ Failed"}: {result.action_title}
              </p>
              {result.status === "executed" ? (
                <p className="mt-1 text-sm">
                  {result.target_provider === "notion"
                    ? `Notion task entry created: ${result.external_id}`
                    : `Jira issue: ${result.external_id}`}
                </p>
              ) : (
                <p className="mt-1 text-sm">{result.error_message || "Execution failed"}</p>
              )}

              {result.target_provider === "notion" && result.insight_page_url && (
                <p className="mt-1 text-sm">
                  Insight page: {" "}
                  <a
                    href={result.insight_page_url}
                    target="_blank"
                    rel="noreferrer"
                    className="font-medium text-blue-700 underline"
                  >
                    Open in Notion
                  </a>
                </p>
              )}

              {result.target_provider === "notion" && !result.insight_page_url && result.status === "executed" && (
                <p className="mt-1 text-sm">Strategy page was not updated. Provide a Report Page ID in Integrations.</p>
              )}
            </article>
          ))
        )}
      </div>

      <button
        onClick={onAnalyzeAnother}
        className="rounded-lg bg-ink px-4 py-2 font-semibold text-white hover:bg-slate-800"
      >
        Start New Decision
      </button>
    </section>
  );
}