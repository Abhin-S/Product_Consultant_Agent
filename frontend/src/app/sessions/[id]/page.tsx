"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import api from "../../../lib/axios";
import { SessionDetail } from "../../../lib/types";
import useRequireAuth from "../../../lib/useRequireAuth";

export default function SessionDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const authStatus = useRequireAuth();

  const [session, setSession] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchDetail = async () => {
      setLoading(true);
      setError("");
      try {
        const response = await api.get<SessionDetail>(`/sessions/${params.id}`);
        setSession(response.data);
      } catch (err: any) {
        const detail = err?.response?.data?.detail;
        setError(typeof detail === "string" ? detail : "Failed to load session.");
      } finally {
        setLoading(false);
      }
    };

    if (params.id) {
      fetchDetail();
    }
  }, [params.id]);

  const ragasMessage = useMemo(() => {
    const status = session?.evaluation_log?.ragas_eval_status;
    if (!status) return "Evaluation not requested";
    if (status === "pending") return "Pending...";
    if (status === "skipped") return "Evaluation skipped";
    if (status === "failed") return "Evaluation failed";
    return "Completed";
  }, [session]);

  if (authStatus === "checking") {
    return <p className="text-sm text-slate-600">Checking authentication...</p>;
  }

  if (loading) {
    return <p>Loading session...</p>;
  }

  if (error) {
    return <p className="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>;
  }

  if (!session) {
    return <p>Session not found.</p>;
  }

  const insight = session.raw_output;

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
        <h1 className="mb-2 text-2xl font-bold">Session Detail</h1>
        <p className="text-sm text-slate-500">{new Date(session.created_at).toLocaleString()}</p>
        <h2 className="mt-4 text-sm font-semibold uppercase tracking-wide text-slate-500">Idea</h2>
        <p className="mt-1 whitespace-pre-wrap text-slate-800">{session.idea_text}</p>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
        <h2 className="text-xl font-semibold">Insight Output</h2>
        <p className="mt-2 rounded-xl bg-slate-50 p-4 text-slate-800">{insight.idea_summary}</p>

        <div className="mt-4 grid gap-4 md:grid-cols-3">
          <div>
            <h3 className="mb-2 font-semibold text-red-700">Risks</h3>
            <ul className="space-y-2">
              {insight.risks.map((risk, idx) => (
                <li key={idx} className="rounded-lg bg-red-50 p-3 text-sm">
                  {risk}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="mb-2 font-semibold text-green-700">Opportunities</h3>
            <ul className="space-y-2">
              {insight.opportunities.map((opportunity, idx) => (
                <li key={idx} className="rounded-lg bg-green-50 p-3 text-sm">
                  {opportunity}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="mb-2 font-semibold text-sky-700">Recommendations</h3>
            <ul className="space-y-2">
              {insight.recommendations.map((recommendation, idx) => (
                <li key={idx} className="rounded-lg bg-blue-50 p-3 text-sm">
                  {recommendation}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
        <h2 className="text-xl font-semibold">Evaluation</h2>
        <p className="mt-1 text-sm text-slate-600">
          Evaluation scores are for developer monitoring purposes.
        </p>

        {session.evaluation_log ? (
          <>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <tbody>
                  <tr className="border-b">
                    <td className="py-2 font-medium">Avg Similarity</td>
                    <td>{session.evaluation_log.avg_similarity_score.toFixed(3)}</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-2 font-medium">Docs Above Threshold</td>
                    <td>{session.evaluation_log.docs_above_threshold}</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-2 font-medium">Context Total Tokens</td>
                    <td>{session.evaluation_log.context_total_tokens}</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-2 font-medium">Fallback Used</td>
                    <td>{session.evaluation_log.used_fallback ? "Yes" : "No"}</td>
                  </tr>
                  <tr>
                    <td className="py-2 font-medium">LLM Latency (ms)</td>
                    <td>{session.evaluation_log.llm_latency_ms.toFixed(1)}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="mt-4 rounded-xl bg-slate-50 p-4">
              <p className="text-sm font-semibold">RAGAS Status: {ragasMessage}</p>
              {session.evaluation_log.ragas_eval_status === "completed" && (
                <div className="mt-2 grid gap-2 text-sm md:grid-cols-2">
                  <p>Context Precision: {session.evaluation_log.context_precision?.toFixed(3)}</p>
                  <p>Context Recall: {session.evaluation_log.context_recall?.toFixed(3)}</p>
                  <p>Faithfulness: {session.evaluation_log.faithfulness?.toFixed(3)}</p>
                  <p>Answer Relevance: {session.evaluation_log.answer_relevance?.toFixed(3)}</p>
                </div>
              )}
            </div>
          </>
        ) : (
          <p className="mt-3 text-sm text-slate-600">No evaluation data for this session.</p>
        )}
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
        <h2 className="text-xl font-semibold">Action Logs</h2>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-100">
              <tr>
                <th className="px-3 py-2">Action Title</th>
                <th className="px-3 py-2">Provider</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">External ID</th>
                <th className="px-3 py-2">Error</th>
                <th className="px-3 py-2">Date</th>
              </tr>
            </thead>
            <tbody>
              {session.action_logs.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-3 py-3">
                    No actions executed yet.
                  </td>
                </tr>
              ) : (
                session.action_logs.map((log) => (
                  <tr key={log.id} className="border-t">
                    <td className="px-3 py-2">{log.title}</td>
                    <td className="px-3 py-2">{log.target_provider}</td>
                    <td className="px-3 py-2">{log.status}</td>
                    <td className="px-3 py-2">{log.external_id || "-"}</td>
                    <td className="px-3 py-2">{log.error_message || "-"}</td>
                    <td className="px-3 py-2">{new Date(log.created_at).toLocaleString()}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      <button
        onClick={() => {
          localStorage.setItem(
            "prefillAnalyze",
            JSON.stringify({
              session_id: session.id,
              insights: session.raw_output,
              used_fallback: session.used_fallback,
              tier1_metrics: session.evaluation_log
                ? {
                    avg_similarity_score: session.evaluation_log.avg_similarity_score,
                    docs_above_threshold: session.evaluation_log.docs_above_threshold,
                    total_docs_retrieved: session.evaluation_log.total_docs_retrieved,
                    context_total_tokens: session.evaluation_log.context_total_tokens,
                    context_local_ratio: session.evaluation_log.context_local_ratio,
                    context_dynamic_ratio: session.evaluation_log.context_dynamic_ratio,
                    used_fallback: session.evaluation_log.used_fallback,
                    articles_fetched: session.evaluation_log.articles_fetched,
                    articles_surviving: session.evaluation_log.articles_surviving,
                    llm_latency_ms: session.evaluation_log.llm_latency_ms,
                    llm_retry_count: session.evaluation_log.llm_retry_count
                  }
                : {
                    avg_similarity_score: 0,
                    docs_above_threshold: 0,
                    total_docs_retrieved: 0,
                    context_total_tokens: 0,
                    context_local_ratio: 0,
                    context_dynamic_ratio: 0,
                    used_fallback: false,
                    articles_fetched: 0,
                    articles_surviving: 0,
                    llm_latency_ms: 0,
                    llm_retry_count: 0
                  },
              evaluation_status:
                session.evaluation_log?.ragas_eval_status === "pending"
                  ? "pending"
                  : session.evaluation_log?.ragas_eval_status === "skipped"
                    ? "skipped"
                    : "not_requested"
            })
          );
          router.push("/analyze?prefill=1");
        }}
        className="rounded-lg bg-ink px-4 py-2 font-semibold text-white hover:bg-slate-800"
      >
        Re-Execute Actions
      </button>
    </div>
  );
}