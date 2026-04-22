"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import api from "../../../lib/axios";
import {
  InsightOutput,
  SessionActionLog,
  SessionChatRequestPayload,
  SessionChatResponse,
  SessionChatTurn,
  SessionDetail
} from "../../../lib/types";
import useRequireAuth from "../../../lib/useRequireAuth";

const INPUT_INSTRUCTION =
  "Describe your product, target users, current positioning, and the problem you are facing...";

function insightDiagnosis(insight: InsightOutput): string {
  return (insight.brand_diagnosis || insight.idea_summary || "No diagnosis available.").trim();
}

function insightSuggestions(insight: InsightOutput): string[] {
  if (Array.isArray(insight.suggested_positioning) && insight.suggested_positioning.length > 0) {
    return insight.suggested_positioning;
  }
  if (Array.isArray(insight.recommendations) && insight.recommendations.length > 0) {
    return insight.recommendations;
  }
  return [];
}

function abstentionText(
  insight: InsightOutput,
  groundingStatus?: SessionChatTurn["grounding_status"]
): string {
  const explicit = (insight.abstention_message || "").trim();
  if (explicit) {
    return explicit;
  }

  if (groundingStatus === "insufficient_context" || groundingStatus === "not_grounded") {
    return (
      "Sorry, I do not have enough grounded evidence in the available documents to answer that question reliably. "
      + "Please ask a question tied to the ingested brand materials or add more relevant evidence and try again."
    );
  }

  return "";
}

function clampTopK(value: number): number {
  if (!Number.isFinite(value)) {
    return 1;
  }

  return Math.min(10, Math.max(1, Math.round(value)));
}

function InsightOutputPanel({
  insight,
  groundingStatus
}: {
  insight: InsightOutput;
  groundingStatus?: SessionChatTurn["grounding_status"];
}) {
  const abstentionMessage = abstentionText(insight, groundingStatus);
  if (abstentionMessage) {
    return (
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-amber-900">Response</p>
        <p className="mt-2 whitespace-pre-wrap break-words text-sm text-slate-800 [overflow-wrap:anywhere]">
          {abstentionMessage}
        </p>
      </div>
    );
  }

  const diagnosis = insightDiagnosis(insight);
  const suggestions = insightSuggestions(insight);
  const risks = Array.isArray(insight.risks) ? insight.risks : [];
  const opportunities = Array.isArray(insight.opportunities) ? insight.opportunities : [];
  const rejectedDirections = Array.isArray(insight.rejected_directions) ? insight.rejected_directions : [];
  const tradeOffs = Array.isArray(insight.trade_offs) ? insight.trade_offs : [];

  const finalPositioning = (insight.final_positioning || "").trim();
  const targetAudience = (insight.target_audience || "").trim();
  const chosenStrategy = (insight.chosen_strategy || "").trim();

  const hasDecisionSection =
    finalPositioning.length > 0 ||
    targetAudience.length > 0 ||
    chosenStrategy.length > 0 ||
    rejectedDirections.length > 0 ||
    tradeOffs.length > 0;

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Brand Diagnosis</p>
        <p className="mt-2 whitespace-pre-wrap break-words text-sm text-slate-800 [overflow-wrap:anywhere]">
          {diagnosis}
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm">
          <p className="font-semibold text-red-700">Risks</p>
          <ul className="mt-2 space-y-2">
            {risks.length === 0 ? (
              <li className="rounded-md bg-white/80 p-2 text-slate-600">No explicit risks provided.</li>
            ) : (
              risks.map((item, idx) => (
                <li key={idx} className="rounded-md bg-white/80 p-2 break-words [overflow-wrap:anywhere]">
                  {item}
                </li>
              ))
            )}
          </ul>
        </div>

        <div className="rounded-lg border border-green-200 bg-green-50 p-3 text-sm">
          <p className="font-semibold text-green-700">Opportunities</p>
          <ul className="mt-2 space-y-2">
            {opportunities.length === 0 ? (
              <li className="rounded-md bg-white/80 p-2 text-slate-600">No explicit opportunities provided.</li>
            ) : (
              opportunities.map((item, idx) => (
                <li key={idx} className="rounded-md bg-white/80 p-2 break-words [overflow-wrap:anywhere]">
                  {item}
                </li>
              ))
            )}
          </ul>
        </div>

        <div className="rounded-lg border border-sky-200 bg-sky-50 p-3 text-sm">
          <p className="font-semibold text-sky-700">Suggested Positioning</p>
          <ul className="mt-2 space-y-2">
            {suggestions.length === 0 ? (
              <li className="rounded-md bg-white/80 p-2 text-slate-600">No suggested positioning provided.</li>
            ) : (
              suggestions.map((item, idx) => (
                <li key={idx} className="rounded-md bg-white/80 p-2 break-words [overflow-wrap:anywhere]">
                  {item}
                </li>
              ))
            )}
          </ul>
        </div>
      </div>

      {hasDecisionSection && (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            {finalPositioning && (
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
                <p className="font-semibold text-slate-700">Final Positioning</p>
                <p className="mt-1 whitespace-pre-wrap break-words text-slate-800 [overflow-wrap:anywhere]">
                  {finalPositioning}
                </p>
              </div>
            )}
            {targetAudience && (
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
                <p className="font-semibold text-slate-700">Target Audience</p>
                <p className="mt-1 whitespace-pre-wrap break-words text-slate-800 [overflow-wrap:anywhere]">
                  {targetAudience}
                </p>
              </div>
            )}
            {chosenStrategy && (
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
                <p className="font-semibold text-slate-700">Chosen Strategy</p>
                <p className="mt-1 whitespace-pre-wrap break-words text-slate-800 [overflow-wrap:anywhere]">
                  {chosenStrategy}
                </p>
              </div>
            )}
          </div>

          {(rejectedDirections.length > 0 || tradeOffs.length > 0) && (
            <div className="grid gap-4 md:grid-cols-2">
              {rejectedDirections.length > 0 && (
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
                  <p className="font-semibold text-slate-700">Rejected Directions</p>
                  <ul className="mt-2 space-y-1">
                    {rejectedDirections.map((item, idx) => (
                      <li key={idx} className="rounded-md bg-white/80 p-2 break-words [overflow-wrap:anywhere]">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {tradeOffs.length > 0 && (
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
                  <p className="font-semibold text-slate-700">Trade-offs</p>
                  <ul className="mt-2 space-y-1">
                    {tradeOffs.map((item, idx) => (
                      <li key={idx} className="rounded-md bg-white/80 p-2 break-words [overflow-wrap:anywhere]">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function EvaluationPanel({
  evaluationLog,
  ragasMessage,
}: {
  evaluationLog: SessionDetail["evaluation_log"];
  ragasMessage: string;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Evaluation</p>
      {evaluationLog ? (
        <>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full table-fixed text-left text-sm">
              <tbody>
                <tr className="border-b">
                  <td className="py-2 pr-3 font-medium align-top break-words [overflow-wrap:anywhere]">Avg Similarity</td>
                  <td className="py-2 align-top break-words [overflow-wrap:anywhere]">
                    {evaluationLog.avg_similarity_score.toFixed(3)}
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="py-2 pr-3 font-medium align-top break-words [overflow-wrap:anywhere]">Docs Above Threshold</td>
                  <td className="py-2 align-top break-words [overflow-wrap:anywhere]">{evaluationLog.docs_above_threshold}</td>
                </tr>
                <tr className="border-b">
                  <td className="py-2 pr-3 font-medium align-top break-words [overflow-wrap:anywhere]">Context Total Tokens</td>
                  <td className="py-2 align-top break-words [overflow-wrap:anywhere]">{evaluationLog.context_total_tokens}</td>
                </tr>
                <tr className="border-b">
                  <td className="py-2 pr-3 font-medium align-top break-words [overflow-wrap:anywhere]">Fallback Used</td>
                  <td className="py-2 align-top break-words [overflow-wrap:anywhere]">
                    {evaluationLog.used_fallback ? "Yes" : "No"}
                  </td>
                </tr>
                <tr>
                  <td className="py-2 pr-3 font-medium align-top break-words [overflow-wrap:anywhere]">LLM Latency (ms)</td>
                  <td className="py-2 align-top break-words [overflow-wrap:anywhere]">
                    {evaluationLog.llm_latency_ms.toFixed(1)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-sm font-medium text-slate-700">RAGAS Status: {ragasMessage}</p>
        </>
      ) : (
        <p className="mt-2 text-sm text-slate-600">No evaluation data for this session.</p>
      )}
    </div>
  );
}

function ActionLogsPanel({ actionLogs }: { actionLogs: SessionActionLog[] }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Action Logs</p>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full table-fixed text-left text-sm">
          <thead className="bg-slate-100 text-slate-700">
            <tr>
              <th className="px-2 py-2 text-left">Action</th>
              <th className="px-2 py-2 text-left">Provider</th>
              <th className="px-2 py-2 text-left">Status</th>
              <th className="px-2 py-2 text-left">Date</th>
            </tr>
          </thead>
          <tbody>
            {actionLogs.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-2 py-2 text-slate-600">
                  No actions executed yet.
                </td>
              </tr>
            ) : (
              actionLogs.map((log) => (
                <tr key={log.id} className="border-t">
                  <td className="px-2 py-2 align-top break-words [overflow-wrap:anywhere]">{log.title}</td>
                  <td className="px-2 py-2 align-top break-words [overflow-wrap:anywhere]">{log.target_provider}</td>
                  <td className="px-2 py-2 align-top break-words [overflow-wrap:anywhere]">{log.status}</td>
                  <td className="px-2 py-2 align-top break-words [overflow-wrap:anywhere]">
                    {new Date(log.created_at).toLocaleString()}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function SessionDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const authStatus = useRequireAuth();

  const [session, setSession] = useState<SessionDetail | null>(null);
  const [chatTurns, setChatTurns] = useState<SessionChatTurn[]>([]);
  const [chatMessage, setChatMessage] = useState("");
  const [followUpTopK, setFollowUpTopK] = useState(5);
  const [followUpUseFallback, setFollowUpUseFallback] = useState(true);
  const [followUpRunEvaluation, setFollowUpRunEvaluation] = useState(true);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchDetail = async () => {
      setLoading(true);
      setError("");
      try {
        const response = await api.get<SessionDetail>(`/sessions/${params.id}`);
        setSession(response.data);
        setChatTurns(response.data.chat_turns || []);
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
    if (status === "not_requested") return "Evaluation not requested";
    if (status === "pending") return "Pending...";
    if (status === "skipped") return "Evaluation skipped";
    if (status === "failed") return "Evaluation failed";
    return "Completed";
  }, [session]);

  const handleSendFollowUp = async () => {
    const message = chatMessage.trim();
    if (!message || !params.id || chatLoading) {
      return;
    }

    setChatLoading(true);
    setChatError("");

    try {
      const normalizedTopK = Number.isFinite(followUpTopK) ? clampTopK(followUpTopK) : 5;
      const requestPayload: SessionChatRequestPayload = {
        message,
        top_k: normalizedTopK,
        use_fallback: followUpUseFallback,
        run_evaluation: followUpRunEvaluation
      };
      const response = await api.post<SessionChatResponse>(`/sessions/${params.id}/chat`, requestPayload);

      const data = response.data;
      if (data.chat_turn) {
        setChatTurns((previous) => [...previous, data.chat_turn as SessionChatTurn]);
      }
      setSession((previous) =>
        previous
          ? {
              ...previous,
              confidence_score: data.insights.confidence_score,
              used_fallback: data.used_fallback,
              evaluation_log: data.evaluation_log ?? previous.evaluation_log,
              action_logs: data.action_logs ?? previous.action_logs,
              chat_turns: data.chat_turn
                ? [...(previous.chat_turns || []), (data.chat_turn as SessionChatTurn)]
                : previous.chat_turns
            }
          : previous
      );
      setChatMessage("");
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setChatError(typeof detail === "string" ? detail : "Failed to send follow-up question.");
    } finally {
      setChatLoading(false);
    }
  };

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

  const initialInsight = session.raw_output;
  const latestInsightForExecution =
    chatTurns.length > 0 ? chatTurns[chatTurns.length - 1].insights : initialInsight;

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
        <h1 className="mb-2 text-2xl font-bold">Session Detail</h1>
        <p className="text-sm text-slate-500">{new Date(session.created_at).toLocaleString()}</p>
        <h2 className="mt-4 text-sm font-semibold uppercase tracking-wide text-slate-500">Decision Question</h2>
        <p className="mt-1 whitespace-pre-wrap break-words text-slate-800 [overflow-wrap:anywhere]">
          {session.idea_text}
        </p>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
        <h2 className="text-xl font-semibold">Initial Insight Output</h2>
        <p className="mt-1 text-sm text-slate-600">
          This is the baseline analysis from the original decision question.
        </p>
        <div className="mt-4">
          <InsightOutputPanel insight={initialInsight} />
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
        <h2 className="text-xl font-semibold">Session Chat</h2>
        <p className="mt-1 text-sm text-slate-600">
          Each follow-up response is rendered as a structured result card with Decision Question, Insight Output,
          Evaluation, and Action Logs.
        </p>

        <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Instruction</p>
          <p className="mt-1 break-words text-sm text-slate-700 [overflow-wrap:anywhere]">{INPUT_INSTRUCTION}</p>
        </div>

        <div className="mt-4 space-y-4">
          {chatTurns.length === 0 ? (
            <p className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm text-slate-600">
              No follow-up messages yet. Start a conversation below.
            </p>
          ) : (
            chatTurns.map((turn) => (
              <article
                key={turn.id}
                className="min-w-0 rounded-xl border border-slate-200 bg-slate-50 p-3"
              >
                <div className="rounded-lg border border-sky-200 bg-sky-50 p-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Decision Question</p>
                  <p className="mt-1 whitespace-pre-wrap break-words text-sm text-slate-800 [overflow-wrap:anywhere]">
                    {turn.user_message}
                  </p>
                </div>

                <div className="mt-3 rounded-lg border border-slate-200 bg-white p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="text-sm font-semibold text-slate-800">Assistant Response</p>
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <span className="rounded-full bg-slate-100 px-2 py-1 text-slate-600">
                        {turn.created_at ? new Date(turn.created_at).toLocaleString() : "Just now"}
                      </span>
                      {turn.grounding_status && (
                        <span className="rounded-full bg-emerald-100 px-2 py-1 text-emerald-900">
                          Grounding: {turn.grounding_status}
                        </span>
                      )}
                      {turn.faithfulness_corrected && (
                        <span className="rounded-full bg-sky-100 px-2 py-1 text-sky-900">Faithfulness corrected</span>
                      )}
                    </div>
                  </div>

                  <div className="mt-4 space-y-4">
                    <InsightOutputPanel insight={turn.insights} groundingStatus={turn.grounding_status} />
                    <EvaluationPanel evaluationLog={session.evaluation_log} ragasMessage={ragasMessage} />
                    <ActionLogsPanel actionLogs={session.action_logs} />
                  </div>
                </div>
              </article>
            ))
          )}
        </div>

        <div className="mt-4 space-y-2">
          <label className="block text-sm font-medium">Follow-up Question</label>
          <textarea
            value={chatMessage}
            onChange={(event) => setChatMessage(event.target.value)}
            rows={4}
            maxLength={1000}
            className="w-full rounded-lg border border-slate-300 p-3 outline-none transition focus:border-sky"
            placeholder={INPUT_INSTRUCTION}
          />

          <div className="grid gap-3 md:grid-cols-3">
            <label className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
              <input
                type="checkbox"
                checked={followUpUseFallback}
                onChange={(event) => setFollowUpUseFallback(event.target.checked)}
              />
              <span className="text-sm">Use web fallback retrieval</span>
            </label>

            <label className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
              <span className="mb-1 block text-sm">Top K results</span>
              <input
                type="number"
                min={1}
                max={10}
                value={followUpTopK}
                onChange={(event) => setFollowUpTopK(clampTopK(Number(event.target.value)))}
                className="w-full rounded border border-slate-300 px-2 py-1 text-sm"
              />
            </label>

            <label className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
              <span className="mb-1 block text-sm">Enable evaluation</span>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={followUpRunEvaluation}
                  onChange={(event) => setFollowUpRunEvaluation(event.target.checked)}
                />
                <span className="text-sm">On</span>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                Evaluation updates the session diagnostics for this follow-up.
              </p>
            </label>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-slate-500">
            <span>{chatMessage.length}/1000</span>
            <button
              type="button"
              onClick={handleSendFollowUp}
              disabled={!chatMessage.trim() || chatLoading}
              className="rounded-lg bg-ink px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
            >
              {chatLoading ? "Sending..." : "Send Follow-up"}
            </button>
          </div>

          {chatError && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{chatError}</p>}
        </div>
      </section>

      <button
        onClick={() => {
          localStorage.setItem(
            "prefillAnalyze",
            JSON.stringify({
              session_id: session.id,
              insights: latestInsightForExecution,
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
