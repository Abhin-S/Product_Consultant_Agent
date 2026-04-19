"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

import ActionConfirmation from "../components/ActionConfirmation";
import ExecutionFeedback from "../components/ExecutionFeedback";
import IdeaInputForm from "../components/IdeaInputForm";
import InsightDisplay from "../components/InsightDisplay";
import api from "../../lib/axios";
import {
  AnalyzeRequest,
  AnalyzeResponse,
  ExecuteResponse,
  InsightOutput,
  Tier1Metrics
} from "../../lib/types";

type AnalyzeUIState = "input" | "loading" | "insights" | "confirm" | "executing" | "feedback";

function AnalyzePageContent() {
  const searchParams = useSearchParams();

  const [uiState, setUiState] = useState<AnalyzeUIState>("input");
  const [analyzeResponse, setAnalyzeResponse] = useState<AnalyzeResponse | null>(null);
  const [executionResults, setExecutionResults] = useState<ExecuteResponse>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    if (searchParams.get("prefill") !== "1") {
      return;
    }

    const raw = localStorage.getItem("prefillAnalyze");
    if (!raw) {
      return;
    }

    try {
      const parsed = JSON.parse(raw) as {
        session_id: string;
        insights: InsightOutput;
        used_fallback: boolean;
        tier1_metrics: Tier1Metrics;
        evaluation_status: "pending" | "skipped" | "not_requested";
      };

      setAnalyzeResponse({
        session_id: parsed.session_id,
        insights: parsed.insights,
        used_fallback: parsed.used_fallback,
        retrieved_sources: [],
        evaluation_status: parsed.evaluation_status,
        tier1_metrics: parsed.tier1_metrics
      });
      setUiState("insights");
    } catch {
      // Ignore malformed prefill payload.
    }
  }, [searchParams]);

  const resetFlow = () => {
    setUiState("input");
    setAnalyzeResponse(null);
    setExecutionResults([]);
    setError("");
  };

  const handleAnalyze = async (payload: AnalyzeRequest) => {
    setError("");
    setUiState("loading");

    try {
      const response = await api.post<AnalyzeResponse>("/analyze", payload);
      setAnalyzeResponse(response.data);
      setUiState("insights");
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Failed to analyze idea.");
      setUiState("input");
    }
  };

  const handleExecute = async (target: "notion" | "jira" | "both", selectedActionIndices: number[]) => {
    if (!analyzeResponse) {
      return;
    }

    setError("");
    setUiState("executing");

    try {
      const response = await api.post<ExecuteResponse>("/execute", {
        session_id: analyzeResponse.session_id,
        target,
        selected_action_indices: selectedActionIndices
      });

      setExecutionResults(response.data);
      setUiState("feedback");
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Execution failed.");
      setUiState("confirm");
    }
  };

  return (
    <div className="space-y-6">
      {error && <p className="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>}

      {uiState === "input" && <IdeaInputForm onSubmit={handleAnalyze} />}

      {uiState === "loading" && (
        <div className="rounded-2xl border border-slate-200 bg-white/90 p-8 text-center shadow">
          <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-4 border-slate-300 border-t-ember" />
          <p className="text-lg font-semibold">Analyzing your idea using Gemma 4...</p>
          <p className="mt-1 text-sm text-slate-600">Retrieving relevant case studies...</p>
        </div>
      )}

      {uiState === "insights" && analyzeResponse && (
        <InsightDisplay
          insight={analyzeResponse.insights}
          usedFallback={analyzeResponse.used_fallback}
          evaluationStatus={analyzeResponse.evaluation_status}
          tier1Metrics={analyzeResponse.tier1_metrics}
          groundingStatus={analyzeResponse.grounding_status}
          faithfulnessCorrected={analyzeResponse.faithfulness_corrected}
          retrievalDiagnostics={analyzeResponse.retrieval_diagnostics}
          onReview={() => setUiState("confirm")}
          onStartOver={resetFlow}
        />
      )}

      {uiState === "confirm" && analyzeResponse && (
        <ActionConfirmation
          actions={analyzeResponse.insights.actions}
          onConfirmExecute={handleExecute}
          onCancel={() => setUiState("insights")}
        />
      )}

      {uiState === "executing" && (
        <div className="rounded-2xl border border-slate-200 bg-white/90 p-8 text-center shadow">
          <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-4 border-slate-300 border-t-pine" />
          <p className="text-lg font-semibold">Executing selected actions...</p>
        </div>
      )}

      {uiState === "feedback" && (
        <ExecutionFeedback
          results={executionResults}
          onAnalyzeAnother={() => {
            localStorage.removeItem("prefillAnalyze");
            resetFlow();
          }}
        />
      )}
    </div>
  );
}

export default function AnalyzePage() {
  return (
    <Suspense fallback={<div className="rounded-2xl border border-slate-200 bg-white/90 p-6" />}>
      <AnalyzePageContent />
    </Suspense>
  );
}