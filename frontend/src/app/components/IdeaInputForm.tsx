"use client";

import { FormEvent, useState } from "react";

import { AnalyzeRequest } from "../../lib/types";

type Props = {
  onSubmit: (payload: AnalyzeRequest) => void | Promise<void>;
};

export default function IdeaInputForm({ onSubmit }: Props) {
  const [idea, setIdea] = useState("");
  const [useFallback, setUseFallback] = useState(true);
  const [topK, setTopK] = useState(5);
  const [runEvaluation, setRunEvaluation] = useState(true);

  const maxChars = 1000;

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    await onSubmit({
      idea,
      use_fallback: useFallback,
      top_k: topK,
      run_evaluation: runEvaluation
    });
  };

  return (
    <section className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
      <h1 className="text-2xl font-bold">Analyze Product Idea</h1>
      <p className="mt-1 text-sm text-slate-600">
        Submit your startup concept to generate strategy insights and execution-ready tasks.
      </p>

      <form onSubmit={handleSubmit} className="mt-5 space-y-4">
        <label className="block">
          <span className="mb-1 block text-sm font-medium">Startup Idea</span>
          <textarea
            value={idea}
            onChange={(e) => setIdea(e.target.value)}
            maxLength={maxChars}
            required
            rows={6}
            className="w-full rounded-lg border border-slate-300 p-3 outline-none transition focus:border-sky"
            placeholder="Describe your product, target users, value proposition, and market assumptions..."
          />
          <span className="mt-1 block text-right text-xs text-slate-500">
            {idea.length}/{maxChars}
          </span>
        </label>

        <div className="grid gap-4 md:grid-cols-3">
          <label className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
            <input
              type="checkbox"
              checked={useFallback}
              onChange={(e) => setUseFallback(e.target.checked)}
            />
            <span className="text-sm">Use web fallback retrieval</span>
          </label>

          <label className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
            <span className="mb-1 block text-sm">Top K results</span>
            <input
              type="number"
              min={1}
              max={10}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full rounded border border-slate-300 px-2 py-1 text-sm"
            />
          </label>

          <label className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
            <span className="mb-1 block text-sm">Enable evaluation</span>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={runEvaluation}
                onChange={(e) => setRunEvaluation(e.target.checked)}
              />
              <span className="text-sm">On</span>
            </div>
            <p className="mt-1 text-xs text-slate-500">
              Evaluation monitors retrieval quality for developers.
            </p>
          </label>
        </div>

        <button
          type="submit"
          disabled={!idea.trim()}
          className="rounded-lg bg-ink px-4 py-2 font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          Analyze Idea
        </button>
      </form>
    </section>
  );
}