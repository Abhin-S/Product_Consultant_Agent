"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import api from "../../lib/axios";
import { ActionItem, UserIntegrationOut } from "../../lib/types";

type Props = {
  actions: ActionItem[];
  onConfirmExecute: (target: "notion" | "jira" | "both", selectedIndices: number[]) => void;
  onCancel: () => void;
};

export default function ActionConfirmation({ actions, onConfirmExecute, onCancel }: Props) {
  const [integrations, setIntegrations] = useState<UserIntegrationOut[]>([]);
  const [selectedIndices, setSelectedIndices] = useState<number[]>(actions.map((_, idx) => idx));
  const [target, setTarget] = useState<"notion" | "jira" | "both">("notion");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadIntegrations = async () => {
      try {
        const response = await api.get<UserIntegrationOut[]>("/integrations");
        setIntegrations(response.data || []);
      } catch {
        setIntegrations([]);
      }
    };
    loadIntegrations();
  }, []);

  const connected = useMemo(() => {
    const providers = new Set(integrations.map((i) => i.provider));
    return {
      notion: providers.has("notion"),
      jira: providers.has("jira")
    };
  }, [integrations]);

  useEffect(() => {
    if (connected.notion) {
      setTarget("notion");
    } else if (connected.jira) {
      setTarget("jira");
    }
  }, [connected.notion, connected.jira]);

  const canExecute = useMemo(() => {
    if (selectedIndices.length === 0) return false;
    if (target === "notion") return connected.notion;
    if (target === "jira") return connected.jira;
    return connected.notion && connected.jira;
  }, [connected.jira, connected.notion, selectedIndices.length, target]);

  const blockedReason = useMemo(() => {
    if (target === "notion" && !connected.notion) return "Please connect Notion first";
    if (target === "jira" && !connected.jira) return "Please connect Jira first";
    if (target === "both" && !(connected.notion && connected.jira)) {
      return "Please connect both Notion and Jira first";
    }
    if (selectedIndices.length === 0) return "Select at least one action";
    return "";
  }, [connected.jira, connected.notion, selectedIndices.length, target]);

  const toggleAction = (index: number) => {
    setSelectedIndices((prev) =>
      prev.includes(index) ? prev.filter((value) => value !== index) : [...prev, index]
    );
  };

  const submitExecution = async () => {
    setError("");
    setLoading(true);
    try {
      await onConfirmExecute(target, selectedIndices.sort((a, b) => a - b));
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Failed to execute actions.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="space-y-4 rounded-2xl border border-slate-200 bg-white/95 p-5 shadow">
      <header>
        <h2 className="text-xl font-semibold">Review Actions Before Execution</h2>
        <p className="text-sm text-slate-600">
          These tasks should be executed after confirming your decisions in Notion.
        </p>
      </header>

      {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm">
        <p>{connected.notion ? "✓ Notion Connected" : "✗ Notion not connected"}</p>
        <p>{connected.jira ? "✓ Jira Connected" : "✗ Jira not connected"}</p>
        {(!connected.notion || !connected.jira) && (
          <p className="mt-1">
            <Link href="/integrations">→ Connect in Integrations</Link>
          </p>
        )}
      </div>

      <div className="space-y-3">
        {actions.map((action, idx) => {
          const priorityClass =
            action.priority === "high"
              ? "bg-red-100 text-red-700"
              : action.priority === "medium"
                ? "bg-yellow-100 text-yellow-700"
                : "bg-green-100 text-green-700";

          return (
            <article key={idx} className="rounded-xl border border-slate-200 p-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="font-semibold">{action.title}</h3>
                  <p className="mt-1 text-sm text-slate-700">{action.description}</p>
                  <p className="mt-1 text-xs text-slate-500">
                    {(action.decision_type || "other").replace("_", " ")} · impact {(action.impact || "medium")}
                  </p>
                  <span className={`mt-2 inline-block rounded-full px-2 py-1 text-xs font-medium ${priorityClass}`}>
                    {action.priority}
                  </span>
                </div>
                <input
                  type="checkbox"
                  checked={selectedIndices.includes(idx)}
                  onChange={() => toggleAction(idx)}
                  className="mt-1"
                />
              </div>
            </article>
          );
        })}
      </div>

      <fieldset className="space-y-2 rounded-xl border border-slate-200 p-3">
        <legend className="px-1 text-sm font-semibold">Target Provider</legend>

        <label className="flex items-center gap-2" title={!connected.notion ? "Notion not connected" : ""}>
          <input
            type="radio"
            name="provider"
            checked={target === "notion"}
            onChange={() => setTarget("notion")}
            disabled={!connected.notion}
          />
          <span>Notion</span>
        </label>

        <label className="flex items-center gap-2" title={!connected.jira ? "Jira not connected" : ""}>
          <input
            type="radio"
            name="provider"
            checked={target === "jira"}
            onChange={() => setTarget("jira")}
            disabled={!connected.jira}
          />
          <span>Jira</span>
        </label>

        <label
          className="flex items-center gap-2"
          title={!(connected.notion && connected.jira) ? "Both providers must be connected" : ""}
        >
          <input
            type="radio"
            name="provider"
            checked={target === "both"}
            onChange={() => setTarget("both")}
            disabled={!(connected.notion && connected.jira)}
          />
          <span>Both</span>
        </label>
      </fieldset>

      <div className="flex flex-wrap gap-3">
        <button
          onClick={submitExecution}
          disabled={!canExecute || loading}
          title={!canExecute ? blockedReason : ""}
          className="rounded-lg bg-ink px-4 py-2 font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {loading ? "Executing..." : "Confirm & Execute"}
        </button>
        <button
          disabled
          className="cursor-not-allowed rounded-lg border border-slate-300 px-4 py-2 font-semibold text-slate-400"
        >
          Edit Actions: Coming in v2
        </button>
        <button
          onClick={onCancel}
          className="rounded-lg border border-slate-300 px-4 py-2 font-semibold text-slate-700 hover:bg-slate-100"
        >
          Cancel
        </button>
      </div>
    </section>
  );
}