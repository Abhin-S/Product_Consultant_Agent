"use client";

import { useState } from "react";

import api from "../../lib/axios";
import { IntegrationConnectRequest, UserIntegrationOut } from "../../lib/types";

type Props = {
  provider: "notion" | "jira";
  label: string;
  integration: UserIntegrationOut | null;
  onRefresh: () => Promise<void> | void;
};

function mask(value: string | null | undefined): string {
  if (!value) return "-";
  return `${value.slice(0, 6)}***`;
}

export default function IntegrationCard({ provider, label, integration, onRefresh }: Props) {
  const [showForm, setShowForm] = useState(false);
  const [token, setToken] = useState("");
  const [notionDatabaseId, setNotionDatabaseId] = useState("");
  const [notionParentPageId, setNotionParentPageId] = useState("");
  const [jiraUrl, setJiraUrl] = useState("");
  const [jiraProjectKey, setJiraProjectKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const connect = async () => {
    setLoading(true);
    setError("");
    try {
      const payload: IntegrationConnectRequest = {
        provider,
        access_token: token
      };

      if (provider === "notion") {
        const trimmedDatabaseId = notionDatabaseId.trim();
        const trimmedParentPageId = notionParentPageId.trim();
        if (trimmedDatabaseId) {
          payload.database_id = trimmedDatabaseId;
        }
        if (trimmedParentPageId) {
          payload.workspace_id = trimmedParentPageId;
        }
      } else {
        payload.workspace_id = jiraUrl;
        payload.database_id = jiraProjectKey;
      }

      await api.post("/integrations/connect", payload);
      setShowForm(false);
      setToken("");
      setNotionDatabaseId("");
      setNotionParentPageId("");
      setJiraUrl("");
      setJiraProjectKey("");
      await onRefresh();
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Failed to save integration.");
    } finally {
      setLoading(false);
    }
  };

  const disconnect = async () => {
    setLoading(true);
    setError("");
    try {
      await api.delete(`/integrations/${provider}`);
      await onRefresh();
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Failed to disconnect integration.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <article className="space-y-3 rounded-2xl border border-slate-200 bg-white/95 p-4 shadow">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">{label}</h2>
        <span
          className={`rounded-full px-2 py-1 text-xs font-medium ${
            integration ? "bg-green-100 text-green-700" : "bg-slate-200 text-slate-700"
          }`}
        >
          {integration ? "Connected" : "Not Connected"}
        </span>
      </div>

      {error && <p className="rounded bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {integration ? (
        <div className="space-y-2 text-sm">
          <p>{provider === "notion" ? "report_page_id" : "workspace_id"}: {mask(integration.workspace_id)}</p>
          <p>database_id: {mask(integration.database_id)}</p>
          <button
            onClick={disconnect}
            disabled={loading}
            className="rounded-lg border border-red-300 px-3 py-2 text-red-700 hover:bg-red-50 disabled:opacity-50"
          >
            Disconnect
          </button>
        </div>
      ) : (
        <>
          {!showForm ? (
            <button
              onClick={() => setShowForm(true)}
              className="rounded-lg bg-ink px-3 py-2 text-white hover:bg-slate-800"
            >
              Connect
            </button>
          ) : (
            <div className="space-y-3">
              <label className="block">
                <span className="mb-1 block text-sm">API Token</span>
                <input
                  type="password"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder={provider === "notion" ? "secret_..." : "Atlassian API token"}
                  className="w-full rounded border border-slate-300 px-3 py-2 text-sm"
                />
              </label>

              {provider === "notion" ? (
                <>
                  <label className="block">
                    <span className="mb-1 block text-sm">Database ID or URL (optional)</span>
                    <input
                      type="text"
                      value={notionDatabaseId}
                      onChange={(e) => setNotionDatabaseId(e.target.value)}
                      placeholder="Paste Notion database ID or URL"
                      className="w-full rounded border border-slate-300 px-3 py-2 text-sm"
                    />
                  </label>

                  <label className="block">
                    <span className="mb-1 block text-sm">Report Page ID or URL (optional)</span>
                    <input
                      type="text"
                      value={notionParentPageId}
                      onChange={(e) => setNotionParentPageId(e.target.value)}
                      placeholder="Paste Notion page ID or URL"
                      className="w-full rounded border border-slate-300 px-3 py-2 text-sm"
                    />
                    <p className="mt-1 text-xs text-slate-500">
                      We auto-parse Notion URLs into IDs and do not auto-create pages/databases in teamspaces.
                    </p>
                  </label>
                </>
              ) : (
                <>
                  <label className="block">
                    <span className="mb-1 block text-sm">Jira URL</span>
                    <input
                      type="text"
                      value={jiraUrl}
                      onChange={(e) => setJiraUrl(e.target.value)}
                      placeholder="https://yourteam.atlassian.net"
                      className="w-full rounded border border-slate-300 px-3 py-2 text-sm"
                    />
                  </label>

                  <label className="block">
                    <span className="mb-1 block text-sm">Project Key</span>
                    <input
                      type="text"
                      value={jiraProjectKey}
                      onChange={(e) => setJiraProjectKey(e.target.value)}
                      placeholder="PROJ"
                      className="w-full rounded border border-slate-300 px-3 py-2 text-sm"
                    />
                  </label>
                </>
              )}

              <button
                onClick={connect}
                disabled={loading || !token}
                className="rounded-lg bg-pine px-3 py-2 text-white hover:bg-teal-700 disabled:opacity-50"
              >
                Save Connection
              </button>
            </div>
          )}
        </>
      )}
    </article>
  );
}