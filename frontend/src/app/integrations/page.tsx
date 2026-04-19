"use client";

import { useEffect, useState } from "react";

import IntegrationCard from "../components/IntegrationCard";
import api from "../../lib/axios";
import { UserIntegrationOut } from "../../lib/types";

export default function IntegrationsPage() {
  const [integrations, setIntegrations] = useState<Record<string, UserIntegrationOut>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchIntegrations = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await api.get<UserIntegrationOut[]>("/integrations");
      const map: Record<string, UserIntegrationOut> = {};
      for (const row of response.data) {
        map[row.provider] = row;
      }
      setIntegrations(map);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Failed to load integrations.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchIntegrations();
  }, []);

  return (
    <section className="space-y-4">
      <h1 className="text-2xl font-bold">Integrations</h1>
      <p className="text-sm text-slate-600">
        Your token is encrypted with AES-256 before storage. It is never exposed via API.
      </p>

      {error && <p className="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>}
      {loading && <p>Loading integrations...</p>}

      <div className="grid gap-4 md:grid-cols-2">
        <IntegrationCard
          provider="notion"
          label="Notion"
          integration={integrations["notion"] || null}
          onRefresh={fetchIntegrations}
        />
        <IntegrationCard
          provider="jira"
          label="Jira"
          integration={integrations["jira"] || null}
          onRefresh={fetchIntegrations}
        />
      </div>
    </section>
  );
}