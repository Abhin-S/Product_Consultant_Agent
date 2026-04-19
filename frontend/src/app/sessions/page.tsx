"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import api from "../../lib/axios";
import { Session } from "../../lib/types";

export default function SessionsPage() {
  const router = useRouter();

  const [sessions, setSessions] = useState<Session[]>([]);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchSessions = async () => {
      setLoading(true);
      setError("");
      try {
        const response = await api.get("/sessions", {
          params: { page, page_size: pageSize }
        });
        setSessions(response.data.items || []);
        setTotal(response.data.total || 0);
      } catch (err: any) {
        const detail = err?.response?.data?.detail;
        setError(typeof detail === "string" ? detail : "Failed to load sessions.");
      } finally {
        setLoading(false);
      }
    };

    fetchSessions();
  }, [page, pageSize]);

  const totalPages = Math.max(Math.ceil(total / pageSize), 1);

  return (
    <section className="space-y-4">
      <h1 className="text-2xl font-bold">Analysis Sessions</h1>

      {error && <p className="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>}

      <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white/95 shadow">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-100 text-slate-700">
            <tr>
              <th className="px-4 py-3">Date</th>
              <th className="px-4 py-3">Idea</th>
              <th className="px-4 py-3">Confidence Score</th>
              <th className="px-4 py-3">Eval Status</th>
              <th className="px-4 py-3">Actions Taken</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td className="px-4 py-4" colSpan={5}>
                  Loading sessions...
                </td>
              </tr>
            ) : sessions.length === 0 ? (
              <tr>
                <td className="px-4 py-4" colSpan={5}>
                  No sessions found.
                </td>
              </tr>
            ) : (
              sessions.map((session) => (
                <tr
                  key={session.id}
                  className="cursor-pointer border-t border-slate-100 hover:bg-slate-50"
                  onClick={() => router.push(`/sessions/${session.id}`)}
                >
                  <td className="px-4 py-3">{new Date(session.created_at).toLocaleString()}</td>
                  <td className="px-4 py-3">{session.idea_text.slice(0, 60)}</td>
                  <td className="px-4 py-3">{(session.confidence_score ?? 0).toFixed(2)}</td>
                  <td className="px-4 py-3">{session.ragas?.status || "not_requested"}</td>
                  <td className="px-4 py-3">{session.actions_taken}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between rounded-xl bg-white/90 px-4 py-3">
        <button
          disabled={page <= 1}
          onClick={() => setPage((p) => Math.max(1, p - 1))}
          className="rounded-md border border-slate-300 px-3 py-1 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Previous
        </button>
        <span className="text-sm text-slate-700">
          Page {page} of {totalPages}
        </span>
        <button
          disabled={page >= totalPages}
          onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
          className="rounded-md border border-slate-300 px-3 py-1 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </section>
  );
}