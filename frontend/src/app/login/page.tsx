"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import api from "../../lib/axios";

const ERROR_MESSAGES: Record<string, string> = {
  google_oauth_denied: "Google sign-in was cancelled.",
  google_sso_not_configured: "Google SSO is not configured on the backend.",
  invalid_google_oauth_state: "Login session expired. Please try again.",
  google_auth_failed: "Unable to verify Google sign-in. Please try again.",
  google_sso_disabled: "Google SSO is disabled on this server."
};

export default function LoginPage() {
  const router = useRouter();

  const [error, setError] = useState("");

  const googleLoginUrl = useMemo(() => {
    const base = process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") || "";
    return base ? `${base}/auth/google/login` : "";
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    localStorage.removeItem("token");

    const params = new URLSearchParams(window.location.search);
    const errorCode = params.get("error") || "";

    if (errorCode) {
      setError(ERROR_MESSAGES[errorCode] || "Google sign-in failed. Please try again.");
    }

    let cancelled = false;

    const checkSession = async () => {
      try {
        await api.get("/auth/me");
        if (!cancelled) {
          router.replace("/analyze");
        }
      } catch {
        // Stay on login page if no active session.
      }
    };

    checkSession();

    return () => {
      cancelled = true;
    };
  }, [router]);

  return (
    <div className="mx-auto mt-12 max-w-md rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-lg">
      <h1 className="mb-1 text-2xl font-bold text-ink">Sign In</h1>
      <p className="mb-6 text-sm text-slate-600">Use your Google account to access Product Consultant AI.</p>

      {error && <p className="mb-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {googleLoginUrl ? (
        <a
          href={googleLoginUrl}
          className="flex w-full items-center justify-center rounded-lg bg-ink px-4 py-2 font-semibold text-white transition hover:bg-slate-800"
        >
          Continue with Google
        </a>
      ) : (
        <p className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">
          Missing NEXT_PUBLIC_API_URL in frontend environment.
        </p>
      )}

      <p className="mt-4 text-xs text-slate-500">Password registration is disabled for this workspace.</p>
    </div>
  );
}