"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import api from "./axios";

type AuthStatus = "checking" | "authenticated";

export default function useRequireAuth(): AuthStatus {
  const router = useRouter();
  const [status, setStatus] = useState<AuthStatus>("checking");

  useEffect(() => {
    let cancelled = false;

    const checkSession = async () => {
      try {
        await api.get("/auth/me");
        if (!cancelled) {
          setStatus("authenticated");
        }
      } catch {
        if (!cancelled) {
          router.replace("/login");
        }
      }
    };

    checkSession();

    return () => {
      cancelled = true;
    };
  }, [router]);

  return status;
}