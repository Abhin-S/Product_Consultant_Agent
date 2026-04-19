"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

import api from "../../lib/axios";

export default function Navbar() {
  const router = useRouter();
  const pathname = usePathname();
  const [email, setEmail] = useState("");
  const hideNav = pathname === "/login" || pathname === "/register";

  useEffect(() => {
    let cancelled = false;

    if (hideNav) {
      setEmail("");
      return () => {
        cancelled = true;
      };
    }

    const fetchCurrentUser = async () => {
      try {
        const response = await api.get<{ email: string }>("/auth/me");
        if (!cancelled) {
          setEmail(response.data.email || "");
        }
      } catch {
        if (!cancelled) {
          setEmail("");
        }
      }
    };

    fetchCurrentUser();

    return () => {
      cancelled = true;
    };
  }, [hideNav, pathname]);

  if (hideNav) {
    return null;
  }

  return (
    <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        <div className="text-lg font-black tracking-tight">Product Consultant AI</div>

        <nav className="flex items-center gap-4 text-sm font-medium">
          <Link href="/analyze">Analyze</Link>
          <Link href="/sessions">Sessions</Link>
          <Link href="/integrations">Integrations</Link>
        </nav>

        <div className="flex items-center gap-3 text-sm">
          <span className="hidden text-slate-600 sm:inline">{email || "Not signed in"}</span>
          <button
            onClick={async () => {
              try {
                await api.post("/auth/logout");
              } catch {
                // Redirect regardless of logout response.
              } finally {
                router.replace("/login");
              }
            }}
            className="rounded-md border border-slate-300 px-3 py-1.5 hover:bg-slate-100"
          >
            Logout
          </button>
        </div>
      </div>
    </header>
  );
}