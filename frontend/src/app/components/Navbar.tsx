"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

function decodeUserEmail(token: string | null): string {
  if (!token) return "";

  try {
    const payloadPart = token.split(".")[1];
    if (!payloadPart) return "";

    const normalized = payloadPart.replace(/-/g, "+").replace(/_/g, "/");
    const json = atob(normalized);
    const payload = JSON.parse(json);
    return payload.sub || "";
  } catch {
    return "";
  }
}

export default function Navbar() {
  const router = useRouter();
  const pathname = usePathname();

  if (pathname === "/login" || pathname === "/register") {
    return null;
  }

  const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;
  const email = decodeUserEmail(token);

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
            onClick={() => {
              localStorage.removeItem("token");
              router.push("/login");
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