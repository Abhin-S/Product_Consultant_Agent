"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function RegisterPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/login");
  }, [router]);

  return (
    <div className="mx-auto mt-12 max-w-md rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-lg">
      <h1 className="mb-1 text-2xl font-bold text-ink">Registration Disabled</h1>
      <p className="text-sm text-slate-600">Use Google SSO from the login page.</p>
    </div>
  );
}