"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

import api from "../../lib/axios";

export default function RegisterPage() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [errors, setErrors] = useState<{ email?: string; password?: string; confirm?: string; form?: string }>(
    {}
  );
  const [loading, setLoading] = useState(false);

  const validate = () => {
    const nextErrors: { email?: string; password?: string; confirm?: string; form?: string } = {};

    if (!email.includes("@")) {
      nextErrors.email = "Enter a valid email address.";
    }
    if (password.length < 6) {
      nextErrors.password = "Password must be at least 6 characters.";
    }
    if (password !== confirmPassword) {
      nextErrors.confirm = "Passwords do not match.";
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!validate()) {
      return;
    }

    setLoading(true);
    try {
      await api.post("/auth/register", { email, password });
      router.push("/login");
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setErrors({ form: typeof detail === "string" ? detail : "Registration failed." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto mt-12 max-w-md rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-lg">
      <h1 className="mb-1 text-2xl font-bold text-ink">Register</h1>
      <p className="mb-6 text-sm text-slate-600">Create your account to start analyzing ideas.</p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <label className="block">
          <span className="mb-1 block text-sm font-medium">Email</span>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="w-full rounded-lg border border-slate-300 px-3 py-2 outline-none focus:border-sky"
          />
          {errors.email && <span className="mt-1 block text-xs text-red-700">{errors.email}</span>}
        </label>

        <label className="block">
          <span className="mb-1 block text-sm font-medium">Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="w-full rounded-lg border border-slate-300 px-3 py-2 outline-none focus:border-sky"
          />
          {errors.password && <span className="mt-1 block text-xs text-red-700">{errors.password}</span>}
        </label>

        <label className="block">
          <span className="mb-1 block text-sm font-medium">Confirm Password</span>
          <input
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
            className="w-full rounded-lg border border-slate-300 px-3 py-2 outline-none focus:border-sky"
          />
          {errors.confirm && <span className="mt-1 block text-xs text-red-700">{errors.confirm}</span>}
        </label>

        {errors.form && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{errors.form}</p>}

        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-lg bg-ink px-4 py-2 font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {loading ? "Creating account..." : "Create Account"}
        </button>
      </form>
    </div>
  );
}