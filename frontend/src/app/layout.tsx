import type { Metadata } from "next";

import Navbar from "./components/Navbar";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Product Consultant Agent",
  description: "RAG-powered product consulting with action execution"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <main className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 lg:px-8">{children}</main>
      </body>
    </html>
  );
}