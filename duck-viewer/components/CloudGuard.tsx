"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LAB_HTTP } from "@/lib/lab";
import { cloudPhaseText, needsRelease, type CloudJob, type CloudProvider } from "@/lib/cloudCompute";

/** Back to the lab's cloud panel, on the same lab this page is talking to. */
function labHref() {
  const lab = new URLSearchParams(window.location.search).get("lab");
  return `/?cloud=1${lab ? `&lab=${encodeURIComponent(lab)}` : ""}`;
}

/**
 * The lab page has the full status bar. Every other page gets this: a paid GPU
 * must stay visible and stoppable wherever the person navigated to.
 */
export function CloudGuard() {
  const pathname = usePathname();
  const onLabPage = pathname === "/";
  const [jobs, setJobs] = useState<CloudJob[]>([]);
  const [unreachable, setUnreachable] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (onLabPage) return;
    let alive = true;
    const load = async () => {
      const results = await Promise.allSettled((["colab", "hf"] as CloudProvider[]).map(async (platform) => {
        const response = await fetch(`${LAB_HTTP}/cloud/${platform}/jobs`, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json() as { jobs?: CloudJob[] };
        return (data.jobs ?? []).map((job) => ({ ...job, platform }));
      }));
      if (!alive) return;
      const reachable = results.flatMap((result) => result.status === "fulfilled" ? [result.value] : []);
      // Keep the last known jobs if the lab vanished: that is exactly when a
      // running GPU must not silently disappear from the screen.
      if (reachable.length) setJobs(reachable.flat());
      setUnreachable(reachable.length === 0);
    };
    void load();
    const poll = window.setInterval(() => void load(), 15000);
    return () => { alive = false; clearInterval(poll); };
  }, [onLabPage]);

  if (onLabPage) return null;
  const active = jobs.filter(needsRelease);
  if (!active.length) return null;

  const disconnect = async () => {
    if (!window.confirm("立即断开云算力？云端尚未下载的训练进度都会丢失。\nDisconnect now? Progress not yet downloaded will be lost.")) return;
    setBusy(true); setError("");
    try {
      await Promise.all([...new Set(active.map((job) => job.platform))].map(async (platform) => {
        const response = await fetch(`${LAB_HTTP}/cloud/${platform}/stop-all`, { method: "POST" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
      }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div role="status" style={{
      position: "fixed", right: 12, bottom: 12, zIndex: 9999, display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
      maxWidth: "calc(100vw - 24px)", padding: "7px 10px", borderRadius: 8, boxSizing: "border-box",
      border: "1px solid rgba(239,126,100,.55)", background: "rgba(24,28,36,.96)",
      color: "#dbe7ed", font: "11px ui-monospace, Menlo, monospace",
    }}>
      <span>☁ {active[0].device ?? active[0].gpu} · {cloudPhaseText(active[0], "zh")}{unreachable ? " · 无法连接后端，状态可能已过期" : ""}</span>
      <Link href={labHref()} style={{ color: "#7db8d8" }}>查看</Link>
      <button type="button" disabled={busy || unreachable} onClick={disconnect} style={{
        border: "1px solid rgba(239,126,100,.55)", borderRadius: 6, background: "none", color: "#ef957d",
        padding: "2px 8px", cursor: busy || unreachable ? "default" : "pointer", font: "inherit",
      }}>{busy ? "断开中…" : "■ 断开云算力"}</button>
      {error && <span style={{ color: "#ef957d" }}>{error}</span>}
    </div>
  );
}
