"use client";

// Past trainings, newest first: what ran, how long it took, how far it got and what it scored.

import { useEffect, useState } from "react";
import { useI18n } from "@/lib/i18n";
import { LAB_HTTP } from "@/lib/lab";
import { requestPolicyOpen } from "@/lib/ui";
import { STATUS_ICON, historyFacts, statusLabel, type HistoryRow } from "@/lib/trainHistory";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";
const small: React.CSSProperties = {
  background: "#1c2230", color: "#9fd0e6", border: "1px solid rgba(112,184,216,0.4)",
  borderRadius: 6, padding: "2px 7px", font: `10px ${mono}`, cursor: "pointer", textDecoration: "none",
};

export function TrainHistory({ online, refreshKey, onWatch }: {
  online: boolean;
  /** Changes when a run ends, so the list picks up the new row without waiting. */
  refreshKey: number;
  onWatch: (row: HistoryRow) => void;
}) {
  const { tr, locale } = useI18n();
  const [open, setOpen] = useState(false);
  const [rows, setRows] = useState<HistoryRow[] | null>(null);
  const [error, setError] = useState("");
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!open || !online) return;
    let stale = false;
    const load = () => fetch(`${LAB_HTTP}/train/history?limit=20`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: { history: HistoryRow[] }) => { if (!stale) { setRows(d.history); setNow(Date.now()); setError(""); } })
      .catch((e) => { if (!stale) setError(e instanceof Error && e.message.includes("404")
        ? tr("This lab is too old to keep a history. Restart duck-lab.", "这个实验室版本较旧，没有训练历史，请重启 duck-lab。")
        : tr("Could not load the history.", "无法加载训练历史。")); });
    load();
    const timer = window.setInterval(load, 20000);
    return () => { stale = true; clearInterval(timer); };
  }, [open, online, refreshKey]); // eslint-disable-line react-hooks/exhaustive-deps -- tr changes identity each render.

  return (
    <div style={{ margin: "8px 0 0" }}>
      <button type="button" aria-expanded={open} onClick={() => setOpen((v) => !v)}
        style={{ border: 0, background: "none", color: "#d8c97d", cursor: "pointer", padding: 0, font: `11px ${mono}` }}>
        {open ? "▾" : "▸"} 📜 {tr("Training history", "训练历史")}
      </button>
      {open && (
        <div style={{ marginTop: 6 }}>
          {error && <div role="alert" style={{ color: "#ef957d", fontSize: 10 }}>{error}</div>}
          {!error && rows === null && <div style={{ color: "#8b93a3", fontSize: 11 }}>{tr("Loading…", "加载中…")}</div>}
          {rows && rows.length === 0 && <div style={{ color: "#8b93a3", fontSize: 11 }}>{tr("No trainings yet.", "还没有训练记录。")}</div>}
          {rows?.map((row) => (
            <div key={row.run} style={{ borderTop: "1px solid rgba(255,255,255,0.07)", padding: "6px 0" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {STATUS_ICON[row.status]} {row.title}
                  <span style={{ color: "#8b93a3", fontSize: 10 }}> · {statusLabel(row.status, locale)}</span>
                </span>
                {row.hasPolicy && row.status !== "training" && (
                  <>
                    <button type="button" style={small} onClick={() => onWatch(row)}>▶ {tr("Stage", "舞台")}</button>
                    <a style={small} href={`${LAB_HTTP}/runs/${encodeURIComponent(row.run)}/policy.onnx`} download>⤓ ONNX</a>
                  </>
                )}
              </div>
              <div style={{ color: "#8b93a3", fontSize: 10, marginTop: 2 }}>{historyFacts(row, now, locale).join(" · ")}</div>
            </div>
          ))}
          {rows && rows.length > 0 && (
            <button type="button" onClick={requestPolicyOpen}
              style={{ border: 0, background: "none", color: "#7db8d8", cursor: "pointer", padding: "6px 0 0", font: `10px ${mono}` }}>
              🧠 {tr("All runs and checkpoints in Models ↗", "在“模型”里查看全部结果和检查点 ↗")}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
