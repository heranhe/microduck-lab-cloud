"use client";

import { useI18n } from "@/lib/i18n";
import type { Tone } from "@/lib/trainResult";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";
const TONE: Record<Tone, { border: string; bg: string }> = {
  good: { border: "rgba(125,216,160,0.5)", bg: "rgba(24,57,40,0.55)" },
  warn: { border: "rgba(216,180,111,0.5)", bg: "rgba(60,48,24,0.5)" },
  bad: { border: "rgba(239,149,125,0.5)", bg: "rgba(70,30,24,0.5)" },
};

export type ResultAction = { label: string; onClick?: () => void; href?: string; primary?: boolean };

/** What a finished run leaves the person with: what happened, and the next click. */
export function ResultCard({ tone, headline, details, actions, note, onDismiss }: {
  tone: Tone; headline: string; details: string[]; actions: ResultAction[]; note?: string; onDismiss: () => void;
}) {
  const { tr } = useI18n();
  const button = (a: ResultAction): React.CSSProperties => ({
    borderRadius: 6, padding: "4px 10px", font: `11px ${mono}`, cursor: "pointer", textDecoration: "none",
    background: a.primary ? "#694318" : "#1c2230", color: a.primary ? "#ffe0a6" : "#9fd0e6",
    border: a.primary ? "1px solid #d39a48" : "1px solid rgba(112,184,216,0.4)", fontWeight: a.primary ? 700 : 400,
  });
  return (
    <div role="status" style={{ margin: "0 0 8px", padding: 8, borderRadius: 8, border: `1px solid ${TONE[tone].border}`, background: TONE[tone].bg }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
        <div style={{ flex: 1, fontWeight: 700 }}>{headline}</div>
        <button type="button" onClick={onDismiss} aria-label={tr("Dismiss", "关闭")}
          style={{ border: 0, background: "none", color: "#8b93a3", cursor: "pointer", font: `12px ${mono}`, padding: 0 }}>✕</button>
      </div>
      {details.length > 0 && <div style={{ color: "#aab3c0", fontSize: 11, margin: "3px 0" }}>{details.join(" · ")}</div>}
      {note && <div style={{ color: "#d8b46f", fontSize: 10, margin: "3px 0", whiteSpace: "pre-wrap" }}>{note}</div>}
      {actions.length > 0 && (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6 }}>
          {actions.map((a) => a.href
            ? <a key={a.label} href={a.href} download style={button(a)}>{a.label}</a>
            : <button key={a.label} type="button" onClick={a.onClick} style={button(a)}>{a.label}</button>)}
        </div>
      )}
    </div>
  );
}
