"use client";

// The training panel's action list. Every row says, on its own right-hand
// side, which ways it can be trained; picking one opens its settings under it.

import { useEffect, useState, type ReactNode } from "react";
import { useI18n } from "@/lib/i18n";
import { takeTeachFilter } from "@/lib/ui";
import {
  filterActionRows,
  supports,
  unavailableReason,
  type ActionFilter,
  type ActionRow,
  type TrainMode,
} from "@/lib/actionCatalog";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";
const COLLAPSED_ROWS = 8;

export type ActionPick = { rowId: string; mode: TrainMode };

type Props = {
  rows: ActionRow[];
  loading: boolean;
  /** Something is already training: starting a second one is explained, not attempted. */
  busyReason: string;
  picked: ActionPick | null;
  /** A one-click look at the action's best finished run, when it has one. */
  watch?: (row: ActionRow) => { title: string; run: () => void } | null;
  onPick: (pick: ActionPick | null) => void;
  renderConfig: (row: ActionRow, mode: TrainMode) => ReactNode;
};

const pill = (on: boolean, supported: boolean): React.CSSProperties => ({
  flexShrink: 0, borderRadius: 6, padding: "3px 8px", font: `11px ${mono}`, cursor: "pointer", whiteSpace: "nowrap",
  background: on ? "#1f4a63" : supported ? "#1c2230" : "transparent",
  color: on ? "#e8f2fb" : supported ? "#9fd0e6" : "#5d6572",
  border: on ? "1px solid #70b8d8" : supported ? "1px solid rgba(112,184,216,0.4)" : "1px dashed rgba(255,255,255,0.16)",
});

export function ActionPicker({ rows, loading, busyReason, picked, watch, onPick, renderConfig }: Props) {
  const { tr, locale } = useI18n();
  const [filter, setFilter] = useState<ActionFilter>(() => takeTeachFilter() ?? "all");
  const [query, setQuery] = useState("");
  const [showAll, setShowAll] = useState(false);
  const [notice, setNotice] = useState<{ rowId: string; text: string } | null>(null);

  useEffect(() => {
    const onFilter = (event: Event) => {
      const next = (event as CustomEvent<ActionFilter>).detail;
      if (next === "cpu" || next === "cloud" || next === "all") setFilter(next);
      takeTeachFilter();
      onPick(null);
    };
    window.addEventListener("microduck:teach-filter", onFilter);
    return () => window.removeEventListener("microduck:teach-filter", onFilter);
  }, [onPick]);

  const matches = filterActionRows(rows, filter, query);
  const searching = filter !== "all" || query.trim() !== "";
  const visible = searching || showAll ? matches : matches.slice(0, COLLAPSED_ROWS);
  const counts = {
    all: rows.length,
    cpu: rows.filter((r) => r.local).length,
    cloud: rows.filter((r) => r.cloud).length,
  };

  const click = (row: ActionRow, mode: TrainMode) => {
    if (busyReason) { setNotice({ rowId: row.id, text: busyReason }); return; }
    if (!supports(row, mode)) { setNotice({ rowId: row.id, text: unavailableReason(row, mode, locale) }); onPick(null); return; }
    setNotice(null);
    onPick(picked?.rowId === row.id && picked.mode === mode ? null : { rowId: row.id, mode });
  };

  const filters: [ActionFilter, string][] = [
    ["all", tr("All", "全部")],
    ["cpu", tr("CPU", "可 CPU")],
    ["cloud", tr("Cloud", "可云算力")],
  ];

  return (
    <div>
      <div style={{ color: "#8b93a3", fontSize: 10, marginBottom: 4 }}>
        {tr("Choose an action, then how to train it", "选择训练动作，再选训练方式")}
      </div>
      <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap", marginBottom: 6 }}>
        {filters.map(([key, label]) => (
          <button key={key} type="button" aria-pressed={filter === key}
            onClick={() => { setFilter(key); setShowAll(false); }}
            style={{
              borderRadius: 12, padding: "1px 8px", font: `10px ${mono}`, cursor: "pointer",
              background: filter === key ? "#263c4d" : "#1c2230", color: filter === key ? "#e8f2fb" : "#9fb4d8",
              border: `1px solid ${filter === key ? "#70b8d8" : "rgba(255,255,255,0.08)"}`,
            }}>
            {label} {counts[key]}
          </button>
        ))}
        <input value={query} onChange={(e) => setQuery(e.target.value)} aria-label={tr("Search actions", "搜索动作")}
          placeholder={tr("Search actions", "搜索动作")}
          style={{ flex: 1, minWidth: 80, boxSizing: "border-box", background: "#12151b", color: "#e8e6e1",
            border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6, padding: "3px 7px", font: `11px ${mono}`, outline: "none" }}/>
      </div>
      {loading && rows.length === 0 && (
        <div style={{ color: "#8b93a3", fontSize: 11, padding: "6px 0" }}>{tr("Loading actions…", "正在加载动作…")}</div>
      )}
      {!loading && matches.length === 0 && (
        <div style={{ color: "#8b93a3", fontSize: 11, padding: "6px 0" }}>
          {rows.length === 0
            ? tr("This robot has no trainable actions yet.", "这个机器人暂时没有可训练的动作。")
            : tr("No action matches. Clear the filter or the search.", "没有匹配的动作，清除筛选或搜索试试。")}
        </div>
      )}
      {visible.map((row) => {
        const open = picked?.rowId === row.id ? picked.mode : null;
        const best = watch?.(row) ?? null;
        return (
          <div key={row.id} style={{ borderTop: "1px solid rgba(255,255,255,0.07)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 0" }}>
              <span style={{ flex: 1, minWidth: 0 }}>
                <span style={{ display: "block", color: "#e8e6e1", fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {row.emoji} {row.title}
                </span>
                <span style={{ display: "block", color: "#7f8997", fontSize: 10 }}>{row.subtitle}</span>
              </span>
              {best && (
                <button type="button" onClick={best.run} aria-label={tr("Watch the best run", "看最佳结果")}
                  title={`${tr("Watch the best run so far", "看目前最好的结果")}: ${best.title}`}
                  style={{ flexShrink: 0, borderRadius: 6, padding: "3px 7px", font: `11px ${mono}`, cursor: "pointer",
                    background: "#1c2230", color: "#d8cfa0", border: "1px solid rgba(216,198,125,0.45)" }}>▶</button>
              )}
              <button type="button" aria-pressed={open === "cpu"} aria-disabled={!row.local || !!busyReason}
                title={row.local ? tr("Train on this computer's CPU", "用本机 CPU 训练") : unavailableReason(row, "cpu", locale)}
                onClick={() => click(row, "cpu")} style={pill(open === "cpu", !!row.local && !busyReason)}>
                {tr("CPU", "CPU 训练")}
              </button>
              <button type="button" aria-pressed={open === "cloud"} aria-disabled={!row.cloud || !!busyReason}
                title={row.cloud ? tr("Train on a cloud GPU", "用云端 GPU 训练") : unavailableReason(row, "cloud", locale)}
                onClick={() => click(row, "cloud")} style={pill(open === "cloud", !!row.cloud && !busyReason)}>
                {tr("Cloud", "云算力")}
              </button>
            </div>
            {notice?.rowId === row.id && (
              <div role="status" style={{ color: "#d8b46f", fontSize: 10, margin: "0 0 6px" }}>{notice.text}</div>
            )}
            {open && (
              <div style={{ margin: "0 0 8px", padding: 8, borderRadius: 8, border: "1px solid rgba(255,255,255,0.09)", background: "rgba(10,13,18,0.58)" }}>
                {renderConfig(row, open)}
              </div>
            )}
          </div>
        );
      })}
      {!searching && matches.length > COLLAPSED_ROWS && (
        <button type="button" onClick={() => setShowAll((v) => !v)}
          style={{ border: 0, background: "none", color: "#7db8d8", cursor: "pointer", padding: "6px 0", font: `10px ${mono}` }}>
          {showAll ? tr("Show fewer", "收起") : tr(`Show all ${matches.length} actions`, `显示全部 ${matches.length} 个动作`)}
        </button>
      )}
    </div>
  );
}
