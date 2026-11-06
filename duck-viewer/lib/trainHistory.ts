// How a past training is described in the history list.

type Locale = "en" | "zh";

export type HistoryRow = {
  run: string;
  title: string;
  behavior: string | null;
  status: "done" | "stopped" | "failed" | "training";
  startedAt: number;
  finishedAt: number;
  durationS: number | null;
  steps: number | null;
  total: number | null;
  score: number | null;
  sps: number | null;
  servo: "feetech" | "official" | null;
  hasPolicy: boolean;
  stages?: number;
  chain?: string;
};

export const STATUS_ICON: Record<HistoryRow["status"], string> = { done: "✅", stopped: "⏸", failed: "❌", training: "▶" };

export function statusLabel(status: HistoryRow["status"], locale: Locale): string {
  const zh: Record<HistoryRow["status"], string> = { done: "完成", stopped: "已停止", failed: "失败", training: "训练中" };
  const en: Record<HistoryRow["status"], string> = { done: "Finished", stopped: "Stopped", failed: "Failed", training: "Training" };
  return (locale === "zh" ? zh : en)[status];
}

/** 1.6M, 655k, 800: the step count a person can compare at a glance. */
export function stepsText(steps: number | null | undefined): string | null {
  if (steps == null || !Number.isFinite(steps) || steps < 0) return null;
  if (steps >= 1e6) return `${(steps / 1e6).toFixed(steps >= 1e7 ? 0 : 1)}M`;
  if (steps >= 1e3) return `${Math.round(steps / 1e3)}k`;
  return String(Math.round(steps));
}

export function durationText(seconds: number | null | undefined): string | null {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return null;
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}m`;
}

export function agoText(epochSeconds: number, nowMs: number, locale: Locale): string {
  const zh = locale === "zh";
  const s = Math.max(0, Math.floor(nowMs / 1000 - epochSeconds));
  if (s < 90) return zh ? "刚刚" : "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return zh ? `${m} 分钟前` : `${m} min ago`;
  const h = Math.floor(m / 60);
  if (h < 48) return zh ? `${h} 小时前` : `${h} h ago`;
  const d = Math.floor(h / 24);
  return zh ? `${d} 天前` : `${d} d ago`;
}

/** The facts worth a line, skipping any the run never recorded rather than printing a made-up 0. */
export function historyFacts(row: HistoryRow, nowMs: number, locale: Locale): string[] {
  const zh = locale === "zh";
  const out: string[] = [];
  const duration = durationText(row.durationS);
  if (duration) out.push(`${zh ? "用时" : "took"} ${duration}`);
  const steps = stepsText(row.steps);
  const total = stepsText(row.total);
  if (steps) out.push(row.status === "done" || !total ? `${steps} ${zh ? "步" : "steps"}` : `${steps}/${total} ${zh ? "步" : "steps"}`);
  if (row.score != null) out.push(`${zh ? "得分" : "score"} ${row.score}`);
  if (row.stages && row.stages > 1) out.push(zh ? `${row.stages} 阶段` : `${row.stages} stages`);
  if (row.servo) out.push(row.servo === "feetech" ? (zh ? "飞特" : "Feetech") : (zh ? "官方 XL330" : "XL330"));
  out.push(agoText(row.finishedAt, nowMs, locale));
  return out;
}
