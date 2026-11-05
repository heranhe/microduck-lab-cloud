// One shape for "how is training going", whether the CPU on this Mac or a
// Colab GPU is doing it. The status bar and the training panel both draw it.

import { cloudBudgetLeft, cloudPhaseText, cloudProgressPercent, type CloudJob } from "./cloudCompute";

type Locale = "en" | "zh";

export type ProgressView = {
  where: "local" | "cloud";
  phase: string;
  /** Null until there is a real number; never a made-up 0%. */
  percent: number | null;
  count: string | null;
  remaining: string | null;
  score: number | null;
  extras: string[];
};

export function formatClock(seconds: number | null | undefined): string {
  const s = Math.max(0, Math.floor(seconds ?? 0));
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(Math.floor(s / 3600))}:${p(Math.floor(s % 3600 / 60))}:${p(s % 60)}`;
}

export type LocalProgressInput = {
  status: "training" | "stopping";
  steps: number;
  total: number;
  etaSeconds?: number | null;
  score?: number | null;
};

export function localProgressView(input: LocalProgressInput, locale: Locale): ProgressView {
  const saving = input.status === "stopping";
  const zh = locale === "zh";
  return {
    where: "local",
    phase: saving ? (zh ? "正在保存进度" : "Saving progress") : (zh ? "训练中" : "Training"),
    percent: input.total > 0 ? Math.min(100, 100 * input.steps / input.total) : null,
    count: `${input.steps.toLocaleString()} / ${input.total.toLocaleString()} ${zh ? "步" : "steps"}`,
    remaining: !saving && input.etaSeconds != null ? formatClock(input.etaSeconds) : null,
    score: input.score != null && Number.isFinite(input.score) ? input.score : null,
    extras: [],
  };
}

export function cloudProgressView(job: CloudJob, nowMs: number, locale: Locale): ProgressView {
  const zh = locale === "zh";
  const p = job.progress;
  const score = p?.reward != null ? Number.parseFloat(p.reward) : NaN;
  const budget = cloudBudgetLeft(job, nowMs);
  const extras: string[] = [];
  if (p?.gpu_util != null) extras.push(`GPU ${p.gpu_util}%`);
  if (budget != null) extras.push(`${zh ? "距时长上限" : "Time limit in"} ${formatClock(budget)}`);
  if (job.rate != null) extras.push(`${job.rate} CCU/h`);
  if (job.ccu_spent != null) extras.push(`${zh ? "已用" : "Spent"} ${job.ccu_spent} CCU`);
  return {
    where: "cloud",
    phase: cloudPhaseText(job, locale),
    percent: cloudProgressPercent(job),
    count: p?.iteration != null && p?.iterations ? `${p.iteration}/${p.iterations} ${zh ? "迭代" : "iterations"}` : null,
    remaining: p?.eta ?? null,
    score: Number.isFinite(score) ? score : null,
    extras,
  };
}

export type ScorePoint = { x: number; y: number };

/** Append one cloud sample. Restarting a job (or a lower iteration) starts a fresh curve. */
export function appendCloudScore(
  history: { jobId: string; points: ScorePoint[] },
  job: Pick<CloudJob, "id" | "progress"> | undefined,
): { jobId: string; points: ScorePoint[] } {
  if (!job) return history.points.length ? { jobId: "", points: [] } : history;
  const base = history.jobId === job.id ? history : { jobId: job.id, points: [] as ScorePoint[] };
  const x = job.progress?.iteration;
  const y = job.progress?.reward != null ? Number.parseFloat(job.progress.reward) : NaN;
  if (x == null || !Number.isFinite(x) || !Number.isFinite(y)) return base;
  const last = base.points.at(-1);
  if (last && x < last.x) return { jobId: job.id, points: [{ x, y }] };
  if (last && last.x === x) return last.y === y ? base : { jobId: job.id, points: [...base.points.slice(0, -1), { x, y }] };
  return { jobId: job.id, points: [...base.points, { x, y }] };
}
