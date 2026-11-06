// What to tell the person when a run ends: that it ended, how, and what to do next.

import type { CloudJob } from "./cloudCompute";

type Locale = "en" | "zh";
export type Finish = "done" | "stopped" | "failed";
export type Tone = "good" | "warn" | "bad";

const RUNNING = new Set(["training", "stopping"]);
const FINISHED = new Set<string>(["done", "stopped", "failed"]);

export function isFinish(status: string | undefined): status is Finish {
  return !!status && FINISHED.has(status);
}

/** The runs that just went from running to finished since the last look. */
export function justFinished(
  previous: Record<string, string>,
  jobs: { runName: string; status: string }[],
): { runName: string; status: Finish }[] {
  return jobs
    .filter((j) => isFinish(j.status) && RUNNING.has(previous[j.runName] ?? ""))
    .map((j) => ({ runName: j.runName, status: j.status as Finish }));
}

export function finishHeadline(status: Finish, title: string, locale: Locale): { tone: Tone; text: string } {
  const zh = locale === "zh";
  if (status === "done") return { tone: "good", text: zh ? `✅ 训练完成 · ${title}` : `✅ Training finished · ${title}` };
  if (status === "stopped") return { tone: "warn", text: zh ? `⏸ 已保存并停止 · ${title}` : `⏸ Saved and stopped · ${title}` };
  return { tone: "bad", text: zh ? `❌ 训练失败 · ${title}` : `❌ Training failed · ${title}` };
}

export function toastFor(status: Finish, title: string, locale: Locale): string {
  const zh = locale === "zh";
  if (status === "done") return zh ? `✅ 训练完成：${title}，可以在训练面板里放到舞台看看` : `✅ Training finished: ${title}. Open Training to try it on stage.`;
  if (status === "stopped") return zh ? `⏸ 已保存并停止：${title}` : `⏸ Saved and stopped: ${title}`;
  return zh ? `❌ 训练失败：${title}，请在训练面板查看原因` : `❌ Training failed: ${title}. Open Training for the reason.`;
}

export function elapsedText(seconds: number | null | undefined): string | null {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return null;
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m${String(s % 60).padStart(2, "0")}s` : `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}m`;
}

/** The newest finished cloud job nobody has dismissed, within the last day. */
export function cloudResultToShow(jobs: CloudJob[], dismissed: string[], nowMs: number): CloudJob | null {
  const DAY = 24 * 3600;
  return jobs
    .filter((j) => (j.state === "done" || j.state === "stopped" || j.state === "failed")
      && j.released === true && !dismissed.includes(j.id)
      && nowMs / 1000 - (j.released_at ?? j.started ?? 0) < DAY)
    .sort((a, b) => (b.released_at ?? b.started ?? 0) - (a.released_at ?? a.started ?? 0))[0] ?? null;
}
