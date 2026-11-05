// What the training panel lists: one row per action, with the two ways it can
// be trained. Kept free of React so the rules (which action works where, what
// the first cloud run asks for) can be tested without a page.

import type { Catalog, Plan, Recipe } from "./plans";
import { cloudTaskFor } from "./cloudCompute";

export type TrainMode = "cpu" | "cloud";
export type ActionFilter = "all" | "cpu" | "cloud";
type Locale = "en" | "zh";

export type ActionRow = {
  id: string;
  robot: string;
  emoji: string;
  title: string;
  subtitle: string;
  /** Text sent to /teach. A bare behavior id matches exactly on the server. */
  local: { behavior: string; text: string } | null;
  cloud: { task: string } | null;
  recommended: boolean;
};

type Suggestion = { text: string; behavior: string; title?: string; emoji?: string };

const ZH_TITLES: Record<string, string> = {
  one_leg: "单脚站立", stand: "保持站立", crouch: "下蹲", spin: "原地转圈", headstand: "头倒立",
  deep_squat: "深蹲", backflip: "后空翻", getup: "从地上站起来", run: "向前奔跑",
};

const LOCAL_SUB = { en: "Built-in action", zh: "内置动作" };
const CLOUD_SUB = { en: "Official task · needs a GPU", zh: "官方任务 · 需要 GPU" };

// A lab that has not answered yet leaves only the suggestion texts.
const ZH_BY_TEXT: Record<string, string> = {
  "stand still": "保持站立", "stand on one leg": "单脚站立", "crouch down": "下蹲",
  "spin in place": "原地转圈", "do a headstand": "头倒立",
};

function pickTitle(id: string, fallback: string, locale: Locale): string {
  return (locale === "zh" && (ZH_TITLES[id] ?? ZH_BY_TEXT[fallback.toLocaleLowerCase()])) || fallback;
}

/** Rows for one robot from the plan catalog (`GET /plans/templates`). */
export function buildActionRows(
  catalog: Catalog,
  robotId: string,
  teach: Suggestion[] | undefined,
  locale: Locale,
): ActionRow[] {
  const recipes = new Map<string, Recipe>(catalog.behaviors.map((b) => [b.id, b]));
  const suggested = new Set((teach ?? []).map((s) => s.behavior));
  const rows = catalog.templates
    .filter((t: Plan) => t.robot === robotId && (t.local || t.cloud))
    .map((t: Plan): ActionRow => {
      const behavior = t.local?.behavior;
      const recipe = behavior ? recipes.get(behavior) : undefined;
      const text = behavior ? ((teach ?? []).find((s) => s.behavior === behavior)?.text ?? behavior) : "";
      return {
        id: t.id ?? `${t.robot}-${behavior ?? t.cloud?.task}`,
        robot: t.robot,
        emoji: recipe?.emoji ?? (t.cloud && !t.local ? "☁" : "🎯"),
        title: behavior ? pickTitle(behavior, recipe?.title ?? t.title, locale) : t.title,
        subtitle: t.cloud && !t.local ? CLOUD_SUB[locale] : LOCAL_SUB[locale],
        local: behavior ? { behavior, text } : null,
        cloud: t.cloud ? { task: t.cloud.task } : null,
        recommended: (!!behavior && suggested.has(behavior)) || (!!t.cloud && !t.local),
      };
    });
  // Stable: recommended first, catalog order otherwise.
  return [...rows.filter((r) => r.recommended), ...rows.filter((r) => !r.recommended)];
}

/** An old lab without /plans/templates still has its suggestion chips. */
export function fallbackActionRows(robotId: string, teach: Suggestion[] | undefined, locale: Locale): ActionRow[] {
  const rows: ActionRow[] = (teach ?? []).map((s) => ({
    // The built-in placeholder list has no behavior ids yet; its text still matches on the server.
    id: `${robotId}-${s.behavior || s.text}`,
    robot: robotId,
    emoji: s.emoji || "🎯",
    title: pickTitle(s.behavior, s.title ?? s.text, locale),
    subtitle: LOCAL_SUB[locale],
    local: { behavior: s.behavior || s.text, text: s.text },
    cloud: null,
    recommended: true,
  }));
  const official: [string, string, string][] = [
    ["official_velocity", "官方行走", "Official walking"],
    ["official_velstand", "官方行走与跌倒恢复", "Official walking and fall recovery"],
  ];
  for (const [behavior, zh, en] of official) {
    const task = cloudTaskFor(robotId, behavior);
    if (task) rows.push({
      id: `${robotId}-${behavior}`, robot: robotId, emoji: "☁", title: locale === "zh" ? zh : en,
      subtitle: CLOUD_SUB[locale], local: null, cloud: { task }, recommended: true,
    });
  }
  return rows;
}

export function filterActionRows(rows: ActionRow[], filter: ActionFilter, query: string): ActionRow[] {
  const q = query.trim().toLocaleLowerCase();
  return rows.filter((r) =>
    (filter === "all" || (filter === "cpu" ? !!r.local : !!r.cloud))
    && (!q || `${r.title} ${r.subtitle} ${r.local?.behavior ?? ""}`.toLocaleLowerCase().includes(q)));
}

export function supports(row: ActionRow, mode: TrainMode): boolean {
  return mode === "cpu" ? !!row.local : !!row.cloud;
}

/** Why a greyed button is grey. Shown on click, because a dead button explains nothing. */
export function unavailableReason(row: ActionRow, mode: TrainMode, locale: Locale): string {
  if (mode === "cloud") {
    return locale === "zh"
      ? "这个动作还没有云端版本，可以用 CPU 训练。"
      : "This action has no cloud version yet. Train it on the CPU.";
  }
  return row.cloud
    ? (locale === "zh" ? "这是官方 GPU 任务，只能在云端训练。" : "This is an official GPU task. It trains in the cloud only.")
    : (locale === "zh" ? "这个动作暂时不能在本机训练。" : "This action can't train on this computer yet.");
}

export type CloudPreset = "trial" | "full";
export type CloudSettings = { iterations: number; maxMinutes: number; maxCcu: string };

/** The first run proves the whole chain cheaply; a full run is one click away. */
export const CLOUD_PRESETS: Record<CloudPreset, CloudSettings> = {
  trial: { iterations: 100, maxMinutes: 30, maxCcu: "5" },
  full: { iterations: 1000, maxMinutes: 180, maxCcu: "" },
};

/** A CCU limit means nothing on the free tier, so it is only sent with a balance. */
export function effectiveCcuLimit(maxCcu: string, balance: number | null | undefined): number | null {
  const value = Number(maxCcu);
  return typeof balance === "number" && balance > 0 && Number.isFinite(value) && value > 0 ? value : null;
}

/** Anything on the account can be drawn on, even a T4; only a known zero is free. */
export function mayUsePaidUnits(balance: number | null | undefined): boolean {
  return !(typeof balance === "number" && balance <= 0);
}

export function cloudConfirmText(
  o: { title: string; gpu: string; iterations: number; maxMinutes: number; ccuLimit: number | null; balance: number | null | undefined },
  locale: Locale,
): string {
  const balance = o.balance ?? (locale === "zh" ? "未知" : "unknown");
  return locale === "zh"
    ? `将在 Colab（${o.gpu}）上训练「${o.title}」：${o.iterations} 次迭代，最长 ${o.maxMinutes} 分钟，`
      + `${o.ccuLimit != null ? `最多消耗 ${o.ccuLimit} 计算单元` : "计算单元不设上限"}（当前余额 ${balance}）。`
      + "到达上限会自动保存并释放算力，训练期间请勿合上电脑。确定开始吗？"
    : `Train "${o.title}" on Colab (${o.gpu}): ${o.iterations} iterations, up to ${o.maxMinutes} minutes, `
      + `${o.ccuLimit != null ? `at most ${o.ccuLimit} compute units` : "no compute-unit limit"} (balance ${balance}). `
      + "It saves and releases the GPU at the limit. Keep this Mac open while it runs. Start?";
}
