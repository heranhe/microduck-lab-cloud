import type { Locale } from "./i18n";

/** Keep protocol names and model/state identifiers intact; translate only UI prose. */
export function simText(locale: Locale, en: string, zh: string): string {
  return locale === "zh" ? zh : en;
}

export function simStateName(value: string, locale: Locale): string {
  if (locale === "en") return value;
  const names: Record<string, string> = {
    fresh: "新鲜", stale: "过期", never: "从未收到", nothing: "无目标",
    auto: "自动", manual: "手动", live: "实时", offline: "离线",
  };
  return names[value] ?? value;
}

export function simBrainLabel(value: string, locale: Locale): string {
  if (locale === "en") return value;
  const names: Record<string, string> = {
    wander: "wander（漫游）", follow: "follow（跟随）", chase: "chase（追球）",
    tidy: "tidy（整理）", script: "script（脚本）", manual: "manual（手动）",
    cruise: "cruise（巡航）", steer: "steer（转向）", spin: "spin（旋转）",
    stuck: "stuck（卡住）", blind: "blind（失明）",
  };
  return names[value] ?? value;
}
