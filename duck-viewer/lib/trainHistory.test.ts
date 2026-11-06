import { describe, expect, it } from "vitest";
import { agoText, durationText, historyFacts, statusLabel, stepsText, type HistoryRow } from "./trainHistory";

const row = (over: Partial<HistoryRow> = {}): HistoryRow => ({
  run: "r", title: "转圈", behavior: "spin", status: "done", startedAt: 0, finishedAt: 1_000_000 - 3 * 3600,
  durationS: 1105.8, steps: 8_000_000, total: 8_000_000, score: 2238.93, sps: 7000, servo: "feetech", hasPolicy: true, ...over,
});
const NOW = 1_000_000 * 1000;

describe("number formatting", () => {
  it("shortens steps and refuses a missing or negative one", () => {
    expect(stepsText(1_641_026)).toBe("1.6M");
    expect(stepsText(8_000_000)).toBe("8.0M");
    expect(stepsText(12_000_000)).toBe("12M");
    expect(stepsText(655_872)).toBe("656k");
    expect(stepsText(800)).toBe("800");
    for (const bad of [null, undefined, -1, Number.NaN]) expect(stepsText(bad as number)).toBeNull();
  });
  it("writes durations a person can read, and nothing for an unknown one", () => {
    expect(durationText(44)).toBe("44s");
    expect(durationText(362.3)).toBe("6m02s");
    expect(durationText(1105.8)).toBe("18m26s");
    expect(durationText(4000)).toBe("1h06m");
    expect(durationText(0)).toBe("0s");
    expect(durationText(null)).toBeNull();
    expect(durationText(-5)).toBeNull();
  });
  it("says how long ago in the person's language", () => {
    expect(agoText(NOW / 1000 - 30, NOW, "zh")).toBe("刚刚");
    expect(agoText(NOW / 1000 - 600, NOW, "zh")).toBe("10 分钟前");
    expect(agoText(NOW / 1000 - 3 * 3600, NOW, "en")).toBe("3 h ago");
    expect(agoText(NOW / 1000 - 5 * 86400, NOW, "zh")).toBe("5 天前");
    expect(agoText(NOW / 1000 + 100, NOW, "zh")).toBe("刚刚");
  });
});

describe("historyFacts", () => {
  it("lists time, steps, score, stages, servo and recency for a finished chain", () => {
    expect(historyFacts(row({ stages: 5 }), NOW, "zh")).toEqual(["用时 18m26s", "8.0M 步", "得分 2238.93", "5 阶段", "飞特", "3 小时前"]);
  });
  it("shows progress against the total for a run that did not finish", () => {
    expect(historyFacts(row({ status: "stopped", steps: 655_872, total: 1_000_000 }), NOW, "zh")).toContain("656k/1.0M 步");
  });
  it("skips what the run never recorded instead of printing zero", () => {
    const facts = historyFacts(row({ durationS: null, score: null, steps: null, servo: null }), NOW, "zh");
    expect(facts).toEqual(["3 小时前"]);
    expect(historyFacts(row({ score: 0 }), NOW, "zh")).toContain("得分 0");
  });
  it("labels each outcome in both languages", () => {
    expect(statusLabel("failed", "zh")).toBe("失败");
    expect(statusLabel("training", "en")).toBe("Training");
  });
});
