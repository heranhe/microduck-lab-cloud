import { describe, expect, it } from "vitest";
import {
  CLOUD_PRESETS,
  buildActionRows,
  cloudConfirmText,
  effectiveCcuLimit,
  fallbackActionRows,
  filterActionRows,
  mayUsePaidUnits,
  supports,
  unavailableReason,
} from "./actionCatalog";
import type { Catalog, Plan, Recipe } from "./plans";

const plan = (over: Partial<Plan>): Plan => ({
  version: 1, title: "t", description: "", success: "", robot: "microduck", source: "template",
  local: null, cloud: null, ...over,
});
const local = (behavior: string) => ({ behavior, steps: 1, weights: {}, stageSteps: {}, stageWeights: {} });
const cloud = (task: string) => ({ task, iterations: 1, envs: 1, gpu: "T4", flavor: "" });
const recipe = (id: string, title: string, emoji = "🦩"): Recipe => ({
  id, robot: "microduck", emoji, title, description: "", howItLearns: "", successMetric: "",
  defaultSteps: 1, terms: [], availableTerms: [], curriculum: [],
});

const catalog: Catalog = {
  templates: [
    plan({ id: "template-spin", title: "Spin in place", local: local("spin") }),
    plan({ id: "template-one_leg", title: "Stand on one leg", local: local("one_leg") }),
    plan({ id: "template-run", title: "Run forward", local: local("run") }),
    plan({ id: "template-official", title: "官方行走", cloud: cloud("Mjlab-Velocity-Flat-MicroDuck") }),
    plan({ id: "template-g1", title: "Stand still (G1)", robot: "g1", local: local("g1_stand") }),
  ],
  behaviors: [recipe("spin", "Spin in place", "🌀"), recipe("one_leg", "Stand on one leg"), recipe("run", "Run forward", "🏃")],
};
const teach = [{ text: "stand on one leg", behavior: "one_leg" }, { text: "spin in place", behavior: "spin" }];

describe("buildActionRows", () => {
  const rows = buildActionRows(catalog, "microduck", teach, "zh");

  it("lists only the chosen robot's actions", () => {
    expect(rows.map((r) => r.id)).not.toContain("template-g1");
    expect(rows).toHaveLength(4);
  });

  it("puts recommended actions and official tasks first, keeping catalog order after", () => {
    expect(rows.map((r) => r.id)).toEqual(["template-spin", "template-one_leg", "template-official", "template-run"]);
  });

  it("marks which way each action can train", () => {
    const byId = Object.fromEntries(rows.map((r) => [r.id, r]));
    expect(supports(byId["template-spin"], "cpu")).toBe(true);
    expect(supports(byId["template-spin"], "cloud")).toBe(false);
    expect(supports(byId["template-official"], "cpu")).toBe(false);
    expect(supports(byId["template-official"], "cloud")).toBe(true);
  });

  it("sends the suggestion's own text, or the bare behavior id", () => {
    const byId = Object.fromEntries(rows.map((r) => [r.id, r]));
    expect(byId["template-one_leg"].local?.text).toBe("stand on one leg");
    expect(byId["template-run"].local?.text).toBe("run");
  });

  it("localizes known titles and leaves the rest as the catalog wrote them", () => {
    const byId = Object.fromEntries(rows.map((r) => [r.id, r]));
    expect(byId["template-one_leg"].title).toBe("单脚站立");
    expect(byId["template-run"].title).toBe("向前奔跑");
    expect(buildActionRows(catalog, "microduck", teach, "en").find((r) => r.id === "template-one_leg")?.title)
      .toBe("Stand on one leg");
  });

  it("returns nothing for a robot with no templates rather than another robot's", () => {
    expect(buildActionRows(catalog, "moss", [], "zh")).toEqual([]);
  });
});

describe("filterActionRows", () => {
  const rows = buildActionRows(catalog, "microduck", teach, "zh");
  it("filters by what each way can train", () => {
    expect(filterActionRows(rows, "cpu", "").every((r) => r.local)).toBe(true);
    expect(filterActionRows(rows, "cpu", "")).toHaveLength(3);
    expect(filterActionRows(rows, "cloud", "").map((r) => r.id)).toEqual(["template-official"]);
  });
  it("searches titles and ids, case-insensitively", () => {
    expect(filterActionRows(rows, "all", "单脚").map((r) => r.id)).toEqual(["template-one_leg"]);
    expect(filterActionRows(rows, "all", "SPIN").map((r) => r.id)).toEqual(["template-spin"]);
    expect(filterActionRows(rows, "all", "no such action")).toEqual([]);
  });
  it("combines filter and search", () => {
    expect(filterActionRows(rows, "cloud", "单脚")).toEqual([]);
  });
});

describe("fallbackActionRows", () => {
  it("survives the placeholder list a lab that has not answered leaves: no duplicate ids, titles still localized", () => {
    const placeholder = ["stand still", "spin in place"].map((text) => ({ text, behavior: "", emoji: "", title: text }));
    const rows = fallbackActionRows("microduck", placeholder, "zh");
    expect(new Set(rows.map((r) => r.id)).size).toBe(rows.length);
    expect(rows.filter((r) => r.local).map((r) => [r.title, r.local?.text])).toEqual([["保持站立", "stand still"], ["原地转圈", "spin in place"]]);
  });
  it("keeps the suggestions and adds the two official tasks for the duck only", () => {
    const duck = fallbackActionRows("microduck", teach, "zh");
    expect(duck.filter((r) => r.cloud).map((r) => r.cloud?.task)).toEqual([
      "Mjlab-Velocity-Flat-MicroDuck", "Mjlab-VelStand-Flat-MicroDuck",
    ]);
    expect(duck.filter((r) => r.local)).toHaveLength(2);
    expect(fallbackActionRows("g1", [{ text: "x", behavior: "g1_stand" }], "en").some((r) => r.cloud)).toBe(false);
  });
});

describe("unavailableReason", () => {
  const [spin, , official] = buildActionRows(catalog, "microduck", teach, "zh");
  it("says why, in terms of the other way that works", () => {
    expect(unavailableReason(spin, "cloud", "zh")).toContain("CPU");
    expect(unavailableReason(official, "cpu", "zh")).toContain("云端");
  });
});

describe("cloud presets and limits", () => {
  it("makes the first run small and bounded", () => {
    expect(CLOUD_PRESETS.trial).toEqual({ iterations: 100, maxMinutes: 30, maxCcu: "5" });
    expect(CLOUD_PRESETS.full.maxMinutes).toBeGreaterThan(CLOUD_PRESETS.trial.maxMinutes);
    expect(CLOUD_PRESETS.full.iterations).toBeGreaterThan(CLOUD_PRESETS.trial.iterations);
  });
  it("sends a CCU limit only when there is a balance to draw on and the limit is a positive number", () => {
    expect(effectiveCcuLimit("5", 200)).toBe(5);
    expect(effectiveCcuLimit("5", 0)).toBeNull();
    expect(effectiveCcuLimit("5", null)).toBeNull();
    expect(effectiveCcuLimit("", 200)).toBeNull();
    expect(effectiveCcuLimit("0", 200)).toBeNull();
    expect(effectiveCcuLimit("abc", 200)).toBeNull();
    expect(effectiveCcuLimit("-3", 200)).toBeNull();
  });
  it("treats only a known zero balance as free", () => {
    expect(mayUsePaidUnits(0)).toBe(false);
    expect(mayUsePaidUnits(200)).toBe(true);
    expect(mayUsePaidUnits(null)).toBe(true);
    expect(mayUsePaidUnits(undefined)).toBe(true);
  });
  it("lists every limit the person is agreeing to", () => {
    const text = cloudConfirmText({ title: "官方行走", gpu: "T4", iterations: 100, maxMinutes: 30, ccuLimit: 5, balance: 200 }, "zh");
    for (const part of ["官方行走", "T4", "100", "30 分钟", "5 计算单元", "200", "合上电脑"]) expect(text).toContain(part);
    expect(cloudConfirmText({ title: "x", gpu: "T4", iterations: 1, maxMinutes: 5, ccuLimit: null, balance: null }, "zh"))
      .toContain("不设上限");
  });
});
