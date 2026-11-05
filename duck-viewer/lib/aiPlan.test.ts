import { describe, expect, it } from "vitest";
import { changedWeights, termLabel } from "./aiPlan";
import type { Recipe } from "./plans";

const recipe = {
  id: "one_leg", robot: "microduck", emoji: "🦩", title: "x", description: "", howItLearns: "", successMetric: "", defaultSteps: 1,
  terms: [{ key: "one_leg_hold", friendly: "单脚站稳", weight: 5, isPenalty: false }, { key: "calm_body", friendly: "身体平稳", weight: 1, isPenalty: true }],
  availableTerms: [{ key: "head_up", friendly: "头抬起", weight: 1, isPenalty: false }], curriculum: [],
} as Recipe;

describe("changedWeights", () => {
  it("reports only what the model moved or added", () => {
    expect(changedWeights(recipe, { one_leg_hold: 5, calm_body: 3, head_up: 2 })).toEqual([
      { key: "calm_body", weight: 3, was: 1 },
      { key: "head_up", weight: 2, was: null },
    ]);
  });
  it("reports nothing when the model only restated the defaults", () => {
    expect(changedWeights(recipe, { one_leg_hold: 5, calm_body: 1 })).toEqual([]);
  });
  it("treats every term as added when the recipe is unknown rather than hiding them", () => {
    expect(changedWeights(undefined, { a: 1 })).toEqual([{ key: "a", weight: 1, was: null }]);
  });
  it("keeps a weight of zero visible", () => {
    expect(changedWeights(recipe, { calm_body: 0 })).toEqual([{ key: "calm_body", weight: 0, was: 1 }]);
  });
});

describe("termLabel", () => {
  it("uses the friendly name, from the recipe's terms or its addable ones, else the key", () => {
    expect(termLabel(recipe, "calm_body")).toBe("身体平稳");
    expect(termLabel(recipe, "head_up")).toBe("头抬起");
    expect(termLabel(recipe, "no_stall")).toBe("no stall");
    expect(termLabel(undefined, "x_y")).toBe("x y");
  });
});
