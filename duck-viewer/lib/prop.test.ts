import { describe, expect, it } from "vitest";
import { propRadius, propShape } from "./prop";

describe("propShape", () => {
  it("reads the server's three payload lengths", () => {
    expect(propShape([1, 2, 3, 0.035])).toBe("sphere");
    expect(propShape([1, 2, 3, 0.033, 0.0575])).toBe("cylinder");
    expect(propShape([1, 2, 3, 0.033, 0.0575, 1, 0, 0, 0])).toBe("cylinder");
    expect(propShape([1, 2, 3, 0.02, 0.02, 0.02, 1, 0, 0, 0])).toBe("box");
  });

  it("has no opinion about an absent or truncated prop", () => {
    expect(propShape(null)).toBe("none");
    expect(propShape(undefined)).toBe("none");
    expect(propShape([])).toBe("none");
    expect(propShape([1, 2, 3])).toBe("none");
  });

  it("calls a box a box and NOT a sphere", () => {
    // The regression: the orange ball mesh asked only "is it a cylinder?",
    // so every box payload left it switched on inside the box, scaled to
    // the box's x half-extent read as a radius.
    const box = [1, 2, 3, 0.02, 0.04, 0.01, 1, 0, 0, 0];
    expect(propShape(box)).not.toBe("sphere");
    expect(propShape(box)).not.toBe("cylinder");
  });
});

describe("propRadius", () => {
  it("takes slot 3 for the two shapes that have a radius", () => {
    expect(propRadius([1, 2, 3, 0.035])).toBeCloseTo(0.035);
    expect(propRadius([1, 2, 3, 0.033, 0.0575])).toBeCloseTo(0.033);
  });

  it("takes a box's WIDER ground half-extent, as GraspProp.radius does", () => {
    expect(propRadius([1, 2, 3, 0.02, 0.04, 0.01, 1, 0, 0, 0]))
      .toBeCloseTo(0.04);
    expect(propRadius([1, 2, 3, 0.05, 0.01, 0.01, 1, 0, 0, 0]))
      .toBeCloseTo(0.05);
  });

  it("falls back for no prop and for a zeroed slot", () => {
    expect(propRadius(null)).toBeCloseTo(0.035);
    expect(propRadius([1, 2, 3, 0])).toBeCloseTo(0.035);
    expect(propRadius(null, 0.1)).toBeCloseTo(0.1);
  });
});
