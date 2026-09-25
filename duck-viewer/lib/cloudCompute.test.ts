import { describe, expect, it } from "vitest";
import { cloudTaskFor } from "./cloudCompute";

describe("cloudTaskFor", () => {
  it("maps the equivalent MicroDuck stand recipe to the official cloud task", () => {
    expect(cloudTaskFor("microduck", "stand")).toBe("Mjlab-VelStand-Flat-MicroDuck");
  });

  it("does not silently replace unsupported local recipes with another cloud task", () => {
    expect(cloudTaskFor("microduck", "one_leg")).toBeNull();
    expect(cloudTaskFor("microduck", "headstand")).toBeNull();
  });

  it("does not offer MicroDuck cloud tasks for another robot", () => {
    expect(cloudTaskFor("g1", "stand")).toBeNull();
  });
});

