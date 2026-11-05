import { describe, expect, it } from "vitest";
import {
  canSaveAndDisconnect,
  cloudBudgetLeft,
  cloudHeartbeatStale,
  cloudPhaseText,
  cloudProgressPercent,
  cloudTaskFor,
  needsRelease,
} from "./cloudCompute";

describe("cloudTaskFor", () => {
  it("offers an explicitly named official recovery task", () => {
    expect(cloudTaskFor("microduck", "official_velstand")).toBe("Mjlab-VelStand-Flat-MicroDuck");
  });

  it("offers the separately named official walking task", () => {
    expect(cloudTaskFor("microduck", "official_velocity")).toBe("Mjlab-Velocity-Flat-MicroDuck");
  });

  it("does not silently replace unsupported local recipes with another cloud task", () => {
    expect(cloudTaskFor("microduck", "stand")).toBeNull();
    expect(cloudTaskFor("microduck", "one_leg")).toBeNull();
    expect(cloudTaskFor("microduck", "headstand")).toBeNull();
  });

  it("does not offer MicroDuck cloud tasks for another robot", () => {
    expect(cloudTaskFor("g1", "stand")).toBeNull();
  });
});

describe("cloud job view", () => {
  it("does not call dependency installation training", () => {
    expect(cloudPhaseText({ state: "running", remote_state: "setup" }, "zh")).toBe("正在安装依赖");
    expect(cloudPhaseText({ state: "running", remote_state: "training" }, "zh")).toBe("训练中");
    expect(cloudPhaseText({ state: "running" }, "en")).toBe("Starting");
    expect(cloudPhaseText({ state: "saving", remote_state: "training" }, "en")).toBe("Saving progress");
  });

  it("treats only a confirmed release as an all-clear", () => {
    expect(needsRelease({ state: "running", released: false })).toBe(true);
    expect(needsRelease({ state: "failed", released: false })).toBe(true);
    expect(needsRelease({ state: "saving" })).toBe(true);
    expect(needsRelease({ state: "done", released: true })).toBe(false);
    expect(needsRelease({ state: "failed", released: true })).toBe(false);
  });

  it("reports progress only when the log gave a number", () => {
    expect(cloudProgressPercent({})).toBeNull();
    expect(cloudProgressPercent({ progress: { iteration: 49, iterations: 100 } })).toBe(50);
    expect(cloudProgressPercent({ progress: { iteration: 0, iterations: 0 } })).toBeNull();
    expect(cloudProgressPercent({ progress: { iteration: 200, iterations: 100 } })).toBe(100);
  });

  it("counts the time limit from assignment, because that is when billing starts", () => {
    expect(cloudBudgetLeft({ allocated_at: 1000, max_minutes: 10 }, 1300_000)).toBe(300);
    expect(cloudBudgetLeft({ allocated_at: 1000, max_minutes: 10 }, 9999_000)).toBe(0);
    expect(cloudBudgetLeft({ allocated_at: null, max_minutes: 10 }, 0)).toBeNull();
  });

  it("offers a save only when a Colab trainer is actually running", () => {
    const job = { platform: "colab" as const, state: "running", remote_state: "training", released: false };
    expect(canSaveAndDisconnect(job)).toBe(true);
    expect(canSaveAndDisconnect({ ...job, remote_state: "setup" })).toBe(false);
    expect(canSaveAndDisconnect({ ...job, state: "detached" })).toBe(false);
    expect(canSaveAndDisconnect({ ...job, platform: "hf" })).toBe(false);
  });

  it("flags stale runtime numbers", () => {
    expect(cloudHeartbeatStale({ state: "running", last_heartbeat_at: 1000 }, 1050_000)).toBe(false);
    expect(cloudHeartbeatStale({ state: "running", last_heartbeat_at: 1000 }, 1200_000)).toBe(true);
    expect(cloudHeartbeatStale({ state: "done", last_heartbeat_at: 1000 }, 1200_000)).toBe(false);
  });
});
