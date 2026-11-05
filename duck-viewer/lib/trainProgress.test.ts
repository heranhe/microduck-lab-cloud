import { describe, expect, it } from "vitest";
import { appendCloudScore, cloudProgressView, formatClock, localProgressView } from "./trainProgress";
import type { CloudJob } from "./cloudCompute";

const job = (over: Partial<CloudJob> = {}): CloudJob => ({
  id: "j1", platform: "colab", task: "t", gpu: "T4", state: "running", remote_state: "training", ...over,
});

describe("formatClock", () => {
  it("formats hours, minutes and seconds, and never goes negative", () => {
    expect(formatClock(3661)).toBe("01:01:01");
    expect(formatClock(-5)).toBe("00:00:00");
    expect(formatClock(null)).toBe("00:00:00");
  });
});

describe("localProgressView", () => {
  it("shows a real percentage and the remaining time while training", () => {
    const v = localProgressView({ status: "training", steps: 250, total: 1000, etaSeconds: 90, score: 4.5 }, "zh");
    expect(v).toMatchObject({ where: "local", phase: "训练中", percent: 25, remaining: "00:01:30", score: 4.5 });
    expect(v.count).toBe("250 / 1,000 步");
  });
  it("drops the estimate while saving, and has no percentage without a total", () => {
    expect(localProgressView({ status: "stopping", steps: 1, total: 10, etaSeconds: 5 }, "en").remaining).toBeNull();
    expect(localProgressView({ status: "training", steps: 0, total: 0 }, "en").percent).toBeNull();
  });
  it("caps at 100 and tells a missing score from a zero one", () => {
    expect(localProgressView({ status: "training", steps: 12, total: 10 }, "en").percent).toBe(100);
    expect(localProgressView({ status: "training", steps: 1, total: 10, score: 0 }, "en").score).toBe(0);
    expect(localProgressView({ status: "training", steps: 1, total: 10, score: null }, "en").score).toBeNull();
    expect(localProgressView({ status: "training", steps: 1, total: 10, score: Number.NaN }, "en").score).toBeNull();
  });
});

describe("cloudProgressView", () => {
  const now = 1_000_000;
  it("uses the same fields as the local view", () => {
    const v = cloudProgressView(job({
      allocated_at: now / 1000 - 600, max_minutes: 30, rate: 1.8, ccu_spent: 0.3,
      progress: { iteration: 49, iterations: 100, reward: "12.5", eta: "00:10:00", gpu_util: 97 },
    }), now, "zh");
    expect(v).toMatchObject({ where: "cloud", phase: "训练中", percent: 50, remaining: "00:10:00", score: 12.5 });
    expect(v.count).toBe("49/100 迭代");
    expect(v.extras).toEqual(["GPU 97%", "距时长上限 00:20:00", "1.8 CCU/h", "已用 0.3 CCU"]);
  });
  it("has no number before the trainer reports one, and says what the runtime is doing", () => {
    const v = cloudProgressView(job({ remote_state: "setup" }), now, "zh");
    expect(v).toMatchObject({ phase: "正在安装依赖", percent: null, count: null, remaining: null, score: null, extras: [] });
  });
  it("keeps a score of zero", () => {
    expect(cloudProgressView(job({ progress: { reward: "0.00" } }), now, "en").score).toBe(0);
  });
});

describe("appendCloudScore", () => {
  const at = (iteration: number, reward: string, id = "j1") => job({ id, progress: { iteration, reward } });
  const empty = { jobId: "", points: [] };

  it("grows one point per new iteration", () => {
    let h = appendCloudScore(empty, at(1, "1"));
    h = appendCloudScore(h, at(2, "2"));
    expect(h.points).toEqual([{ x: 1, y: 1 }, { x: 2, y: 2 }]);
  });
  it("ignores a poll that changed nothing, and replaces a revised score at the same iteration", () => {
    const h = appendCloudScore(empty, at(1, "1"));
    expect(appendCloudScore(h, at(1, "1"))).toBe(h);
    expect(appendCloudScore(h, at(1, "3")).points).toEqual([{ x: 1, y: 3 }]);
  });
  it("starts over for another job or an iteration that went backwards", () => {
    const h = appendCloudScore(appendCloudScore(empty, at(5, "5")), at(6, "6"));
    expect(appendCloudScore(h, at(1, "9", "j2")).points).toEqual([{ x: 1, y: 9 }]);
    expect(appendCloudScore(h, at(2, "9")).points).toEqual([{ x: 2, y: 9 }]);
  });
  it("does not splice a new job onto the old curve even when its iteration is higher", () => {
    const h = appendCloudScore(empty, at(3, "3"));
    const next = appendCloudScore(h, at(9, "9", "j2"));
    expect(next).toEqual({ jobId: "j2", points: [{ x: 9, y: 9 }] });
  });
  it("skips samples without real numbers instead of plotting zero", () => {
    const h = appendCloudScore(empty, at(1, "1"));
    expect(appendCloudScore(h, job({ progress: { iteration: 2 } }))).toBe(h);
    expect(appendCloudScore(h, at(2, "nan"))).toBe(h);
  });
  it("clears when there is no job", () => {
    const h = appendCloudScore(empty, at(1, "1"));
    expect(appendCloudScore(h, undefined)).toEqual({ jobId: "", points: [] });
    expect(appendCloudScore(empty, undefined)).toBe(empty);
  });
});
