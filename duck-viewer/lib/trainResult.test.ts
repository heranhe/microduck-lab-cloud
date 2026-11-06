import { describe, expect, it } from "vitest";
import { cloudResultToShow, elapsedText, finishHeadline, isFinish, justFinished, toastFor } from "./trainResult";
import type { CloudJob } from "./cloudCompute";

describe("justFinished", () => {
  it("reports a run only on the look where it changes from running to finished", () => {
    expect(justFinished({ a: "training" }, [{ runName: "a", status: "done" }])).toEqual([{ runName: "a", status: "done" }]);
    expect(justFinished({ a: "stopping" }, [{ runName: "a", status: "stopped" }])).toEqual([{ runName: "a", status: "stopped" }]);
    expect(justFinished({ a: "done" }, [{ runName: "a", status: "done" }])).toEqual([]);      // already told
    expect(justFinished({ a: "training" }, [{ runName: "a", status: "training" }])).toEqual([]);
  });
  it("stays quiet for a run it never saw running, such as one loaded from disk or the first look after a reload", () => {
    expect(justFinished({}, [{ runName: "old", status: "done" }])).toEqual([]);
  });
  it("tells failures too, and handles several runs at once", () => {
    const out = justFinished({ a: "training", b: "training", c: "training" },
      [{ runName: "a", status: "failed" }, { runName: "b", status: "done" }, { runName: "c", status: "training" }]);
    expect(out.map((r) => r.runName)).toEqual(["a", "b"]);
  });
});

describe("wording", () => {
  it("says what happened and its tone", () => {
    expect(finishHeadline("done", "转圈", "zh")).toEqual({ tone: "good", text: "✅ 训练完成 · 转圈" });
    expect(finishHeadline("stopped", "x", "en").tone).toBe("warn");
    expect(finishHeadline("failed", "x", "zh").tone).toBe("bad");
  });
  it("points the toast at the next step only when there is one", () => {
    expect(toastFor("done", "转圈", "zh")).toContain("放到舞台");
    expect(toastFor("failed", "转圈", "zh")).toContain("原因");
    expect(isFinish("done") && isFinish("failed") && !isFinish("training") && !isFinish(undefined)).toBe(true);
  });
  it("formats elapsed time and refuses a made-up one", () => {
    expect(elapsedText(42)).toBe("42s");
    expect(elapsedText(125)).toBe("2m05s");
    expect(elapsedText(3700)).toBe("1h01m");
    expect(elapsedText(null)).toBeNull();
    expect(elapsedText(-1)).toBeNull();
    expect(elapsedText(Number.NaN)).toBeNull();
  });
});

describe("cloudResultToShow", () => {
  const now = 1_000_000_000;
  const job = (over: Partial<CloudJob>): CloudJob => ({ id: "j", platform: "colab", task: "t", gpu: "T4", state: "done", released: true, released_at: now / 1000 - 60, ...over });
  it("shows the newest released finished job that was not dismissed", () => {
    const older = job({ id: "old", released_at: now / 1000 - 600 });
    const newer = job({ id: "new" });
    expect(cloudResultToShow([older, newer], [], now)?.id).toBe("new");
    expect(cloudResultToShow([older, newer], ["new"], now)?.id).toBe("old");
  });
  it("never shows a running job, one still billing, or one older than a day", () => {
    expect(cloudResultToShow([job({ state: "running" })], [], now)).toBeNull();
    expect(cloudResultToShow([job({ released: false })], [], now)).toBeNull();
    expect(cloudResultToShow([job({ released: undefined })], [], now)).toBeNull();
    expect(cloudResultToShow([job({ released_at: now / 1000 - 90_000 })], [], now)).toBeNull();
  });
  it("includes a failed job so the reason is not lost", () => {
    expect(cloudResultToShow([job({ state: "failed" })], [], now)?.state).toBe("failed");
  });
});
