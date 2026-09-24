// The swipe axis lock, pinned. A trackpad swipe is never axis-aligned, and
// the failure this file was written for was per-event dominance: one slightly
// diagonal gesture flipped between trucking and zooming, doing both at once.
// The lock is a state machine over a stream, so what a test can hold is that
// one stream of events yields ONE verdict — and that a pause ends it.

import { describe, expect, it } from "vitest";
import { createSwipeGesture, GESTURE_GAP_MS, LOCK_AFTER_PX } from "@/lib/swipe";

/** Feed a stream 8 ms apart (a trackpad's rate) from t=1000. */
function stream(events: Array<[number, number]>, t0 = 1000, dt = 8) {
  const g = createSwipeGesture();
  return events.map(([dx, dy], i) => g.feed(dx, dy, t0 + i * dt));
}

describe("swipe axis lock", () => {
  it("holds the ambiguous first pixels back from both", () => {
    expect(stream([[1, 0], [1, 1]])).toEqual(["hold", "hold"]);
  });

  it("locks a horizontal gesture to slide, and keeps it there", () => {
    // Diagonal drift after the lock — every later event stays a slide.
    const v = stream([[4, 1], [4, 1], [1, 6], [0, 9], [-1, 7]]);
    expect(v).toEqual(["hold", "slide", "slide", "slide", "slide"]);
  });

  it("locks a vertical gesture to zoom, and keeps it there", () => {
    const v = stream([[1, 4], [1, 4], [7, 1], [9, 0]]);
    expect(v).toEqual(["hold", "zoom", "zoom", "zoom"]);
  });

  it("commits only once the accumulated motion passes the threshold", () => {
    const g = createSwipeGesture();
    let t = 1000;
    const step = LOCK_AFTER_PX / 4;
    // Three sub-threshold events (3/4 of the budget), then the fourth commits.
    expect(g.feed(step, 0, t)).toBe("hold");
    expect(g.feed(step, 0, (t += 8))).toBe("hold");
    expect(g.feed(step, 0, (t += 8))).toBe("hold");
    expect(g.feed(step, 0, (t += 8))).toBe("slide");
  });

  it("survives a momentum fling: the coast keeps the lock", () => {
    const g = createSwipeGesture();
    let t = 1000;
    g.feed(9, 2, t);
    // Fling events thin out but stay inside the gap.
    for (const gap of [16, 40, 90, 170]) {
      expect(g.feed(0.5, 0.4, (t += gap))).toBe("slide");
    }
  });

  it("a pause ends the gesture — the next swipe decides fresh", () => {
    const g = createSwipeGesture();
    let t = 1000;
    expect(g.feed(9, 0, t)).toBe("slide");
    // Fingers lifted: past the gap, a vertical swipe must be a zoom, not a
    // slide inherited from the swipe before it.
    t += GESTURE_GAP_MS + 1;
    expect(g.feed(0, 9, t)).toBe("zoom");
  });

  it("the very first event of the page is a fresh gesture", () => {
    // No `lastAt` seeded from a real clock: performance.now() at page load is
    // small, and a machine-uptime-sized timestamp must not look like a pause
    // that never happened (or, worse, a continuation).
    const g = createSwipeGesture();
    expect(g.feed(9, 0, 0)).toBe("slide");
  });
});
