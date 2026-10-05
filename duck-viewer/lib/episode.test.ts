// Episode-boundary detection, pinned. The reason this module exists is that a
// clip ending and a ball being kicked look the same on the stage — measured on
// a live trainee, the one ball movement over 80 mm in 400 frames was 1145 mm in
// a single frame and coincided with the step counter resetting, against 7 mm a
// frame of ordinary rolling. The two cases below that matter most are the
// dropped-frame reset (step 7, not step 0) and the fall-versus-ended split,
// because those are the ones a naive implementation gets wrong.

import { describe, expect, it } from "vitest";
import {
  EPISODE_FLASH_MS,
  episodeEdge,
  episodeFlashAlpha,
  episodeFlashLabel,
} from "@/lib/episode";

describe("episodeEdge", () => {
  it("sees a boundary when step goes backwards", () => {
    expect(episodeEdge({ step: 340, falls: 2 }, { step: 0, falls: 2 })).toEqual({
      reset: true,
      cause: "ended",
    });
  });

  it("sees a boundary even when the new episode's first frame is not step 0", () => {
    // The socket drops frames under load, so `step === 0` is routinely never
    // observed. An equality test would miss most real resets.
    expect(episodeEdge({ step: 398, falls: 0 }, { step: 7, falls: 0 }).reset).toBe(true);
  });

  it("calls it a FALL only when the cumulative counter increments", () => {
    expect(episodeEdge({ step: 120, falls: 3 }, { step: 2, falls: 4 }).cause).toBe("fell");
    expect(episodeEdge({ step: 120, falls: 3 }, { step: 2, falls: 3 }).cause).toBe("ended");
  });

  it("does not read a falls counter going DOWN as a fall", () => {
    // A fresh roster resets `falls`; that is not the duck falling over.
    const e = episodeEdge({ step: 50, falls: 9 }, { step: 1, falls: 0 });
    expect(e).toEqual({ reset: true, cause: "ended" });
  });

  it("is silent while an episode is simply running", () => {
    expect(episodeEdge({ step: 10, falls: 0 }, { step: 11, falls: 0 }).reset).toBe(false);
    // ...including a frame that repeats, which a dropped/duplicated frame does
    expect(episodeEdge({ step: 11, falls: 0 }, { step: 11, falls: 0 }).reset).toBe(false);
  });

  it("catches the SHORT episode, which is the one worth showing", () => {
    // An early fall ends at step ~12. A "decreased by a lot" test would miss it.
    expect(episodeEdge({ step: 12, falls: 0 }, { step: 1, falls: 1 })).toEqual({
      reset: true,
      cause: "fell",
    });
  });

  it("returns no edge for a duck that just joined, rather than throwing", () => {
    expect(episodeEdge(null, { step: 4, falls: 0 }).reset).toBe(false);
    expect(episodeEdge({ step: 4, falls: 0 }, undefined).reset).toBe(false);
    expect(episodeEdge({ step: NaN, falls: 0 }, { step: 1, falls: 0 }).reset).toBe(false);
  });
});

describe("the flash", () => {
  it("names the two cases differently", () => {
    expect(episodeFlashLabel("fell")).toContain("fell");
    expect(episodeFlashLabel("ended")).not.toContain("fell");
  });

  it("fades from 1 to 0 and stays clamped", () => {
    expect(episodeFlashAlpha(0)).toBe(1);
    expect(episodeFlashAlpha(EPISODE_FLASH_MS / 2)).toBeCloseTo(0.5);
    expect(episodeFlashAlpha(EPISODE_FLASH_MS)).toBe(0);
    expect(episodeFlashAlpha(EPISODE_FLASH_MS * 10)).toBe(0);
    expect(episodeFlashAlpha(-5)).toBe(0);
    expect(episodeFlashAlpha(NaN)).toBe(0);
  });
});
