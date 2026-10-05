/** Episode boundaries, made visible.
 *
 * A clip ending and a ball being flung look identical on the stage: the duck
 * and its ball both teleport to a fresh spawn in one frame. Measured on a live
 * dribble trainee, 400 frames — exactly one ball movement over 80 mm, 1145 mm
 * in a single frame, and it coincided with the step counter resetting; normal
 * rolling is 7 mm a frame. So every "the ball suddenly shot off sideways" a
 * person reports is either a real kick or an episode boundary, and the stage
 * gave them no way to tell which.
 *
 * The signal is the server's own per-duck `step`, which counts control steps
 * within the current episode and returns to ~0 on reset. `falls` is cumulative,
 * so a fall is the boundary where it ALSO increments — which is the difference
 * between "it fell over" and "the clip simply ended", and those read very
 * differently to someone watching.
 *
 * Pure so it can be tested without a socket or a canvas: the caller keeps the
 * previous sample and asks what changed.
 */

export type EpisodeCause = "fell" | "ended";

export interface EpisodeSample {
  /** Control steps into the current episode (server: Duck.step). */
  step: number;
  /** Cumulative falls for this duck (server: Duck.falls). */
  falls: number;
}

export interface EpisodeEdge {
  /** True on the single frame an episode boundary is first visible. */
  reset: boolean;
  /** Why it ended, or null when this is not a boundary. */
  cause: EpisodeCause | null;
}

const NONE: EpisodeEdge = { reset: false, cause: null };

/** Compare consecutive samples for one duck.
 *
 * A boundary is `step` going BACKWARDS. Not "step === 0": the socket drops
 * frames under load, so the first frame of a new episode is routinely step 3
 * or step 7 rather than 0, and an equality test would miss most resets. Not
 * "step decreased by a lot" either — an episode that ends at step 12 (an early
 * fall) is exactly the case worth showing.
 *
 * `null`/`undefined` samples return no edge rather than throwing: a duck can
 * appear mid-stream, and a viewer that crashes on a joining duck is worse than
 * one that misses its first boundary.
 */
export function episodeEdge(
  prev: EpisodeSample | null | undefined,
  cur: EpisodeSample | null | undefined,
): EpisodeEdge {
  if (!prev || !cur) return NONE;
  if (!Number.isFinite(prev.step) || !Number.isFinite(cur.step)) return NONE;
  if (cur.step >= prev.step) return NONE;
  // Cumulative counter, so only an INCREASE means this boundary was a fall.
  // A server that resets `falls` (a fresh roster) must not read as a fall.
  const fell = Number.isFinite(prev.falls) && Number.isFinite(cur.falls)
    && cur.falls > prev.falls;
  return { reset: true, cause: fell ? "fell" : "ended" };
}

/** What to show for a boundary, and for how long.
 *
 * 900 ms: long enough to catch out of the corner of an eye at a 7-second
 * episode cadence, short enough not to be on screen for a seventh of the
 * clip it is annotating.
 */
export const EPISODE_FLASH_MS = 900;

export function episodeFlashLabel(cause: EpisodeCause): string {
  return cause === "fell" ? "↺ fell — new clip" : "↺ new clip";
}

/** 1 → 0 over `EPISODE_FLASH_MS`, clamped. Drives the badge's fade.
 *
 * Takes the elapsed time rather than reading a clock so it stays pure and the
 * caller can drive it from the render loop's own clock.
 */
export function episodeFlashAlpha(elapsedMs: number): number {
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) return 0;
  if (elapsedMs >= EPISODE_FLASH_MS) return 0;
  return 1 - elapsedMs / EPISODE_FLASH_MS;
}
