// Two-finger trackpad swipe → one gesture, one meaning.
//
// A trackpad reports swipes as a wheel-event stream, and a swipe is never
// perfectly axis-aligned. Deciding per event (horizontal-dominant → slide,
// vertical-dominant → zoom) flipped between the two mid-swipe on any slightly
// diagonal motion and did both at once — it felt awful. So the axis is
// decided ONCE per gesture, from the first few pixels of accumulated motion,
// and held until the stream pauses (a momentum fling keeps events flowing
// well under the gap, so the lock survives the coast).
//
// This is the pure decision machine; the DOM plumbing that feeds it lives in
// `components/useTruckSwipe.ts`, and both the lab page and /sim use it so the
// two pages fly the same way.

/** What the caller should do with one wheel event. */
export type SwipeVerdict =
  | "hold"  // axis still undecided — swallow it, give it to neither
  | "slide" // gesture is horizontal — truck the camera by dx
  | "zoom"; // gesture is vertical — let OrbitControls have it

export interface SwipeGesture {
  /** Classify one wheel event (deltas in PIXELS, timestamp in ms). */
  feed(dx: number, dy: number, nowMs: number): SwipeVerdict;
}

/** Silence longer than this ends the gesture: the next event starts a new one. */
export const GESTURE_GAP_MS = 180;
/** Accumulated |dx|+|dy| before the axis is committed. */
export const LOCK_AFTER_PX = 6;

export function createSwipeGesture(
  gapMs = GESTURE_GAP_MS,
  lockAfterPx = LOCK_AFTER_PX,
): SwipeGesture {
  let axis: "x" | "y" | null = null;
  let undecidedX = 0;
  let undecidedY = 0;
  let lastAt = -Infinity;
  return {
    feed(dx, dy, nowMs) {
      if (nowMs - lastAt > gapMs) {
        axis = null;
        undecidedX = undecidedY = 0;
      }
      lastAt = nowMs;
      if (axis === null) {
        undecidedX += Math.abs(dx);
        undecidedY += Math.abs(dy);
        if (undecidedX + undecidedY < lockAfterPx) return "hold";
        axis = undecidedX > undecidedY ? "x" : "y";
      }
      return axis === "x" ? "slide" : "zoom";
    },
  };
}

/** Rare line-mode mice report lines, not pixels — normalize roughly. */
export function wheelPixels(delta: number, deltaMode: number): number {
  return deltaMode ? delta * 16 : delta;
}
