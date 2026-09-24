"use client";

// Two-finger trackpad swipe → lateral camera truck, shared by the lab page
// and /sim so both pages fly the same way.
//
// Horizontal-dominant swipes become truck impulses (the same motion as A/D,
// applied per frame by CameraKeys); vertical-dominant and ctrlKey (pinch)
// events pass through untouched so OrbitControls keeps zooming. Capture-phase
// + passive:false, because preventDefault must beat the browser's two-finger
// back/forward navigation and stopPropagation must keep OrbitControls from
// also treating the event as zoom. Only STAGE events count: the side panels
// (policies, teach, HUD, the /sim inspector) scroll with two fingers and must
// keep doing so, so anything whose target isn't the wrapper itself or the
// <canvas> is ignored. The axis lock that makes one swipe mean one thing is
// in `lib/swipe.ts`.

import { useEffect } from "react";
import { truckImpulse } from "@/lib/camera";
import { createSwipeGesture, wheelPixels } from "@/lib/swipe";

export function useTruckSwipe(root: React.RefObject<HTMLElement | null>) {
  useEffect(() => {
    const gesture = createSwipeGesture();
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey) return; // pinch-zoom stays OrbitControls'
      const el = root.current;
      const t = e.target;
      if (!el || !(t instanceof HTMLElement) || !el.contains(t)) return;
      if (t !== el && t.tagName !== "CANVAS") return; // panel UI scrolls natively

      const dx = wheelPixels(e.deltaX, e.deltaMode);
      const dy = wheelPixels(e.deltaY, e.deltaMode);
      const verdict = gesture.feed(dx, dy, performance.now());
      if (verdict === "zoom") return; // whole gesture = zoom (OrbitControls)
      // "hold" keeps the ambiguous first pixels back from BOTH.
      if (verdict === "slide") truckImpulse(dx);
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener("wheel", onWheel, { capture: true, passive: false });
    return () => window.removeEventListener("wheel", onWheel, true);
  }, [root]);
}
