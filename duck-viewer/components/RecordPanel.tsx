"use client";

// 🎥 capture panel (top-center): select a duck, hit record, get content.
// The flow: camera glides to a ¾ shot of the duck (RecordCamera in
// Viewer.tsx), MediaRecorder captures the WebGL canvas — DOM labels and
// panels are not part of the canvas, so takes come out clean — then the lab
// server converts the upload to mp4 + gif (POST /captures) and the panel
// offers both as downloads. The take itself is useTake.ts, shared with the
// /sim page's SimRecord; this file is the lab page's layout and framing.

import { useEffect, useRef, useState } from "react";
import { LAB_HTTP, type LabClient } from "@/lib/lab";
import { useSelectedDuck } from "@/lib/select";
import { useHudRight } from "@/lib/ui";
import { captureReset } from "@/lib/record";
import { slug, stamp, useTake } from "./useTake";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

/** Camera glide before the recorder rolls (matches RecordCamera's damping —
 *  the shot has settled by then, so takes don't open with a swish). */
const FRAMING_MS = 1200;

// No `left` here — the component computes it per render: centered, but never
// under the top-left HUD panel (see the layout block in RecordPanel).
const panelStyle: React.CSSProperties = {
  position: "fixed",
  top: 10,
  zIndex: 20,
  display: "flex",
  alignItems: "center",
  gap: 8,
  background: "rgba(14, 16, 20, 0.86)",
  border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 8,
  padding: "6px 10px",
  color: "#d8dee8",
  fontFamily: mono,
  fontSize: 12,
  backdropFilter: "blur(6px)",
};

const btnStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.06)",
  border: "1px solid rgba(255,255,255,0.14)",
  borderRadius: 6,
  color: "#d8dee8",
  fontFamily: mono,
  fontSize: 12,
  padding: "3px 10px",
  cursor: "pointer",
};

const linkStyle: React.CSSProperties = {
  ...btnStyle,
  textDecoration: "none",
  color: "#7db8d8",
};

export function RecordPanel({
  clientRef,
}: {
  clientRef: React.MutableRefObject<LabClient | null>;
}) {
  const selected = useSelectedDuck();
  const take = useTake({ framingMs: FRAMING_MS });
  const { cap } = take;

  // Layout inputs for the HUD-dodging `left` computed at the bottom: the
  // HUD's live right edge, this panel's own width, and the window width.
  const hudRight = useHudRight();
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [panelW, setPanelW] = useState(220);
  const [winW, setWinW] = useState(() =>
    typeof window === "undefined" ? 1200 : window.innerWidth
  );
  useEffect(() => {
    const onResize = () => setWinW(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    setPanelW(el.offsetWidth);
    const ro = new ResizeObserver(() => setPanelW(el.offsetWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const duckName = () =>
    selected
      ? clientRef.current?.frame?.ducks.find((d) => d.id === selected)?.name ?? "duck"
      : null;

  // The duck is captured by ID at start, so deselecting mid-take doesn't
  // lose the shot (RecordCamera frames cap.duckId).
  const start = () => {
    if (!selected) return;
    take.start(duckName() ?? "duck", selected);
  };

  // 📷 is instant and client-side: name the file after the selected duck (or
  // the whole lab) and capture SYNCHRONOUSLY — the download must fire inside
  // this click's user gesture (see lib/record.ts).
  const snap = () => take.snap(`${slug(duckName() ?? "duck-lab")}-${stamp()}`);

  let content: React.ReactNode;
  if (cap.phase === "idle") {
    content = (
      <>
        {selected && (
          <button style={btnStyle} onClick={start} title="film the selected duck — the camera frames it, then mp4 + gif land in captures/">
            🎥 record
          </button>
        )}
        <button style={btnStyle} onClick={snap} title="download a PNG of the current view (selection ring hidden for the shot)">
          📷 shot
        </button>
      </>
    );
  } else if (cap.phase === "framing" || cap.phase === "recording") {
    content = (
      <>
        {cap.phase === "framing" ? (
          <span>🎥 framing…</span>
        ) : (
          <>
            <span style={{ color: "#e07a5f" }}>●</span>
            <span>{take.secs}s</span>
            <button style={btnStyle} onClick={take.stop}>
              ■ stop
            </button>
          </>
        )}
        <button style={btnStyle} onClick={take.cancel} title="discard the take">
          ✕
        </button>
      </>
    );
  } else if (cap.phase === "processing") {
    content = <span>⏳ making mp4 + gif…</span>;
  } else if (cap.phase === "done" && cap.result) {
    content = (
      <>
        <span>🎥 {cap.result.name}</span>
        <a style={linkStyle} href={`${LAB_HTTP}${cap.result.mp4}`}>
          ⬇ mp4 {Math.max(1, Math.round(cap.result.mp4Kb / 1024))}MB
        </a>
        <a style={linkStyle} href={`${LAB_HTTP}${cap.result.gif}`}>
          ⬇ gif {Math.max(1, Math.round(cap.result.gifKb / 1024))}MB
        </a>
        <button style={btnStyle} onClick={captureReset}>
          ✕
        </button>
      </>
    );
  } else {
    content = (
      <>
        <span style={{ color: "#e07a5f" }}>
          ⚠ {cap.error ?? "capture failed"}
        </span>
        <button style={btnStyle} onClick={captureReset}>
          ✕
        </button>
      </>
    );
  }

  // Centered at the top — but never UNDER the HUD: a wide duck-lab panel
  // (long duck names) used to reach right beneath these buttons. The HUD
  // publishes its right edge (useHudRight); slide right of it when centering
  // would collide, and keep a margin from the right edge as a backstop.
  const centered = (winW - panelW) / 2;
  const left = Math.round(
    Math.min(Math.max(centered, hudRight + 12), Math.max(12, winW - panelW - 12))
  );
  return (
    <div ref={wrapRef} data-policy-ui style={{ ...panelStyle, left }}>
      {content}
    </div>
  );
}
