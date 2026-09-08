"use client";

// 🎥 record / 📷 shot for the /sim page, inline in its top bar. Unlike the
// lab page's RecordPanel there is no framing glide: the take is whatever the
// user is looking at, and the camera stays theirs for the duration (orbit,
// fly keys, follow a duck by hand). Files are named after the scenario. The
// footage is the WebGL canvas only, so the panels, labels and the scrub bar
// never appear in it — for a debugging clip WITH the numbers burned in, use
// `uv run record-world` instead (the record-world skill).

import { LAB_HTTP } from "@/lib/lab";
import { captureReset } from "@/lib/record";
import { slug, stamp, useTake } from "./useTake";

export function SimRecord({
  scenario,
  btn,
}: {
  scenario: string | null;
  btn: React.CSSProperties;
}) {
  const take = useTake();
  const { cap } = take;
  const subject = `sim-${slug(scenario ?? "world")}`;
  const link: React.CSSProperties = { ...btn, textDecoration: "none", color: "#7db8d8", lineHeight: 1.4 };

  if (cap.phase === "idle") {
    return (
      <>
        <button style={btn} onClick={() => take.start(subject)} title="film the view — your camera, up to 60 s; mp4 + gif land in microduck_local/captures/">
          🎥 record
        </button>
        <button style={btn} onClick={() => take.snap(`${subject}-${stamp()}`)} title="download a PNG of the current view">
          📷 shot
        </button>
      </>
    );
  }
  if (cap.phase === "framing" || cap.phase === "recording") {
    return (
      <>
        <button style={{ ...btn, borderColor: "#e5484d" }} onClick={take.stop} title="stop and save">
          <span style={{ color: "#e5484d" }}>●</span> {take.secs}s ■ stop
        </button>
        <button style={btn} onClick={take.cancel} title="discard the take">
          ✕
        </button>
      </>
    );
  }
  if (cap.phase === "processing") return <span style={{ color: "#9aa5b1" }}>⏳ making mp4 + gif…</span>;
  if (cap.phase === "done" && cap.result) {
    return (
      <>
        <span style={{ color: "#9aa5b1" }}>🎥 {cap.result.name}</span>
        <a style={link} href={`${LAB_HTTP}${cap.result.mp4}`}>
          ⬇ mp4 {Math.max(1, Math.round(cap.result.mp4Kb / 1024))}MB
        </a>
        <a style={link} href={`${LAB_HTTP}${cap.result.gif}`}>
          ⬇ gif {Math.max(1, Math.round(cap.result.gifKb / 1024))}MB
        </a>
        <button style={btn} onClick={captureReset}>
          ✕
        </button>
      </>
    );
  }
  return (
    <>
      <span style={{ color: "#e5484d" }}>⚠ {cap.error ?? "capture failed"}</span>
      <button style={btn} onClick={captureReset}>
        ✕
      </button>
    </>
  );
}
