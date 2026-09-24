"use client";

// 🎥 record / 📷 shot for the /sim page, inline in its top bar. Unlike the
// lab page's RecordPanel there is no framing glide: the take is whatever the
// user is looking at, and the camera stays theirs for the duration (orbit,
// fly keys, follow a duck by hand). Files are named after the scenario. The
// footage is the WebGL canvas only, so the panels, labels and the scrub bar
// never appear in it — for a debugging clip WITH the numbers burned in, use
// `uv run record-world` instead (the record-world skill).

import { LAB_HTTP } from "@/lib/lab";
import { useI18n } from "@/lib/i18n";
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
  const { tr } = useI18n();
  const { cap } = take;
  const subject = `sim-${slug(scenario ?? "world")}`;
  const link: React.CSSProperties = { ...btn, textDecoration: "none", color: "#7db8d8", lineHeight: 1.4 };

  if (cap.phase === "idle") {
    return (
      <>
        <button style={btn} onClick={() => take.start(subject)} title={tr("Film the current camera view for up to 60 seconds; files are saved in microduck_local/captures/", "录制当前相机视角，最长 60 秒；文件保存在 microduck_local/captures/")}>
          🎥 {tr("record", "录制")}
        </button>
        <button style={btn} onClick={() => take.snap(`${subject}-${stamp()}`)} title={tr("Download a PNG of the current view", "下载当前画面的 PNG 图片")}>
          📷 {tr("shot", "截图")}
        </button>
      </>
    );
  }
  if (cap.phase === "framing" || cap.phase === "recording") {
    return (
      <>
        <button style={{ ...btn, borderColor: "#e5484d" }} onClick={take.stop} title={tr("Stop and save", "停止并保存")}>
          <span style={{ color: "#e5484d" }}>●</span> {take.secs}s ■ {tr("stop", "停止")}
        </button>
        <button style={btn} onClick={take.cancel} title={tr("Discard the recording", "丢弃录制内容")}>
          ✕
        </button>
      </>
    );
  }
  if (cap.phase === "processing") return <span style={{ color: "#9aa5b1" }}>⏳ {tr("making mp4 + gif…", "正在生成 mp4 和 gif…")}</span>;
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
      <span style={{ color: "#e5484d" }}>⚠ {cap.error ?? tr("capture failed", "捕获失败")}</span>
      <button style={btn} onClick={captureReset}>
        ✕
      </button>
    </>
  );
}
