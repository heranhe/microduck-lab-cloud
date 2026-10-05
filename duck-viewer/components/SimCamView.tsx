"use client";

// WHAT THE CAMERAS CAN SEE, drawn in the room (/sim, under the sensors
// toggle T). Asked on /sim 2026-09-28: "how come the depth sensor can't see
// that small cylinder far away — is it too small? are we sure it's working?"
// The answer was in the detector's geometry and nowhere on the page, so this
// draws it:
//
//   * the head camera's field of view on the floor: its two side edges, and
//     the near edge where the bottom of the frame meets the floor (nothing
//     nearer than that is in the picture at all)
//   * a ring on every object, coloured by WHY it is or is not a detection
//     (`sensors/detector.Detector.explain`, sent as `det.why`):
//       green   seen — big enough to be found on every frame
//       amber   sometimes — the size ramp; the label says how often
//       red     too small — in view and unblocked, but under the detector's
//               smallest box at this range
//       violet  blocked — in view but something is in the way
//       grey    out of view / beyond range (no label, to keep it quiet)
//     with a sight line from the lens to each one it can see
//   * the WRIST camera's cone out to its depth range (MOSS: 0.6 m)
//
// One robot at a time: the selected one, or the only one in the room.
// Per-frame painting follows the page's rule — no React state per frame: one
// LineSegments rewritten in place, and a fixed pool of labels moved by ref.

import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { createPortal } from "react-dom";
import * as THREE from "three";

import type { Scene } from "@/lib/lab";
import { loadJSON, saveJSON } from "@/lib/persist";
import { getSelectedDuck } from "@/lib/select";
import { getDuckLabels } from "@/lib/ui";
import { PanelToggle } from "./Panel";
import { HANDLE, useDrag } from "./useDrag";
import { OVERLAY_LAYER, quatRotate, type DetPayload, type SimClient } from "@/lib/sim";

/** `[name, why, p, x, y, z, radius, nearSome, nearAll, blockedBy]`
 *  (world_server._explain_cached): nearSome / nearAll are the ranges, m,
 *  inside which a target this size is found at all / on every frame. */
type WhyRow = [string, string, number, number, number, number, number, number?, number?, (string | null)?];
type CamDet = DetPayload & { why?: WhyRow[]; wrist?: { fov: [number, number]; range: number } };

const COLORS: Record<string, string> = {
  seen: "#3ecf6e",
  marginal: "#f2b632",
  small: "#e5534b",
  blocked: "#a47ee0",
  outside: "#8b949e",
  far: "#8b949e",
};
const WEDGE = "#43c2b8";
const WRIST = "#5fb3e8";
/** How far the floor wedge is drawn, m — the room, not the 6 m spec range. */
const WEDGE_M = 3.0;
const FLOOR_Z = 0.004;
const MAX_SEG = 1200;
const MAX_LABELS = 24;
/** The body a wrist camera is mounted on (robots/moss.py ARM_CAMERA_BODY); its
 *  x axis is the optical axis. The field and range come in `det.wrist`. */
const WRIST_BODY = "moss_arm_camera";
/** The filled floor wedge: K strips between the near edge and WEDGE_M. */
const WEDGE_K = 12;

/** The label for a row, worded as what MOSS can or cannot see and WHY —
 *  the first cut ("cap0 · too small at 1.7 m") read as a claim MOSS was
 *  making about the cap, when it is the simulator explaining why MOSS
 *  cannot see it. null = draw the ring only.
 *
 *  `labelsOn` is the 🏷 toggle and it is a REQUIRED argument, not a check the
 *  caller is trusted to make first: the bug this signature exists for is a
 *  label that nobody remembered to gate. There is no way to get text out of
 *  here without answering the question, and a new caller that forgets does
 *  not compile. (It cannot help a component that invents its own text — no
 *  cheap check can — but it makes THIS text impossible to leak.) */
export function whyLabel(row: WhyRow, dist: number, labelsOn: boolean): string | null {
  if (!labelsOn) return null;
  const [name, why, p, , , , , nearSome, , by] = row;
  const d = `${dist.toFixed(1)} m`;
  switch (why) {
    case "seen": return `MOSS sees ${name} · ${d}`;
    case "marginal": return `MOSS sees ${name} in ${Math.round(p * 100)}% of frames · ${d}`;
    case "small":
      return `can't see ${name}: too small from ${d}` + (nearSome ? ` (needs < ${nearSome.toFixed(1)} m)` : "");
    case "blocked":
      return `can't see ${name}: ` + (by === "own body" ? "its own arm is in the way" : by ? `${by} is in the way` : "blocked");
    default: return null;
  }
}

/** A camera-frame direction (x forward, y left, z up) at this bearing
 *  (+left) and elevation (+up), through the pinhole. */
function camDir(q: number[], bearing: number, elev: number): [number, number, number] {
  const v: [number, number, number] = [1, Math.tan(bearing), Math.tan(elev)];
  const n = Math.hypot(v[0], v[1], v[2]);
  return quatRotate(q, [v[0] / n, v[1] / n, v[2] / n]);
}

export function CamOverlay({
  client,
  robotScenes,
  enabled,
}: {
  client: SimClient;
  robotScenes: Record<string, Scene>;
  enabled: boolean;
}) {
  const lines = useRef<THREE.LineSegments>(null);
  useEffect(() => {
    lines.current?.layers.set(OVERLAY_LAYER);
    fill.current?.layers.set(OVERLAY_LAYER);
    rings.current?.layers.set(OVERLAY_LAYER);
  }, []);
  const wristIdx = useMemo(() => {
    const out: Record<string, number> = {};
    for (const [id, sc] of Object.entries(robotScenes)) out[id] = sc.bodies.indexOf(WRIST_BODY);
    return out;
  }, [robotScenes]);
  const pos = useMemo(() => new Float32Array(MAX_SEG * 2 * 3), []);
  const colors = useMemo(() => new Float32Array(MAX_SEG * 2 * 3), []);
  const col = useMemo(() => new THREE.Color(), []);
  const fill = useRef<THREE.Mesh>(null);
  const rings = useRef<THREE.InstancedMesh>(null);
  const mtx = useMemo(() => new THREE.Matrix4(), []);
  const fillPos = useMemo(() => new Float32Array(WEDGE_K * 6 * 3), []);
  const groups = useRef<(THREE.Group | null)[]>([]);
  const tags = useRef<(HTMLDivElement | null)[]>([]);
  const legend = useRef<HTMLDivElement | null>(null);
  // Minimizes to its title bar, like every other /sim panel (remembered).
  const [legendOpen, setLegendOpen] = useState(() => loadJSON("simCamLegendOpen", true));
  useEffect(() => saveJSON("simCamLegendOpen", legendOpen), [legendOpen]);
  // Draggable by its title bar (double-click re-docks), as MOSS's map is.
  const drag = useDrag("simCamLegendPos", legend, 60, 12);

  useFrame(() => {
    const ls = lines.current;
    if (!ls) return;
    const f = client.frame;
    // 🏷 labels covers EVERY piece of text floating in the 3-D scene, not just
    // the ducks' names. These tags were the only labels a moss-yard has —
    // `Duck.tsx` (the flag's other reader) never mounts there, because a
    // non-duck body draws as `SimStage.RobotBody`, which has no name label —
    // so the button governed nothing at all on that page while "MOSS sees
    // can1 · 0.1 m" stayed on screen. The RINGS, sight lines, wedge and wrist
    // cone are not labels and stay with this overlay's own toggle; turning
    // text off must not cost you the geometry you came for.
    //
    // Read per FRAME, not subscribed: a `useSyncExternalStore` inside the r3f
    // tree flushes unreliably when the write comes from the DOM tree
    // (`lib/ui.ts` says why, and `Duck.tsx` reads it the same way).
    const labelsOn = getDuckLabels();
    let n = 0;
    let nl = 0;
    let nf = 0;
    let nr = 0;
    let drewAny = false;
    const seg = (a: number[], b: number[], c: string) => {
      if (n >= MAX_SEG) return;
      const k = n * 6;
      pos[k] = a[0]; pos[k + 1] = a[1]; pos[k + 2] = a[2];
      pos[k + 3] = b[0]; pos[k + 4] = b[1]; pos[k + 5] = b[2];
      col.set(c);
      col.toArray(colors, k);
      col.toArray(colors, k + 3);
      n++;
    };
    // A solid flat ring on the floor: a 1 px line ring vanished on the wood.
    const ring = (x: number, y: number, r: number, c: string) => {
      const im = rings.current;
      if (!im || nr >= MAX_LABELS * 2) return;
      mtx.makeScale(r, r, 1).setPosition(x, y, FLOOR_Z + 0.001);
      im.setMatrixAt(nr, mtx);
      col.set(c);
      im.setColorAt(nr, col);
      nr++;
    };
    if (f && enabled) {
      const sel = getSelectedDuck();
      const ducks = f.ducks.filter((d) => (sel ? d.id === sel : f.ducks.length === 1));
      for (const d of ducks) {
        const det = d.sensors?.det as CamDet | undefined;
        if (!det?.why || !det.cam || det.cam.length < 7) continue;
        drewAny = true;
        const o = det.cam.slice(0, 3);
        const q = det.cam.slice(3, 7);
        const [fh, fv] = (det.fov ?? [60, 45]).map((v) => (v * Math.PI) / 360);
        // The view on the floor: where the frame's bottom edge lands, and the
        // two side edges from there out to WEDGE_M.
        let prevNear: number[] | null = null;
        let prevFar: number[] | null = null;
        const K = WEDGE_K;
        for (let i = 0; i <= K; i++) {
          const b = -fh + (2 * fh * i) / K;
          const dir = camDir(q, b, -fv);
          if (dir[2] >= -1e-6) { prevNear = null; continue; }
          const t = (FLOOR_Z - o[2]) / dir[2];
          const p = [o[0] + dir[0] * t, o[1] + dir[1] * t, FLOOR_Z];
          const h = Math.hypot(dir[0], dir[1]);
          const far = [o[0] + (dir[0] / h) * WEDGE_M, o[1] + (dir[1] / h) * WEDGE_M, FLOOR_Z - 0.001];
          if (prevNear) seg(prevNear, p, WEDGE);
          if (prevNear && prevFar && nf < WEDGE_K) {
            const quad = [prevNear, p, far, prevNear, far, prevFar];
            quad.forEach((v, j) => fillPos.set([v[0], v[1], FLOOR_Z - 0.001], (nf * 6 + j) * 3));
            nf++;
          }
          prevNear = p;
          prevFar = far;
          if (i === 0 || i === K) seg(p, far, WEDGE);
        }
        // Every object, ringed by why; a sight line and a label where it matters.
        for (const row of det.why) {
          const [, why, , x, y, z, r] = row;
          const c = COLORS[why] ?? COLORS.outside;
          ring(x, y, Math.max(r * 1.5, 0.045), c);
          if (why === "seen" || why === "marginal") seg(o, [x, y, z], c);
          const text = whyLabel(row, Math.hypot(x - o[0], y - o[1]), labelsOn);
          if (text && nl < MAX_LABELS) {
            const g = groups.current[nl];
            const tag = tags.current[nl];
            if (g && tag) {
              g.position.set(x, y, z + Math.max(r, 0.02) + 0.05);
              g.visible = true;
              if (tag.style.display) tag.style.display = "";
              if (tag.textContent !== text) tag.textContent = text;
              tag.style.borderColor = c;
              tag.style.color = c;
            }
            nl++;
          }
        }
        // The wrist camera's cone, out to its depth range.
        const wi = d.robot ? wristIdx[d.robot] ?? -1 : -1;
        const wb = wi >= 0 ? d.bodies[wi] : undefined;
        if (wb && det.wrist) {
          const wo = [wb[0], wb[1], wb[2]];
          const wq = [wb[3], wb[4], wb[5], wb[6]];
          const [wh, wv] = det.wrist.fov.map((v) => (v * Math.PI) / 360);
          const R = det.wrist.range;
          const corners = [[wh, wv], [-wh, wv], [-wh, -wv], [wh, -wv]].map(([b, e]) => {
            const dir = camDir(wq, b, e);
            return [wo[0] + dir[0] * R, wo[1] + dir[1] * R, wo[2] + dir[2] * R];
          });
          corners.forEach((p, i) => {
            seg(wo, p, WRIST);
            seg(p, corners[(i + 1) % 4], WRIST);
          });
        }
      }
    }
    const lg = legend.current;
    const want = f && enabled && drewAny ? "block" : "none";
    if (lg && lg.style.display !== want) lg.style.display = want;
    for (let i = nl; i < MAX_LABELS; i++) {
      const g = groups.current[i];
      if (g && g.visible) g.visible = false;
      const tag = tags.current[i];
      if (tag && tag.style.display !== "none") tag.style.display = "none";
    }
    (ls.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    (ls.geometry.getAttribute("color") as THREE.BufferAttribute).needsUpdate = true;
    ls.geometry.setDrawRange(0, n * 2);
    const im = rings.current;
    if (im) {
      im.count = nr;
      im.instanceMatrix.needsUpdate = true;
      if (im.instanceColor) im.instanceColor.needsUpdate = true;
    }
    const fm = fill.current;
    if (fm) {
      (fm.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
      fm.geometry.setDrawRange(0, nf * 6);
    }
  });

  return (
    <>
      <lineSegments ref={lines} frustumCulled={false}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[pos, 3]} />
          <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        </bufferGeometry>
        <lineBasicMaterial vertexColors transparent opacity={0.95} />
      </lineSegments>
      <instancedMesh ref={rings} args={[undefined, undefined, MAX_LABELS * 2]} frustumCulled={false} renderOrder={2}>
        <ringGeometry args={[0.72, 1, 28]} />
        <meshBasicMaterial transparent opacity={0.9} depthWrite={false} side={THREE.DoubleSide} toneMapped={false} />
      </instancedMesh>
      <mesh ref={fill} frustumCulled={false} renderOrder={1}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[fillPos, 3]} />
        </bufferGeometry>
        <meshBasicMaterial color={WEDGE} transparent opacity={0.1} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
      <Html fullscreen zIndexRange={[15, 0]} style={{ pointerEvents: "none" }}>
        {/* Portalled to <body>: inside the canvas's overlay it stacks UNDER the
            page's panels (the inspector covered its title bar), and its drag
            position would be off by the overlay's offset. */}
        {createPortal(
        <div ref={legend} style={{ display: "none", position: "fixed", zIndex: 50, maxWidth: 330, pointerEvents: "auto",
          ...(drag.pos ? { left: drag.pos.x, top: drag.pos.y } : { left: 56, bottom: 64 }),
          font: "10px/1.5 ui-monospace, Menlo, monospace", color: "#c9d1d9", padding: "6px 8px",
          background: "rgba(16,18,22,0.82)", border: "1px solid #2d333b", borderRadius: 4 }}>
          <div
            onPointerDown={drag.onPointerDown}
            onDoubleClick={drag.reset}
            title="drag to move · double-click to re-dock"
            style={{ ...HANDLE, display: "flex", alignItems: "flex-start", color: "#e6edf3", marginBottom: legendOpen ? 3 : 0 }}
          >
            <span style={{ flex: 1 }}>HEAD CAMERA · what MOSS can see</span>
            <PanelToggle open={legendOpen} onToggle={() => setLegendOpen((v) => !v)} what="the camera legend" />
          </div>
          <div style={{ display: legendOpen ? "block" : "none" }}>
          <div><span style={{ color: COLORS.seen }}>●</span> sees it every frame</div>
          <div><span style={{ color: COLORS.marginal }}>●</span> sees it on some frames (small, or at range)</div>
          <div><span style={{ color: COLORS.small }}>●</span> can&apos;t see it: too few pixels at this range</div>
          <div><span style={{ color: COLORS.blocked }}>●</span> can&apos;t see it: something is in the way</div>
          <div><span style={{ color: COLORS.outside }}>●</span> outside the camera&apos;s view</div>
          <div style={{ color: "#8b949e", marginTop: 3 }}>
            The simulator&apos;s explanation, from where things really are — not MOSS&apos;s own belief.
            The boxes in the camera inset name the object for the same reason; MOSS&apos;s own
            detector only reports a class, and in a tidy room every prop is one class.
            Teal wedge: the camera&apos;s view on the floor. Blue cone: the wrist depth camera (0.6 m).
          </div>
          </div>
        </div>,
          document.body,
        )}
      </Html>
      {Array.from({ length: MAX_LABELS }, (_, i) => (
        <group key={i} ref={(g) => { groups.current[i] = g; }} visible={false}>
          <Html center zIndexRange={[20, 0]} style={{ pointerEvents: "none" }}>
            <div
              ref={(el) => { tags.current[i] = el; }}
              style={{
                display: "none",
                font: "10px ui-monospace, Menlo, monospace",
                whiteSpace: "nowrap",
                padding: "1px 5px",
                border: "1px solid",
                borderRadius: 3,
                background: "rgba(16,18,22,0.78)",
              }}
            />
          </Html>
        </group>
      ))}
    </>
  );
}
