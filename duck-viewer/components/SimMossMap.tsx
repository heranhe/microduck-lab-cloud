"use client";

// MOSS'S OWN MAP of the room — what it BELIEVES, under the map toggle (M).
// Asked on /sim 2026-09-28: "we need a UX for this so we know what it's
// talking about ... a little model of the room that it maps". The camera
// overlay (SimCamView) is the SIMULATOR's explanation of what the camera can
// see; this is the robot's side, from `brain.inputs.map`
// (brain/tidy_moss.TidyMoss.map_payload), all in its odometry frame:
//
//   * a mini-map panel: the walls and anything else its DEPTH has found
//     solid (light cells), what it has bumped into (orange), the work area
//     it worked out from them — or, on a MOSS with no depth, the scenario's
//     walls it was handed, drawn dashed and labelled "given" — where it has
//     looked (brighter = more recently), every object it remembers (solid =
//     confirmed by a second sighting, hollow = seen once, red x = given up
//     on), where it thinks it is and what its camera covers, the rim route,
//     and the leg it is on
//   * the same memory in the room: a square on the floor at each remembered
//     spot, and a line to where it is driving — so belief can be compared
//     with where things really are
//
// One robot: the selected one, or the only one with a map. Per-frame drawing
// follows the page's rule: no React state per frame (canvas + one
// LineSegments, both written from useFrame).

import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { createPortal } from "react-dom";
import * as THREE from "three";

import { loadJSON, saveJSON } from "@/lib/persist";
import { getSelectedDuck } from "@/lib/select";
import { PanelToggle } from "./Panel";
import { HANDLE, useDrag } from "./useDrag";
import { OVERLAY_LAYER, type SimClient } from "@/lib/sim";

/** [x, y, size, hits, confirmed, age_s, writtenOff] */
type MemRow = [number, number, number, number, number, number, number];
/** `moss_search.RoomMap.payload`: the depth's occupancy grid. */
export interface RoomGrid {
  x0: number;
  y0: number;
  cell: number;
  n: number;
  /** Flat indices (iy * n + ix) of the cells the depth found solid. */
  solid: number[];
  /** Where a patrol leg stalled against something the depth did not see. */
  felt: [number, number][];
  /** Share of a full turn the scans have looked along, 0..1. */
  swept: number;
}
export interface MossMap {
  area: [number, number, number, number] | null;
  /** Where the work area came from: its own depth, or the scenario. */
  areaSrc?: "sensed" | "given" | null;
  room?: RoomGrid | null;
  /**
   * The wrist camera while it scans at rest, and ONLY while it scans:
   * [x, y, heading of its view, near radius, far radius]. The radii come from
   * the robot (`tidy_moss._wrist_floor_band`) because the floor it can report
   * is a BAND, not a wedge from the lens: aimed down from 0.30 m up, its
   * nearest floor is ~0.21 m out and the detector's range cuts it off at
   * ~0.54 m. Drawing the full wedge claimed the floor beside the tracks,
   * which the camera cannot see.
   */
  wcam?: [number, number, number, number, number] | null;
  rim: [number, number][];
  mem: MemRow[];
  target: [number, number] | null;
  goal: [number, number] | null;
  look: [number, number] | null;
  leg: string | null;
  cov: { n: [number, number]; age: string } | null;
}

const W = 230;
/** MOSS's wrist camera, `moss.ARM_CAMERA_HFOV_DEG` — the same 70 deg
 *  `WRIST_CAMERAS` renders its inset with. */
const WRIST_HFOV_DEG = 70;
const TEAL = "#43c2b8";
const AMBER = "#f2b632";
const RED = "#e5534b";
const GREY = "#8b949e";
const ORANGE = "#f0883e";
const WALL = "#c9d1d9";
const MAX_SEG = 2000;
const FLOOR_Z = 0.006;

const LEG_TEXT: Record<string, string> = {
  spin: "turning on the spot",
  choose: "choosing where to look",
  memory: "going back to a remembered object",
  rim: "patrolling the rim",
};

/** What the panel's status line says. */
export function mapStatus(m: MossMap): string {
  const conf = m.mem.filter((r) => r[4] && !r[6]).length;
  const once = m.mem.filter((r) => !r[4]).length;
  const leg = m.target ? "going for an object" : m.leg ? LEG_TEXT[m.leg] ?? m.leg : "busy with an object";
  return `${conf} remembered${once ? ` · ${once} seen once` : ""} · ${leg}`;
}

/** Where the walls on the map come from, in words. */
export function wallsStatus(m: MossMap): string {
  if (m.areaSrc === "sensed") return "Walls: found by its depth camera.";
  if (m.areaSrc === "given") return "Walls: GIVEN to it (dashed) — this MOSS has no depth.";
  if (m.room) return `Walls: still mapping — has looked along ${Math.round(m.room.swept * 100)}% of a turn.`;
  return "Walls: none yet.";
}

/** The outline of the floor the wrist camera can report, metres: an annular
 *  sector between the near and far radius the robot publishes, so the map
 *  never claims the floor beside the tracks (which is below its view). */
export function wristBand(
  wcam: NonNullable<MossMap["wcam"]>, steps = 6,
): [number, number][] {
  const [x, y, a, near, far] = wcam;
  const half = (WRIST_HFOV_DEG / 2 * Math.PI) / 180;
  const at = (r: number, k: number): [number, number] => {
    const t = a - half + (2 * half * k) / steps;
    return [x + r * Math.cos(t), y + r * Math.sin(t)];
  };
  const out: [number, number][] = [];
  for (let k = 0; k <= steps; k++) out.push(at(far, k));
  for (let k = steps; k >= 0; k--) out.push(at(near, k));
  return out;
}

/** The panel's extent: the work area once there is one, else what the depth
 *  has found so far (and the robot), else a room-sized box round the robot. */
export function mapView(m: MossMap, pose?: number[] | null): [number, number, number, number] | null {
  if (m.area) return m.area;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  const add = (x: number, y: number) => {
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  };
  const g = m.room;
  if (g) for (const k of g.solid) add(g.x0 + ((k % g.n) + 0.5) * g.cell, g.y0 + (Math.floor(k / g.n) + 0.5) * g.cell);
  if (pose) {
    add(pose[0] - 0.5, pose[1] - 0.5);
    add(pose[0] + 0.5, pose[1] + 0.5);
  }
  if (!Number.isFinite(x0)) return null;
  return [x0, x1, y0, y1];
}

export function MossMapOverlay({ client, enabled }: { client: SimClient; enabled: boolean }) {
  const panel = useRef<HTMLDivElement | null>(null);
  const cv = useRef<HTMLCanvasElement | null>(null);
  const status = useRef<HTMLDivElement | null>(null);
  const walls = useRef<HTMLDivElement | null>(null);
  // Minimizes to its title bar, like every other /sim panel (remembered).
  const [open, setOpen] = useState(() => loadJSON("simMossMapOpen", true));
  useEffect(() => saveJSON("simMossMapOpen", open), [open]);
  const openRef = useRef(open);
  openRef.current = open;
  // Draggable by its title bar like the inspector (double-click re-docks);
  // the position persists. `useDrag` works in viewport coordinates and this
  // panel lives in the canvas's overlay, so subtract where that overlay is.
  const drag = useDrag("simMossMapPos", panel, 60, 12);
  const lines = useRef<THREE.LineSegments>(null);
  useEffect(() => {
    lines.current?.layers.set(OVERLAY_LAYER);
  }, []);
  const pos = useMemo(() => new Float32Array(MAX_SEG * 2 * 3), []);
  const colors = useMemo(() => new Float32Array(MAX_SEG * 2 * 3), []);
  const col = useMemo(() => new THREE.Color(), []);
  const tick = useRef(0);

  useFrame(() => {
    const f = client.frame;
    const sel = getSelectedDuck();
    const d = f && enabled
      ? f.ducks.find((x) => (sel ? x.id === sel : true) && (x.brain?.inputs as { map?: MossMap })?.map)
      : undefined;
    const m = d ? (d.brain.inputs as { map?: MossMap }).map : undefined;
    const pose = d?.odomEst;
    const pn = panel.current;
    const view = m ? mapView(m, pose) : null;
    const show = view ? "block" : "none";
    if (pn && pn.style.display !== show) pn.style.display = show;

    // --- in the room: memory squares, rim route, the leg being driven
    let n = 0;
    const seg = (a: number[], b: number[], c: string) => {
      if (n >= MAX_SEG) return;
      const k = n * 6;
      pos[k] = a[0]; pos[k + 1] = a[1]; pos[k + 2] = FLOOR_Z;
      pos[k + 3] = b[0]; pos[k + 4] = b[1]; pos[k + 5] = FLOOR_Z;
      col.set(c);
      col.toArray(colors, k);
      col.toArray(colors, k + 3);
      n++;
    };
    if (m) {
      // what the depth found solid, and what it bumped into
      const g = m.room;
      if (g) {
        const h = g.cell / 2 - 0.004;
        for (const k of g.solid) {
          const x = g.x0 + ((k % g.n) + 0.5) * g.cell, y = g.y0 + (Math.floor(k / g.n) + 0.5) * g.cell;
          seg([x - h, y - h], [x + h, y + h], "#6e7681"); seg([x - h, y + h], [x + h, y - h], "#6e7681");
        }
        for (const [x, y] of g.felt) {
          seg([x - 0.06, y - 0.06], [x + 0.06, y + 0.06], ORANGE); seg([x - 0.06, y + 0.06], [x + 0.06, y - 0.06], ORANGE);
        }
      }
      for (const [x, y, size, , conf, , off] of m.mem) {
        const h = Math.max(size * 0.8, 0.03);
        const c = off ? RED : conf ? TEAL : GREY;
        seg([x - h, y - h], [x + h, y - h], c); seg([x + h, y - h], [x + h, y + h], c);
        seg([x + h, y + h], [x - h, y + h], c); seg([x - h, y + h], [x - h, y - h], c);
      }
      if (m.rim.length > 1) m.rim.forEach((p, i) => seg(p, m.rim[(i + 1) % m.rim.length], "#2f6f6a"));
      if (pose && m.goal) seg([pose[0], pose[1]], m.goal, AMBER);
    }
    const ls = lines.current;
    if (ls) {
      (ls.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
      (ls.geometry.getAttribute("color") as THREE.BufferAttribute).needsUpdate = true;
      ls.geometry.setDrawRange(0, n * 2);
    }

    // --- the panel (every 3rd frame is plenty)
    if (!m || !view || !openRef.current || (tick.current++ % 3) !== 0) return;
    const c = cv.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    const [x0, x1, y0, y1] = view;
    const pad = 0.1;
    // Fit the width, but never taller than 220 px: a half-mapped room can be
    // a long thin strip before the far walls are in.
    const sx = Math.min(W / (x1 - x0 + 2 * pad), 220 / (y1 - y0 + 2 * pad));
    const H = Math.round((y1 - y0 + 2 * pad) * sx);
    if (c.height !== H) c.height = H;
    const px = (x: number) => (x - x0 + pad) * sx;
    const py = (y: number) => H - (y - y0 + pad) * sx;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, W, H);
    // where it has looked
    if (m.cov && m.area) {
      const [nx, ny] = m.cov.n;
      const cw = (x1 - x0) / nx, ch = (y1 - y0) / ny;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const a = m.cov.age[j * nx + i];
        if (!a || a === "-") continue;
        ctx.fillStyle = `rgba(67,194,184,${(0.42 - Number(a) * 0.04).toFixed(3)})`;
        ctx.fillRect(px(x0 + i * cw), py(y0 + (j + 1) * ch), cw * sx + 0.5, ch * sx + 0.5);
      }
    }
    // the work area: sensed (thin, from the cells below), or GIVEN (dashed)
    if (m.area) {
      const [ax0, ax1, ay0, ay1] = m.area;
      const given = m.areaSrc === "given";
      ctx.strokeStyle = given ? GREY : "#3d444d";
      ctx.lineWidth = given ? 1.5 : 1;
      if (given) ctx.setLineDash([6, 4]);
      ctx.strokeRect(px(ax0), py(ay1), (ax1 - ax0) * sx, (ay1 - ay0) * sx);
      ctx.setLineDash([]);
      if (given) {
        ctx.fillStyle = GREY;
        ctx.font = "9px ui-monospace, Menlo, monospace";
        ctx.fillText("given, not sensed", px(ax0) + 4, py(ay1) + 11);
      }
    }
    // what its depth found solid, and what it bumped into
    const g = m.room;
    if (g) {
      ctx.fillStyle = WALL;
      const w = Math.max(g.cell * sx, 1.5);
      for (const k of g.solid) {
        const x = g.x0 + (k % g.n) * g.cell, y = g.y0 + (Math.floor(k / g.n) + 1) * g.cell;
        ctx.fillRect(px(x), py(y), w, w);
      }
      ctx.strokeStyle = ORANGE;
      ctx.lineWidth = 2;
      for (const [x, y] of g.felt) {
        const X = px(x), Y = py(y);
        ctx.beginPath();
        ctx.moveTo(X - 4, Y - 4); ctx.lineTo(X + 4, Y + 4);
        ctx.moveTo(X + 4, Y - 4); ctx.lineTo(X - 4, Y + 4);
        ctx.stroke();
      }
    }
    // the rim route
    if (m.rim.length > 1) {
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "#2f6f6a";
      ctx.lineWidth = 1;
      ctx.beginPath();
      m.rim.forEach(([x, y], i) => (i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))));
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // the robot, its camera, and its leg
    if (pose) {
      const [rx, ry, yaw] = pose;
      ctx.fillStyle = "rgba(67,194,184,0.10)";
      ctx.beginPath();
      ctx.moveTo(px(rx), py(ry));
      const half = (43.5 * Math.PI) / 180, R = 1.0;
      for (let k = 0; k <= 8; k++) {
        const a = yaw - half + (2 * half * k) / 8;
        ctx.lineTo(px(rx + R * Math.cos(a)), py(ry + R * Math.sin(a)));
      }
      ctx.closePath();
      ctx.fill();
      // the wrist camera, scanning while the arm rests: the BAND of floor it
      // can actually report, not a wedge from the lens (see `wcam`)
      if (m.wcam) {
        ctx.fillStyle = "rgba(88,166,255,0.16)";
        ctx.beginPath();
        wristBand(m.wcam).forEach(([x, y], k) =>
          k ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y)));
        ctx.closePath();
        ctx.fill();
      }
      if (m.goal) {
        ctx.strokeStyle = AMBER;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(px(rx), py(ry));
        ctx.lineTo(px(m.goal[0]), py(m.goal[1]));
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.save();
      ctx.translate(px(rx), py(ry));
      ctx.rotate(-yaw);
      ctx.fillStyle = "#e6edf3";
      ctx.beginPath();
      ctx.moveTo(9, 0); ctx.lineTo(-6, 5); ctx.lineTo(-6, -5);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
    // what it remembers
    for (const [x, y, , , conf, , off] of m.mem) {
      const X = px(x), Y = py(y);
      if (off) {
        ctx.strokeStyle = RED;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(X - 4, Y - 4); ctx.lineTo(X + 4, Y + 4);
        ctx.moveTo(X + 4, Y - 4); ctx.lineTo(X - 4, Y + 4);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(X, Y, 4, 0, Math.PI * 2);
        if (conf) { ctx.fillStyle = TEAL; ctx.fill(); }
        else { ctx.strokeStyle = GREY; ctx.lineWidth = 1.2; ctx.stroke(); }
      }
    }
    if (m.target) {
      ctx.strokeStyle = AMBER;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px(m.target[0]), py(m.target[1]), 7, 0, Math.PI * 2);
      ctx.stroke();
    }
    const txt = mapStatus(m);
    const s = status.current;
    if (s && s.textContent !== txt) s.textContent = txt;
    const wt = wallsStatus(m);
    const ws = walls.current;
    if (ws && ws.textContent !== wt) ws.textContent = wt;
  });

  return (
    <>
      <lineSegments ref={lines} frustumCulled={false}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[pos, 3]} />
          <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        </bufferGeometry>
        <lineBasicMaterial vertexColors transparent opacity={0.9} />
      </lineSegments>
      <Html fullscreen style={{ pointerEvents: "none" }}>
        {/* Portalled to <body>: inside the canvas's overlay it stacks UNDER the
            page's panels (the inspector covered its title bar), and its drag
            position would be off by the overlay's offset. */}
        {createPortal(
        <div ref={panel} style={{ display: "none", position: "fixed", zIndex: 50, width: W + 16, pointerEvents: "auto",
          ...(drag.pos ? { left: drag.pos.x, top: drag.pos.y } : { right: 12, bottom: 96 }),
          font: "10px/1.45 ui-monospace, Menlo, monospace", color: "#c9d1d9", padding: 8,
          background: "rgba(16,18,22,0.88)", border: "1px solid #2d333b", borderRadius: 4 }}>
          <div
            onPointerDown={drag.onPointerDown}
            onDoubleClick={drag.reset}
            title="drag to move · double-click to re-dock"
            style={{ ...HANDLE, display: "flex", alignItems: "flex-start", color: "#e6edf3" }}
          >
            <span style={{ flex: 1 }}>MOSS&apos;S MAP · what it believes</span>
            <PanelToggle open={open} onToggle={() => setOpen((v) => !v)} what="MOSS's map" />
          </div>
          <div style={{ display: open ? "block" : "none" }}>
          <div ref={status} style={{ color: GREY, marginBottom: 4 }} />
          <canvas ref={cv} width={W} height={180} style={{ display: "block", width: W }} />
          <div style={{ marginTop: 4 }}>
            <span style={{ color: TEAL }}>●</span> remembered{"  "}
            <span style={{ color: GREY }}>○</span> seen once{"  "}
            <span style={{ color: RED }}>✕</span> gave up{"  "}
            <span style={{ color: AMBER }}>◯</span> going for
          </div>
          <div>
            <span style={{ color: WALL }}>■</span> solid to its depth{"  "}
            <span style={{ color: ORANGE }}>✕</span> bumped into{"  "}
            <span style={{ color: "#58a6ff" }}>◢</span> wrist camera scanning
          </div>
          <div ref={walls} style={{ color: "#e6edf3" }} />
          <div style={{ color: GREY }}>
            Shaded: floor it has looked at (brighter = more recent). Dashed teal: its rim patrol.
          </div>
          </div>
        </div>,
          document.body,
        )}
      </Html>
    </>
  );
}
