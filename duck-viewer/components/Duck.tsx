"use client";

// One duck: a Three.js group per MuJoCo body, poses smoothed toward the
// latest streamed frame. All geoms of a body are merged into ONE geometry at
// load time, so a duck is ~16 draw calls — the naive per-geom version
// (70 meshes/duck, all casting shadows) lost the WebGL context with 8 ducks
// on screen. Per-geom MJCF material colors survive the merge as a
// vertex-color channel, so the eye ring / mouth / shells keep their own
// colors without costing extra draw calls.

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import type { DuckFrame, Scene, SceneGeom } from "@/lib/lab";
import { assignDrag } from "@/lib/assign";
import { captureWantsCleanFrame } from "@/lib/record";
import { getSelectedDuck } from "@/lib/select";
import { propRadius, propShape } from "@/lib/prop";
import {
  episodeEdge,
  episodeFlashAlpha,
  episodeFlashLabel,
  type EpisodeCause,
} from "@/lib/episode";
import { getDuckLabels } from "@/lib/ui";
import { CREASE_ANGLE_DEG, G1_KINDS, g1PartKind, useG1Materials, weldAndSmooth, weldTolerance } from "./G1Look";
import { MARS_KINDS, marsPartKind, useMarsMaterials } from "./MarsLook";
import { duckMouths, MOUTH_TRAVEL_RAD } from "@/lib/mouth";
import { SHELL_MATERIALS, TEAM_COLORWAYS, TRIM_MATERIALS, teamColor, type TeamName, POSE_SMOOTH_HZ, simRate } from "@/lib/sim";

/** three's cylinder mesh is Y-up and this scene is Z-up, so every
 *  prop carries this quarter turn about X underneath its own
 *  orientation. */
const CYL_Z_UP = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0), Math.PI / 2,
);

// FALLBACK body-name → color, used only against servers that predate rgba
// streaming (whole body painted one guessed color).
function bodyColor(name: string): string {
  if (/foot|ankle/.test(name)) return "#e8862e"; // webbed orange
  if (/head|beak/.test(name)) return "#f5efe0";
  if (/neck/.test(name)) return "#e8862e";
  if (/trunk/.test(name)) return "#f5efe0"; // cream body
  if (/hip|knee|leg/.test(name)) return "#4a4e57"; // dark joints
  return "#c8c2b4";
}

// The MJCF materials are OnShape-export appearances, and a few don't match
// the printed robot: the yellow eye ring exports dark grey, the soft TPU
// mouth pink, and the beak/shoes a flat gold where the real parts are
// orange with yellow soles. Override those by material name; everything
// else renders straight from the streamed rgba.
// Per-material colour overrides by MJCF material name. The 2026-09 upstream
// CAD re-export (microduck_rl #29) carries the right colours itself — orange
// beak and shoes, yellow eye ring and soles, light-grey face — so the table is
// empty; it stays as the place to put the next export's mistakes.
const MATERIAL_FIX: Record<string, string> = {};

/** A team's repaint of the printed parts, by MJCF material name: the four
 *  shell parts (head, trunk, legs, hips) take the colorway and the trim parts
 *  (beak, feet, ankles, soles) take its trim — every printed part gets one of
 *  the two, so a duck is one colour from the beak down. The server paints the
 *  same names in the composed world (world/compose.py); the viewer has to do
 *  its own because it draws every duck from ONE single-robot scene. */
function teamPaint(team: string | null | undefined): Record<string, string> {
  const look = teamColor(team) && team && team in TEAM_COLORWAYS ? TEAM_COLORWAYS[team as TeamName] : null;
  if (!look) return {};
  const out: Record<string, string> = {};
  for (const m of SHELL_MATERIALS) out[m] = look.shell;
  for (const m of TRIM_MATERIALS) out[m] = look.trim;
  return out;
}

/** Resolved sRGB→linear color for one geom (team → override → MJCF rgba → fallback). */
function geomColor(g: SceneGeom, bodyName: string, out: THREE.Color, paint: Record<string, string> = {}): THREE.Color {
  const team = g.mat ? paint[g.mat] : undefined;
  if (team) return out.set(team);
  const fix = g.mat ? MATERIAL_FIX[g.mat] : undefined;
  if (fix) return out.set(fix);
  if (g.rgba) return out.setRGB(g.rgba[0], g.rgba[1], g.rgba[2], THREE.SRGBColorSpace);
  return out.set(bodyColor(bodyName));
}

export interface BodyGeometry {
  name: string;
  geometry: THREE.BufferGeometry | null;
  /** Which material set draws this body (lib/robots.robotLook):
   *
   *  - undefined / "duck" — the merged vertex-colour mesh the duck has
   *    always been;
   *  - "g1" — welded + smoothed, groups indexing G1_KINDS materials
   *    (components/G1Look.tsx), the same look as the /sim page's G1;
   *  - "mars" — the same shape of thing for MARS_KINDS
   *    (components/MarsLook.tsx): a graphite chassis and head that survive
   *    the dark stage, the colorway's accent on the arm and gripper, and the
   *    frame markers drawn as nothing;
   *  - "generic" — welded + smoothed like the G1 but painted per geom from
   *    the scene dump's own `rgba`. That is how a Menagerie model arrives in
   *    its MJCF's colours, with no component and no colour table per robot.
   */
  look?: "g1" | "generic" | "mars";
}

/** Merge every geom of every body into one geometry per body (body-local
 *  frame), painting each geom's material color into a vertex-color channel.
 *
 *  `team` repaints the printed parts in that colorway. The colour is baked
 *  into the geometry, so a caller wanting two teams on screen builds one set
 *  PER COLORWAY and shares it across that team's ducks — not one per duck.
 *  Eight ducks with a set each is what lost the WebGL context before the
 *  bodies were merged at all (duck-viewer/README.md). */
export function buildBodyGeometries(
  scene: Scene,
  team?: string | null,
  opts: { look?: "g1" | "generic" | "mars" } = {}
): BodyGeometry[] {
  const paint = teamPaint(team);
  // A non-duck body: welded + smoothed, one draw per material group. The G1
  // and MARS group by PART KIND (a visor or a chassis, each with its own
  // material); "generic" groups by nothing and is drawn from its own vertex
  // colours, so a robot the viewer has never seen still arrives in its own
  // paint. `kinds` is both the switch and the index table — a look with a
  // material array is exactly a look with a kind list.
  const kinds: readonly string[] | null =
    opts.look === "g1" ? G1_KINDS : opts.look === "mars" ? MARS_KINDS : null;
  const cad = kinds !== null || opts.look === "generic";
  // Vertices arrive in metres (the duck) or in millimetre ints with a
  // vertScale (the G1 — a 21 MB dump instead of 78 MB of floats). Scaling
  // here rather than at the call site is what keeps a second robot from
  // arriving 1000x too big, off camera, with nothing in the console.
  const vs = scene.vertScale ?? 1;
  // The weld has to know the lattice the dump was quantised to — see
  // G1Look.weldTolerance. A flat tolerance that is fine on a millimetre dump
  // merges real vertices on a finer one.
  const tol = weldTolerance(vs);
  // MARS is the boxy one: its shells are bevelled panels, and smoothing
  // across those bevels is what made its head arrive fluted. See
  // G1Look.weldAndSmooth for why this is opt-in rather than the default.
  const crease = opts.look === "mars" ? CREASE_ANGLE_DEG : undefined;
  const meshGeos = scene.meshes.map((m) => {
    const g = new THREE.BufferGeometry();
    const v = vs === 1 ? m.v : m.v.map((x) => x * vs);
    g.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
    g.setIndex(m.f);
    return g;
  });
  const mat = new THREE.Matrix4();
  const quat = new THREE.Quaternion();
  const col = new THREE.Color();
  const out = scene.bodies.map((name, b) => {
    const parts = scene.geoms
      .filter((g) => g.body === b)
      .map((g) => {
        const geo = cad
          ? weldAndSmooth(meshGeos[g.mesh], tol, crease)
          : meshGeos[g.mesh].clone();
        quat.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]); // wxyz → xyzw
        mat.compose(new THREE.Vector3(...g.pos), quat, new THREE.Vector3(1, 1, 1));
        geo.applyMatrix4(mat);
        geomColor(g, name, col, paint);
        const n = geo.getAttribute("position").count;
        const colors = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) col.toArray(colors, i * 3);
        geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
        const kind =
          opts.look === "g1"
            ? G1_KINDS.indexOf(g1PartKind(g.name, g.mat, col))
            : opts.look === "mars"
              ? MARS_KINDS.indexOf(marsPartKind(name, g.name))
              : 0;
        return { geo, kind };
      });
    if (!parts.length) return { name, geometry: null };
    if (!cad) {
      const merged = mergeGeometries(parts.map((p) => p.geo), false);
      parts.forEach((p) => p.geo.dispose());
      merged.computeVertexNormals();
      return { name, geometry: merged };
    }
    if (!kinds) {
      // Generic: keep the welded normals (re-smoothing the merge would blend
      // across part seams, which is what makes a CAD robot look melted) and
      // merge into ONE group — the per-geom colour already rode in on the
      // vertex-colour channel, so one draw call paints the whole body.
      const merged = mergeGeometries(parts.map((p) => p.geo), false);
      parts.forEach((p) => p.geo.dispose());
      return { name, geometry: merged, look: "generic" as const };
    }
    // G1 / MARS: keep each part's welded normals (re-smoothing the merge would
    // blend across part seams) and group the parts by kind, one draw per
    // material.
    parts.sort((a, b) => a.kind - b.kind);
    const merged = mergeGeometries(parts.map((p) => p.geo), true);
    parts.forEach((p) => p.geo.dispose());
    const groups: { start: number; count: number; materialIndex: number }[] = [];
    merged.groups.forEach((grp, i) => {
      const last = groups[groups.length - 1];
      if (last && last.materialIndex === parts[i].kind) last.count += grp.count;
      else groups.push({ start: grp.start, count: grp.count, materialIndex: parts[i].kind });
    });
    merged.clearGroups();
    groups.forEach((grp) => merged.addGroup(grp.start, grp.count, grp.materialIndex));
    return { name, geometry: merged, look: opts.look as "g1" | "mars" };
  });
  meshGeos.forEach((g) => g.dispose());
  return out;
}

export function Duck({
  bodies,
  frameRef,
  offset,
  label,
  duckId,
}: {
  bodies: BodyGeometry[];
  frameRef: React.MutableRefObject<DuckFrame | null>;
  offset: [number, number]; // grid offset in MuJoCo XY
  label: string;
  duckId: string; // stable stream id ("d0"…, "trainee") — assignment target
}) {
  const bodyRefs = useRef<(THREE.Group | null)[]>([]);
  // The hinged lower bill (world/compose.py `split_jaw`): its group takes the
  // streamed pose like every body, and its MESH takes the voice on top — a
  // rotation about the body origin, which is the pivot, on the hinge axis.
  const mouthIdx = useMemo(() => bodies.findIndex((b) => b.name === "mouth"), [bodies]);
  const billRef = useRef<THREE.Mesh>(null);
  const g1Materials = useG1Materials(bodies.some((b) => b.look === "g1"));
  // MARS's material table. The colorway is a VIEWER default today
  // (MARS_DEFAULT_COLORWAY, Innate's Blue / White hero): a duck's colorway
  // rides in on the frame's `team`, and that channel carries the four Pollen
  // DUCK colorways — the lab validates it against them and rejects anything
  // else (world/scenario.py), so a MARS cannot borrow it. A per-slot choice
  // would pass a `colorway` here from a new field on the roster row.
  const marsMaterials = useMarsMaterials(bodies.some((b) => b.look === "mars"));
  // A generic body is lit like the G1's shell but takes its colour from the
  // geometry, so one material serves every one of them.
  const genericMaterial = useMemo(
    () => new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.42, metalness: 0.2 }),
    []
  );
  const labelRef = useRef<THREE.Group>(null);
  const labelDivRef = useRef<HTMLDivElement>(null);
  const spawnDivRef = useRef<HTMLDivElement>(null);
  // EPISODE BOUNDARIES. A clip ending looks exactly like the ball being
  // kicked away — both teleport duck and ball in one frame — so the boundary
  // is annotated rather than left to be inferred. State in refs, not React:
  // this updates per-frame inside the Canvas tree like the rest of the label.
  const resetFlashRef = useRef<{ at: number; cause: EpisodeCause } | null>(null);
  const lastEpisodeRef = useRef<{ step: number; falls: number } | null>(null);
  const resetDivRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const ballRef = useRef<THREE.Mesh>(null);
  const propRef = useRef<THREE.Mesh>(null);
  const boxRef = useRef<THREE.Mesh>(null);
  const ghostRef = useRef<THREE.Mesh>(null);
  const ghostPropRef = useRef<THREE.Mesh>(null);
  const ghostRingRef = useRef<THREE.Mesh>(null);
  // `lineSegments`, not `line`: the JSX intrinsic `line` resolves to SVG's,
  // so it type-errors on `visible` (r3f's own long-standing name clash).
  const ghostLineRef = useRef<THREE.LineSegments>(null);
  const targetRef = useRef<THREE.Mesh>(null);
  const targetPinRef = useRef<THREE.Mesh>(null);
  const tmpP = useMemo(() => new THREE.Vector3(), []);
  const tmpQ = useMemo(() => new THREE.Quaternion(), []);
  // The error segment's two endpoints. PER DUCK and stable across renders:
  // a module-level array would be shared by every duck on the stage and
  // they would all draw the last one's line, and a fresh array inline in
  // `args` would make r3f rebuild the attribute on every re-render.
  const ghostLinePositions = useMemo(() => new Float32Array(6), []);

  useFrame((_, dt) => {
    const duck = frameRef.current;
    if (!duck) return;
    // Scaled by the world's speed: the filter's lag is fixed in WALL
    // time, the sim time a frame carries is not (lib/sim.ts POSE_SMOOTH_HZ).
    const alpha = 1 - Math.exp(-POSE_SMOOTH_HZ * simRate.speed * Math.min(dt, 0.1));
    duck.bodies.forEach((pose, b) => {
      const grp = bodyRefs.current[b];
      if (!grp) return;
      tmpP.set(pose[0], pose[1], pose[2]);
      tmpQ.set(pose[4], pose[5], pose[6], pose[3]); // wxyz → xyzw
      grp.position.lerp(tmpP, alpha);
      grp.quaternion.slerp(tmpQ, alpha);
    });
    // Lip-sync: open the bill to the wider of what the servo is doing and
    // what the voice asks, so a duck carrying a toy keeps its grip while it
    // chirps, and a silent duck shows exactly the physics. The servo's
    // opening is already in the streamed pose, so only the EXCESS is added.
    if (billRef.current) {
      const voice = duckMouths.open(duckId, performance.now() / 1000);
      billRef.current.rotation.y = Math.max(0, voice - (duck.mouth ?? 0)) * MOUTH_TRAVEL_RAD;
    }
    // Float the label above the trunk (body 1 = trunk_base in this model).
    const trunk = duck.bodies[1];
    if (labelRef.current && trunk) {
      tmpP.set(trunk[0], trunk[1], trunk[2] + 0.22);
      labelRef.current.position.lerp(tmpP, alpha);
    }
    // Drop-target highlight while a policy chip is dragged/armed: floor ring
    // under the duck + emphasized label. Driven straight off the shared store
    // (no React state — this flips at pointer speed). The same ring doubles
    // as the click-selection marker (Delete removes the selected duck); an
    // active assign hover wins the color so the drop target stays legible.
    const hovered = assignDrag.mode !== null && assignDrag.hoverDuck === duckId;
    const selected = getSelectedDuck() === duckId;
    // The ring lives IN the 3D scene, so unlike the DOM labels it would land
    // in captured footage — hide it while a 🎥 take is framing/rolling.
    // (📷 snapshots hide it themselves via the hideInCapture tag below.)
    const filming = captureWantsCleanFrame();
    if (ringRef.current) {
      ringRef.current.visible = (hovered || selected) && !filming;
      (ringRef.current.material as THREE.MeshBasicMaterial).color.set(
        hovered ? "#7db8d8" : "#e8b24a"
      );
      if (trunk) {
        tmpP.set(trunk[0], trunk[1], 0.004);
        ringRef.current.position.lerp(tmpP, alpha);
      }
    }
    // The ball this trainee is working on: [x, y, z, r] in the duck's own
    // frame. find_ball's is virtual, a ball-scene recipe's is a real body —
    // one field, because "what ball is this duck on" is one question.
    const ball = duck.ball;
    // WHICH of the three shapes this payload is, decided ONCE. A 5th number
    // is the prop's half-height (a cylinder), a 10th makes it a box; the
    // table and the reason this is not three ad-hoc length tests here are
    // in `lib/prop.ts`. Asked separately, the sphere's own test only ruled
    // out the cylinder — so every box was drawn with the orange ball still
    // switched on inside it.
    const shape = propShape(ball);
    const isBox = shape === "box";
    const isCan = shape === "cylinder";
    if (boxRef.current) {
      boxRef.current.visible = isBox;
      if (ball && isBox) {
        tmpP.set(ball[0], ball[1], ball[2]);
        boxRef.current.position.lerp(tmpP, alpha);
        // half-extents, and the geometry is 2 units across, so scale = half
        boxRef.current.scale.set(ball[3], ball[4], ball[5]);
        tmpQ.set(ball[7], ball[8], ball[9], ball[6]);   // three is (x,y,z,w)
        boxRef.current.quaternion.slerp(tmpQ, alpha);
      }
    }
    if (propRef.current) {
      propRef.current.visible = isCan;
      if (ball && isCan) {
        tmpP.set(ball[0], ball[1], ball[2]);
        propRef.current.position.lerp(tmpP, alpha);
        const r = ball[3] || 0.033;
        propRef.current.scale.set(r, ball[4] || 0.0575, r);
        // three's cylinder is Y-up and this scene is Z-up, so the mesh gets a
        // fixed quarter turn about X...
        if (ball.length >= 9) {
          // ...THEN THE CAN'S OWN ORIENTATION on top of it. Without this the
          // can was drawn bolt upright whatever the physics did, so a lab
          // spawning half its cans on their side looked like one that never
          // varied — which is exactly what it was reported as, twice. The
          // server sends [x, y, z, r, halfH, qw, qx, qy, qz]; a payload that
          // stops at five keeps the old upright behaviour.
          tmpQ.set(ball[6], ball[7], ball[8], ball[5]);   // three is (x,y,z,w)
          tmpQ.multiply(CYL_Z_UP);
          propRef.current.quaternion.slerp(tmpQ, alpha);
        } else {
          propRef.current.rotation.set(Math.PI / 2, 0, 0);
        }
      }
    }
    if (ballRef.current) {
      // ...and ONLY for a sphere. `!isCan` left this on for a box too.
      ballRef.current.visible = shape === "sphere";
      if (ball && shape === "sphere") {
        tmpP.set(ball[0], ball[1], ball[2]);
        ballRef.current.position.lerp(tmpP, alpha);
        const r = propRadius(ball);
        ballRef.current.scale.set(r, r, r);
      }
    }
    // THE GHOST: where the policy BELIEVES the ball is. Snapped, not lerped —
    // a belief jumps when a sighting lands, and smoothing it would draw a
    // confidence the estimate does not have. Fades with `conf` so dead
    // reckoning looks like dead reckoning, and turns amber the moment the
    // ball leaves frame.
    const ghost = duck.ballGhost;
    // The TRUTH's footprint radius, which for a box is not `ball[3]` — see
    // `propRadius`. The ghost is drawn at the size of the thing it is a
    // belief about, so it has to ask the same question the truth did.
    const gr = propRadius(ball);
    // The belief is drawn as the SAME SHAPE as the thing it is a belief
    // about: a 6th number is the prop's half-height, exactly as the 5th is
    // on `ball`. A cylinder's ghost drawn as a sphere reads as a different
    // object sitting next to the real one.
    const ghostIsCan = !!ghost && ghost.length > 5;
    for (const [ref, mine] of [
      [ghostRef, !ghostIsCan] as const,
      [ghostPropRef, ghostIsCan] as const,
    ]) {
      if (!ref.current) continue;
      ref.current.visible = !!ghost && mine;
      if (ghost && mine) {
        ref.current.position.set(ghost[0], ghost[1], ghost[2]);
        if (ghostIsCan) {
          ref.current.scale.set(gr, ghost[5] ?? 0.0575, gr);
          // ...AND THE SAME ORIENTATION, for the reason above: an upright
          // ghost beside a can lying on its side reads as a second object,
          // which is the thing this block exists to avoid. The ORIENTATION
          // is the can's own, not the belief's — the 32-slot contract
          // carries `target_base` as position only, so the tracker has no
          // opinion about pose. What is believed is WHERE; the shape and its
          // attitude are the object's, and drawing them truthfully is what
          // makes the gap between ghost and can readable as a position
          // error rather than as two different things.
          if (ball && ball.length >= 9) {
            tmpQ.set(ball[6], ball[7], ball[8], ball[5]);
            tmpQ.multiply(CYL_Z_UP);
            ref.current.quaternion.copy(tmpQ);   // not slerped: see above
          } else {
            ref.current.rotation.set(Math.PI / 2, 0, 0);   // three is Y-up
          }
        } else {
          ref.current.scale.set(gr, gr, gr);
        }
        const m = ref.current.material as THREE.MeshStandardMaterial;
        m.opacity = 0.2 + 0.45 * (ghost[3] ?? 0);
        m.color.set(ghost[4] ? "#43c2b8" : "#e8b24a");
      }
    }
    if (ghostRingRef.current) {
      ghostRingRef.current.visible = !!ghost;
      if (ghost) {
        ghostRingRef.current.position.set(ghost[0], ghost[1], 0.003);
        const s2 = gr * (1.6 + 2.6 * (1 - (ghost[3] ?? 0)));   // widens as the belief goes stale
        ghostRingRef.current.scale.set(s2, s2, 1);
        (ghostRingRef.current.material as THREE.MeshBasicMaterial).color.set(
          ghost[4] ? "#43c2b8" : "#e8b24a");
      }
    }
    // THE LONG GOAL: a ring on the floor at the reach radius, with a pin so
    // it reads at a distance. This is where the BALL is being taken, not
    // where the duck is going — the two differ the moment the ball is off
    // the line, which is the whole skill.
    const tgt = duck.dribbleTarget;
    if (targetRef.current) {
      targetRef.current.visible = !!tgt;
      if (tgt) {
        targetRef.current.position.set(tgt[0], tgt[1], 0.004);
        const rr = tgt[3] || 0.15;
        targetRef.current.scale.set(rr, rr, 1);
      }
    }
    if (targetPinRef.current) {
      targetPinRef.current.visible = !!tgt;
      if (tgt) targetPinRef.current.position.set(tgt[0], tgt[1], 0.06);
    }
    // ...and the error itself, as a segment from the belief to the truth.
    if (ghostLineRef.current) {
      const show = !!ghost && !!ball;
      ghostLineRef.current.visible = show;
      if (show && ghost && ball) {
        const g = ghostLineRef.current.geometry as THREE.BufferGeometry;
        const a = g.getAttribute("position") as THREE.BufferAttribute;
        a.setXYZ(0, ghost[0], ghost[1], ghost[2]);
        a.setXYZ(1, ball[0], ball[1], ball[2]);
        a.needsUpdate = true;
      }
    }
    if (labelDivRef.current) {
      const s = labelDivRef.current.style;
      // HUD 🏷 toggle, applied per-frame like the rest of the label styling
      // (a React subscription inside the Canvas tree flushed a beat late).
      s.display = getDuckLabels() ? "" : "none";
      s.transform = hovered ? "scale(1.3)" : selected ? "scale(1.15)" : "none";
      s.color = hovered ? "#9fd4f0" : selected ? "#e8b24a" : "#fff";
      s.fontWeight = hovered || selected ? "700" : "400";
    }
    // Spawn note updates straight off the stream (textContent, no React
    // churn) — it changes at every episode reset.
    if (spawnDivRef.current) {
      const spawn = duck.spawn && duck.spawn !== "standing" ? `↻ ${duck.spawn}` : "";
      const parts = [spawn];
      if (duck.assist) parts.push("🤝 spotting");
      if (duck.handed && duck.handoff) parts.push(`→ ${duck.handoff}`);
      const txt = parts.filter(Boolean).join(" · ");
      if (spawnDivRef.current.textContent !== txt)
        spawnDivRef.current.textContent = txt;
    }
    // The boundary itself: `step` going backwards, with `falls` incrementing
    // telling a fall from a clip that simply ran out. Both come straight off
    // the stream; see lib/episode.ts for why it is not a `step === 0` test.
    const sample = { step: duck.step, falls: duck.falls };
    const edge = episodeEdge(lastEpisodeRef.current, sample);
    if (edge.reset && edge.cause) {
      resetFlashRef.current = { at: performance.now(), cause: edge.cause };
    }
    lastEpisodeRef.current = sample;
    if (resetDivRef.current) {
      const flash = resetFlashRef.current;
      const alpha = flash ? episodeFlashAlpha(performance.now() - flash.at) : 0;
      const st = resetDivRef.current.style;
      if (alpha <= 0) {
        if (st.opacity !== "0") {
          st.opacity = "0";
          resetDivRef.current.textContent = "";
        }
        if (flash) resetFlashRef.current = null;
      } else {
        const txt2 = episodeFlashLabel(flash!.cause);
        if (resetDivRef.current.textContent !== txt2)
          resetDivRef.current.textContent = txt2;
        st.opacity = String(alpha);
        // A fall is the one worth a different colour: it is a failure, where a
        // clip running out is just the next attempt starting.
        st.color = flash!.cause === "fell" ? "#f08a8a" : "#7fd4a0";
      }
    }
  });

  return (
    <group position={[offset[0], offset[1], 0]}>
      {bodies.map((body, b) =>
        body.geometry ? (
          <group key={b} ref={(el) => void (bodyRefs.current[b] = el)}>
            {body.look === "g1" && g1Materials ? (
              <mesh geometry={body.geometry} material={g1Materials} />
            ) : body.look === "mars" && marsMaterials ? (
              <mesh geometry={body.geometry} material={marsMaterials} />
            ) : body.look === "generic" ? (
              <mesh geometry={body.geometry} material={genericMaterial} />
            ) : (
              <mesh geometry={body.geometry} ref={b === mouthIdx ? billRef : undefined}>
                <meshStandardMaterial vertexColors roughness={0.55} metalness={0.08} />
              </mesh>
            )}
          </group>
        ) : (
          <group key={b} ref={(el) => void (bodyRefs.current[b] = el)} />
        )
      )}
      {/* the ball a 🔎 find_ball duck is looking for (unit sphere, scaled to
          the streamed radius) — orange like the real 70 mm kick ball */}
      <mesh ref={ballRef} visible={false} castShadow>
        <sphereGeometry args={[1, 24, 16]} />
        <meshStandardMaterial color="#ff8c00" roughness={0.5} />
      </mesh>
      {/* ...and the same slot for a prop that is NOT a ball. A MOSS trainee
          practises on a 66 x 115 mm drinks can, and a sphere of its radius is
          a picture of a different object: you cannot see it topple, and a can
          lying down looks identical to one standing up. The server sends a
          fifth number (half-height) when the prop is a cylinder; without it
          nothing here changes and the duck's ball is the sphere it always
          was. Z-up, like every other body in this scene. */}
      <mesh ref={propRef} visible={false} castShadow>
        <cylinderGeometry args={[1, 1, 2, 24]} />
        <meshStandardMaterial color="#ee7362" roughness={0.55} />
      </mesh>
      {/* ...and a BOX, for the litter that is not a can. The server tells the
          shapes apart by payload LENGTH — 4 a sphere, 9 a cylinder, 10 a box —
          so a lab training on six shapes stops drawing every one of them as a
          can. */}
      <mesh ref={boxRef} visible={false} castShadow>
        <boxGeometry args={[2, 2, 2]} />
        <meshStandardMaterial color="#d8cf9e" roughness={0.7} />
      </mesh>
      {/* THE GHOST — the belief, beside the truth above. Translucent, and a
          ring that WIDENS as the estimate goes stale, so "I last saw it a
          second ago" reads differently from "I can see it". */}
      <mesh ref={ghostRef} visible={false}>
        <sphereGeometry args={[1, 20, 14]} />
        <meshStandardMaterial color="#43c2b8" transparent opacity={0.5} depthWrite={false} roughness={0.4} />
      </mesh>
      {/* the same ghost for a prop that is not a ball — the belief wears the
          shape of the thing it is a belief about. */}
      <mesh ref={ghostPropRef} visible={false}>
        <cylinderGeometry args={[1, 1, 2, 20]} />
        <meshStandardMaterial color="#43c2b8" transparent opacity={0.5} depthWrite={false} roughness={0.4} />
      </mesh>
      <mesh ref={ghostRingRef} visible={false} rotation={[0, 0, 0]}>
        <ringGeometry args={[0.85, 1, 32]} />
        <meshBasicMaterial color="#43c2b8" transparent opacity={0.55} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      {/* the long goal — a flat ring at the reach radius plus a standing pin */}
      <mesh ref={targetRef} visible={false}>
        <ringGeometry args={[0.88, 1, 40]} />
        <meshBasicMaterial color="#e8b24a" transparent opacity={0.85} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <mesh ref={targetPinRef} visible={false}>
        <cylinderGeometry args={[0.006, 0.006, 0.12, 8]} />
        <meshBasicMaterial color="#e8b24a" transparent opacity={0.7} depthWrite={false} />
      </mesh>
      {/* THE ERROR SEGMENT. `frustumCulled={false}` because this is the one
          thing here that moves by rewriting its VERTICES instead of its
          transform: three computes a bounding sphere once, lazily, from
          whatever the attribute held at the time — all zeros — and
          `needsUpdate` on the attribute never invalidates it. Left on, the
          line is culled whenever the duck's own origin leaves the frustum,
          which is every time someone zooms in on the ball it is pointing
          at. Same answer as SimStage's blob mesh. `ghostLinePositions` is
          the other half of it: inline, `args` is a new array identity on
          every re-render, r3f rebuilds the attribute and the segment reads
          zeros for a frame. */}
      <lineSegments ref={ghostLineRef} visible={false} frustumCulled={false}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[ghostLinePositions, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#43c2b8" transparent opacity={0.8} />
      </lineSegments>
      {/* drop-target ring, flat on the floor (XY plane in this Z-up group).
          hideInCapture: 📷 snapshots hide it for their capture render. */}
      <mesh
        ref={ringRef}
        visible={false}
        position={[0, 0, 0.004]}
        userData={{ hideInCapture: true }}
      >
        <ringGeometry args={[0.16, 0.19, 48]} />
        <meshBasicMaterial
          color="#7db8d8"
          transparent
          opacity={0.85}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <group ref={labelRef}>
        {/* DOM label (drei Text's GPU glyph atlas was losing the WebGL
            context in the embedded browser — keep labels off the GPU).
            zIndexRange tops out below the overlay panels (zIndex 20) so
            labels can never scribble over the HUD/policies/teach UI.
            Stays mounted when labels are toggled off — the useFrame above
            flips the inner div's display instead. */}
        <Html center zIndexRange={[10, 0]} style={{ pointerEvents: "none" }}>
          <div
            ref={labelDivRef}
            style={{
              color: "#fff",
              fontFamily: "ui-monospace, Menlo, monospace",
              fontSize: 11,
              whiteSpace: "nowrap",
              textShadow: "0 1px 3px rgba(0,0,0,0.9)",
              opacity: 0.9,
              transition: "transform 120ms ease, color 120ms ease",
            }}
          >
            {label}
            <div
              ref={spawnDivRef}
              style={{
                fontSize: 9,
                color: "#e8b24a",
                textAlign: "center",
                minHeight: 11,
              }}
            />
            <div
              ref={resetDivRef}
              style={{
                fontSize: 9,
                fontWeight: 700,
                textAlign: "center",
                minHeight: 11,
                opacity: 0,
                transition: "none",
                pointerEvents: "none",
              }}
            />
          </div>
        </Html>
      </group>
    </group>
  );
}
