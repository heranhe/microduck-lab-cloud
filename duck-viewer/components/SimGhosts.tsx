"use client";

/** THE GHOST: what a brain BELIEVES is where, drawn beside what is there.
 *
 *  Every number a duck acts on about an object is an ESTIMATE, and the gap
 *  between it and the truth is the thing this repo keeps having to measure
 *  indirectly — a stale track, a ball that rolled out of frame, a plan laid
 *  on a belief that was already wrong. This draws the gap: a translucent
 *  blob at the position the brain would act on, a ring at its stated
 *  uncertainty, and a line to where the object actually is. When a duck
 *  walks toward a ball it last saw a second ago, the line is the error.
 *
 *  **Why a sphere, whatever the object is.** The detector models every target
 *  as a sphere of a given radius (`world/arena.Target`, and a prop's own
 *  `Prop.radius()`), so a brain does not believe in a cube or a cylinder — it
 *  believes in a blob at a bearing and a range. Drawing the ghost in the real
 *  object's shape would claim the estimate carries an orientation it has
 *  never had.
 *
 *  **What the ring means, exactly.** It is the track's own `sigma` — read off
 *  the tracker, never recomputed here, so the page cannot disagree with the
 *  controller about what was believed. A 2-D position error is RADIAL, so a
 *  circle at 1σ contains about 39% of the errors and 2σ about 86%, NOT the
 *  68/95 that a per-axis reading suggests (calibrated in
 *  `scripts/probe_shot_gate.py`). The ring is labelled 1σ and nothing here
 *  implies a confidence it does not carry.
 *
 *  Pools, not mounts: tracks appear and vanish every few frames, and a
 *  component per track would remount continuously. One instanced sphere, one
 *  instanced ring and one line buffer, all sized MAX_GHOSTS and truncated by
 *  `count` each frame. The frame is read inside `useFrame` off `client.frame`
 *  (never through a React subscription — see duck-viewer/README.md). */

import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

import { getSelectedDuck } from "@/lib/select";
import type { SimClient, SimObject } from "@/lib/sim";

const MAX_GHOSTS = 48;
const Z_RING = 0.002;            // just off the floor, above the contact blobs
const MIN_R = 0.02;              // a ghost smaller than this is invisible at stage distance
const MIN_SIGMA = 0.015;         // ...and a ring at 0 reads as a dot rather than "certain"

/** One tint per duck, so two ducks' beliefs about the same ball read apart.
 *  Ordered by the frame's duck order, which is the scenario's. */
const TINTS = ["#5ad1ff", "#ffd45a", "#ff7ad1", "#8affa0", "#c9a6ff", "#ff9a6a"];

export function SimGhosts({ client, enabled = true }: { client: SimClient; enabled?: boolean }) {
  const balls = useRef<THREE.InstancedMesh>(null);
  const rings = useRef<THREE.InstancedMesh>(null);
  const lines = useRef<THREE.LineSegments>(null);

  const m = useMemo(() => new THREE.Matrix4(), []);
  const col = useMemo(() => new THREE.Color(), []);
  const linePos = useMemo(() => new Float32Array(MAX_GHOSTS * 6), []);
  const lineCol = useMemo(() => new Float32Array(MAX_GHOSTS * 6), []);
  const lineGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(lineCol, 3));
    return g;
  }, [linePos, lineCol]);

  // NO `vertexColors` on these two. An InstancedMesh tints itself through
  // `instanceColor`, which three defines as USE_INSTANCING_COLOR on its own
  // as soon as `setColorAt` has run — exactly how the stage's contact blobs
  // do it. Setting `vertexColors: true` instead sends the shader looking for
  // a GEOMETRY-level colour attribute that a sphere and a ring do not have,
  // and every ghost came out grey (which on a four-duck pitch defeats the
  // point: telling whose belief is whose is the whole reason for the tint).
  // The line segments below DO set it, because their geometry really does
  // carry a per-vertex colour buffer.
  const ghostMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        transparent: true,
        opacity: 0.38,
        depthWrite: false,
        roughness: 0.4,
        metalness: 0,
      }),
    [],
  );
  const ringMat = useMemo(
    () => new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.65, depthWrite: false, side: THREE.DoubleSide }),
    [],
  );
  const lineMat = useMemo(() => new THREE.LineBasicMaterial({ transparent: true, opacity: 0.8, vertexColors: true }), []);

  useFrame(() => {
    const bi = balls.current, ri = rings.current, li = lines.current;
    if (!bi || !ri || !li) return;
    const f = client.frame;
    if (!f || !enabled) {
      bi.count = 0;
      ri.count = 0;
      lineGeom.setDrawRange(0, 0);
      return;
    }
    // The truth to compare against: every streamed object, by id. A track's
    // `name` is the sim's own name for the thing it is tracking, which is
    // exactly that id — so this is a lookup and not a nearest-match guess.
    const truth = new Map<string, SimObject>();
    for (const o of f.objects) truth.set(o.id, o);

    const sel = getSelectedDuck();
    let n = 0;
    let di = 0;
    for (const d of f.ducks) {
      const tint = TINTS[di++ % TINTS.length];
      if (sel !== null && sel !== d.id) continue;   // one duck's beliefs at a time when one is picked
      for (const tr of d.brain?.inputs?.tracks ?? []) {
        if (n >= MAX_GHOSTS || !tr.pred) continue;
        const real = truth.get(tr.name);
        // Radius: the real object's own, so the ghost is the size of the
        // thing it stands for. A track on something not in `objects` (a goal
        // post, another duck) still gets a ghost, at MIN_R.
        const r = Math.max(MIN_R, real?.prop ? real.prop.size[0] / 2 : real?.kind === "ball" ? 0.035 : MIN_R);
        const z = real ? real.pose[2] : r;
        col.set(tint);

        m.makeScale(r, r, r);
        m.setPosition(tr.pred[0], tr.pred[1], z);
        bi.setMatrixAt(n, m);
        bi.setColorAt(n, col);

        const s = Math.max(MIN_SIGMA, tr.sigma ?? 0);
        m.makeScale(s, s, 1);
        m.setPosition(tr.pred[0], tr.pred[1], Z_RING);
        ri.setMatrixAt(n, m);
        ri.setColorAt(n, col);

        // The error line: belief → truth. Absent when there is no truth to
        // point at, which is drawn as a zero-length segment rather than a
        // line to the origin.
        const o = n * 6;
        linePos[o] = tr.pred[0];
        linePos[o + 1] = tr.pred[1];
        linePos[o + 2] = z;
        linePos[o + 3] = real ? real.pose[0] : tr.pred[0];
        linePos[o + 4] = real ? real.pose[1] : tr.pred[1];
        linePos[o + 5] = real ? real.pose[2] : z;
        for (let k = 0; k < 2; k++) {
          lineCol[o + k * 3] = col.r;
          lineCol[o + k * 3 + 1] = col.g;
          lineCol[o + k * 3 + 2] = col.b;
        }
        n++;
      }
    }
    bi.count = n;
    ri.count = n;
    bi.instanceMatrix.needsUpdate = true;
    ri.instanceMatrix.needsUpdate = true;
    if (bi.instanceColor) bi.instanceColor.needsUpdate = true;
    if (ri.instanceColor) ri.instanceColor.needsUpdate = true;
    lineGeom.attributes.position.needsUpdate = true;
    lineGeom.attributes.color.needsUpdate = true;
    lineGeom.setDrawRange(0, n * 2);
  });

  return (
    <group>
      <instancedMesh ref={balls} args={[undefined, undefined, MAX_GHOSTS]} material={ghostMat} frustumCulled={false}>
        <sphereGeometry args={[1, 16, 12]} />
      </instancedMesh>
      <instancedMesh ref={rings} args={[undefined, undefined, MAX_GHOSTS]} material={ringMat} frustumCulled={false}>
        {/* unit outer radius, so the instance scale IS sigma */}
        <ringGeometry args={[0.88, 1, 40]} />
      </instancedMesh>
      <lineSegments ref={lines} geometry={lineGeom} material={lineMat} frustumCulled={false} />
    </group>
  );
}
