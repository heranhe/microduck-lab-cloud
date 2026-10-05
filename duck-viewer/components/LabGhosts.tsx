"use client";

/** The BELIEFS a lab body wants drawn, beside the truth — any number of them.
 *
 *  The single `ballGhost` field was enough while a scene held one trainee
 *  chasing one thing. It is not: the lab runs a duck on a ball and a MOSS on
 *  a can at once, and one robot may track several objects (a tidy brain has a
 *  toy AND a basket). Each entry is self-describing, so this component needs
 *  no lookup table and no id matching against a separate list.
 *
 *  Pools, not mounts: beliefs appear and vanish as sightings land, and a
 *  component per ghost would remount continuously. Two mesh pools (sphere and
 *  cylinder, so a can reads as a can and you can see it topple), a ring pool
 *  for the staleness radius, and one line buffer for the error segments.
 *
 *  Snapped, not lerped — a belief JUMPS when a sighting lands, and smoothing
 *  it would draw a confidence the estimate does not have. */

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

import type { LabGhost } from "@/lib/lab";

const MAX = 6;

export function LabGhosts({ ghosts }: { ghosts?: LabGhost[] | null }) {
  const spheres = useRef<(THREE.Mesh | null)[]>([]);
  const cylinders = useRef<(THREE.Mesh | null)[]>([]);
  const rings = useRef<(THREE.Mesh | null)[]>([]);
  const lines = useRef<THREE.LineSegments>(null);

  const linePos = useMemo(() => new Float32Array(MAX * 6), []);
  const lineGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
    return g;
  }, [linePos]);

  useFrame(() => {
    const gs = (ghosts ?? []).slice(0, MAX);
    let nSphere = 0, nCyl = 0, nLine = 0;
    for (let i = 0; i < MAX; i++) {
      const g = gs[i];
      const ring = rings.current[i];
      if (ring) {
        ring.visible = !!g;
        if (g) {
          ring.position.set(g.pos[0], g.pos[1], 0.003);
          // The ring WIDENS as the belief goes stale, so "I last saw it a
          // second ago" reads differently from "I can see it".
          const s = (g.r || 0.035) * (1.6 + 2.6 * (1 - (g.conf ?? 0)));
          ring.scale.set(s, s, 1);
          (ring.material as THREE.MeshBasicMaterial).color.set(g.seen ? "#43c2b8" : "#e8b24a");
        }
      }
      if (g && g.truth) {
        const o = nLine * 6;
        linePos[o] = g.pos[0]; linePos[o + 1] = g.pos[1]; linePos[o + 2] = g.pos[2];
        linePos[o + 3] = g.truth[0]; linePos[o + 4] = g.truth[1]; linePos[o + 5] = g.truth[2];
        nLine++;
      }
    }
    // Shape pools are filled independently of the ghost index, so a cylinder
    // and a sphere in the same frame each take the next free slot of their
    // own kind rather than colliding on one.
    for (const g of gs) {
      const cyl = g.shape === "cylinder";
      const pool = cyl ? cylinders.current : spheres.current;
      const idx = cyl ? nCyl++ : nSphere++;
      const m = pool[idx];
      if (!m) continue;
      m.visible = true;
      m.position.set(g.pos[0], g.pos[1], g.pos[2]);
      const r = g.r || 0.035;
      if (cyl) m.scale.set(r, (g.halfH || r) , r);
      else m.scale.set(r, r, r);
      const mat = m.material as THREE.MeshStandardMaterial;
      mat.opacity = 0.2 + 0.45 * (g.conf ?? 0);
      mat.color.set(g.seen ? "#43c2b8" : "#e8b24a");
    }
    for (let i = nSphere; i < MAX; i++) { const m = spheres.current[i]; if (m) m.visible = false; }
    for (let i = nCyl; i < MAX; i++) { const m = cylinders.current[i]; if (m) m.visible = false; }
    if (lines.current) {
      lines.current.visible = nLine > 0;
      lineGeom.attributes.position.needsUpdate = true;
      lineGeom.setDrawRange(0, nLine * 2);
    }
  });

  const idx = Array.from({ length: MAX }, (_, i) => i);
  return (
    <group>
      {idx.map((i) => (
        <mesh key={`s${i}`} ref={(el) => { spheres.current[i] = el; }} visible={false}>
          <sphereGeometry args={[1, 18, 12]} />
          <meshStandardMaterial color="#43c2b8" transparent opacity={0.5} depthWrite={false} roughness={0.4} />
        </mesh>
      ))}
      {idx.map((i) => (
        // unit radius, unit HALF-height: the frame loop scales x/z by the
        // radius and y by the half-height, so a can keeps its proportions.
        <mesh key={`c${i}`} ref={(el) => { cylinders.current[i] = el; }} visible={false} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[1, 1, 2, 16]} />
          <meshStandardMaterial color="#43c2b8" transparent opacity={0.5} depthWrite={false} roughness={0.4} />
        </mesh>
      ))}
      {idx.map((i) => (
        <mesh key={`r${i}`} ref={(el) => { rings.current[i] = el; }} visible={false}>
          <ringGeometry args={[0.85, 1, 32]} />
          <meshBasicMaterial color="#43c2b8" transparent opacity={0.55} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      ))}
      <lineSegments ref={lines} geometry={lineGeom} visible={false}>
        <lineBasicMaterial color="#43c2b8" transparent opacity={0.8} />
      </lineSegments>
    </group>
  );
}
