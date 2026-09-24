// A team's colours are baked into the duck geometry's vertex-color channel
// (components/Duck.tsx buildBodyGeometries). The screenshot cannot tell a
// cream shell from a white one under this lighting, so the lock is here.

import { describe, expect, it } from "vitest";

import { buildBodyGeometries } from "@/components/Duck";
import { PITCH_TEAMS, SHELL_MATERIALS, TEAM_COLORWAYS, TRIM_MATERIALS } from "./sim";
import type { Scene } from "./lab";

/** One body, one triangle a material: the smallest scene that carries a
 *  material name through the merge. */
function scene(mats: string[]): Scene {
  return {
    bodies: ["world", "trunk_base"],
    meshes: [{ v: [0, 0, 0, 1, 0, 0, 0, 1, 0], f: [0, 1, 2] }],
    geoms: mats.map((mat) => ({
      mesh: 0, body: 1, pos: [0, 0, 0] as [number, number, number],
      quat: [1, 0, 0, 0] as [number, number, number, number], mat, rgba: [0.8, 0.8, 0.8, 1] as [number, number, number, number],
    })),
  } as unknown as Scene;
}

/** The first vertex colour of the merged trunk geometry, per geom in order. */
function colors(mats: string[], team?: string | null): [number, number, number][] {
  const geo = buildBodyGeometries(scene(mats), team).find((b) => b.name === "trunk_base")!.geometry!;
  const c = geo.getAttribute("color");
  // Three vertices a geom, in the order the geoms were merged.
  return mats.map((_, i) => [c.getX(i * 3), c.getY(i * 3), c.getZ(i * 3)]);
}

describe("buildBodyGeometries: a team's colorway", () => {
  const shell = SHELL_MATERIALS[0];
  const trim = TRIM_MATERIALS[0];
  const leg = "leg_material";
  const other = "xl330_material";

  it("paints every printed part of the named team and nothing else", () => {
    const [a, b, c, d] = colors([shell, trim, leg, other], "sky");
    const [pa, pb, pc, pd] = colors([shell, trim, leg, other]);
    expect(a).not.toEqual(pa);
    expect(b).not.toEqual(pb);
    expect(c).not.toEqual(pc);
    expect(d).toEqual(pd);                       // a servo is a servo on every duck
  });

  it("gives two teams different shells", () => {
    expect(colors([shell], "cream")[0]).not.toEqual(colors([shell], "sky")[0]);
    expect(colors([shell], "cream")[0]).toEqual(colors([shell], "cream")[0]);
  });

  // The complaint that produced this: a duck rendered as a patchwork, because
  // the printed thigh plate, shoe rim, hip and soft mouth carry CAD-export
  // colours no colorway claimed. Every SHELL part must land on exactly one
  // colour and every TRIM part on exactly one, for either pitch team.
  it("puts every printed part of a team on one of its two colours", () => {
    for (const t of PITCH_TEAMS) {
      const shells = colors(SHELL_MATERIALS, t).map((c) => c.join(","));
      const trims = colors(TRIM_MATERIALS, t).map((c) => c.join(","));
      expect(new Set(shells).size).toBe(1);
      expect(new Set(trims).size).toBe(1);
      expect(shells[0]).not.toEqual(trims[0]);
    }
  });

  it("leaves the duck alone for no team and for a colorway that does not exist", () => {
    const plain = colors([shell, trim, leg, other]);
    expect(colors([shell, trim, leg, other], null)).toEqual(plain);
    expect(colors([shell, trim, leg, other], "puce")).toEqual(plain);
  });

  it("knows the same four colorways the server does, each a shell and a trim", () => {
    expect(Object.keys(TEAM_COLORWAYS).sort()).toEqual(["cream", "graphite", "lavender", "sky"]);
    for (const look of Object.values(TEAM_COLORWAYS)) expect(look.shell).not.toEqual(look.trim);
  });
});
