import { describe, expect, it } from "vitest";

import { mapView, wallsStatus, wristBand, type MossMap } from "@/components/SimMossMap";

const base: MossMap = { area: null, rim: [], mem: [], target: null, goal: null, look: null, leg: null, cov: null };
// A 1 m x 0.5 m patch of grid, 0.1 m cells, with two solid cells.
const room = { x0: -1, y0: -1, cell: 0.1, n: 20, solid: [5 * 20 + 3, 12 * 20 + 15], felt: [] as [number, number][], swept: 0.4 };

describe("MOSS's map", () => {
  it("frames the work area once it has one", () => {
    expect(mapView({ ...base, area: [-1.5, 1.5, -1.2, 1.2], room }, [0, 0, 0])).toEqual([-1.5, 1.5, -1.2, 1.2]);
  });
  it("frames what the depth has found, and the robot, before it has an area", () => {
    const v = mapView({ ...base, room }, [0, 0, 0])!;
    expect(v[0]).toBeCloseTo(-1 + 3.5 * 0.1);        // the solid cell at ix 3
    expect(v[1]).toBeCloseTo(-1 + 15.5 * 0.1);       // ...and at ix 15
    expect(v[2]).toBeCloseTo(-0.5);                  // the robot's box reaches lower than the cells
  });
  it("draws the wrist camera's reportable BAND, not a wedge from the lens", () => {
    // looking along +x from the origin, reporting floor from 0.21 m to 0.54 m
    const band = wristBand([0, 0, 0, 0.21, 0.54]);
    const r = band.map(([x, y]) => Math.hypot(x, y));
    expect(Math.min(...r)).toBeCloseTo(0.21);
    expect(Math.max(...r)).toBeCloseTo(0.54);
    // the floor beside the tracks is NOT claimed: nothing inside the near edge
    expect(r.every((v) => v >= 0.21 - 1e-9)).toBe(true);
    // and it stays inside the 70 deg horizontal view
    const bearings = band.map(([x, y]) => Math.abs(Math.atan2(y, x)));
    expect(Math.max(...bearings)).toBeCloseTo((35 * Math.PI) / 180);
    // it closes: the ring walks out along the far arc and back along the near
    expect(band).toHaveLength(14);
  });
  it("says where the walls came from", () => {
    expect(wallsStatus({ ...base, area: [0, 1, 0, 1], areaSrc: "sensed", room })).toMatch(/depth camera/);
    expect(wallsStatus({ ...base, area: [0, 1, 0, 1], areaSrc: "given" })).toMatch(/GIVEN/);
    expect(wallsStatus({ ...base, room })).toMatch(/40% of a turn/);
  });
});
