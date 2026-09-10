// The brain menu files learned runs under their use case; this is the
// arithmetic behind the headings (lib/sim.ts groupLearned).

import { describe, expect, it } from "vitest";

import { applyFloorClick, makePitch, makeRoom } from "@/components/SimEditor";
import { goalDefenders, groupLearned, LEARNED_GROUPS, PITCH_TEAMS,
  type LearnedInfo, type Scenario } from "./sim";

const b = (name: string, group: string | null, title: string | null = null): LearnedInfo => ({
  name, group, title, description: null,
});

describe("groupLearned", () => {
  it("files by group in menu order and skips empty groups", () => {
    const out = groupLearned([b("z1-s81", "null-pair"), b("follow-v4", "shipped-followers", "Follower v4"), b("z2-s81", "null-pair")]);
    expect(out.map(([label]) => label)).toEqual(["Followers (shipped)", "Null pair (seeds 81–84)"]);
    expect(out[1][1].map((x) => x.name)).toEqual(["z1-s81", "z2-s81"]);
  });

  it("files an unknown or missing group under Other, never drops it", () => {
    const out = groupLearned([b("striker-v1", null), b("odd", "no-such-group")]);
    expect(out).toEqual([["Other", [b("striker-v1", null), b("odd", "no-such-group")]]]);
  });

  it("returns nothing for nothing", () => {
    expect(groupLearned([])).toEqual([]);
    expect(LEARNED_GROUPS.at(-1)![0]).toBe("other");
  });
});

import { menuBrains } from "./sim";

describe("menuBrains", () => {
  const runs: LearnedInfo[] = [
    b("follow-v4", "shipped-followers", "Follower v4"),
    b("follow-v1", "shipped-followers", "Follower v1"),
    b("p-n256-s31", "capacity"), b("p-n256-s32", "capacity"),
    b("z1-s81", "null-pair"),
  ];

  it("offers only the shipped brains by default, and says how many it hid", () => {
    const m = menuBrains(runs, "wander", false);
    expect(m.groups.map(([label, bs]) => [label, bs.length])).toEqual([["Followers (shipped)", 2]]);
    expect(m.hidden).toBe(3);
  });

  it("never hides the brain the duck is on", () => {
    const m = menuBrains(runs, "learned:z1-s81", false);
    expect(m.groups.map(([label]) => label)).toEqual(["Followers (shipped)", "Null pair (seeds 81–84)"]);
    expect(m.hidden).toBe(2);
  });

  it("shows everything when asked", () => {
    const m = menuBrains(runs, null, true);
    expect(m.groups.length).toBe(3);
    expect(m.hidden).toBe(0);
  });
});

describe("applyFloorClick: placing a duck on a pitch", () => {
  // The editor's job in Track 4.2: a duck placed on a pitch joins the team of
  // the half it stands in and faces the goal that team attacks. Any other
  // placement is refused by the server on save (a team facing both goals),
  // so the default has to be the legal one.
  const pitch = (): Scenario => ({
    version: 1,
    name: "p",
    seed: 0,
    floor: { size: [4, 3] },
    walls: [],
    boxes: [],
    balls: [{ pos: [0, 0], radius: 0.035, mass: 0.015 }],
    ducks: [],
    goal_width: 0.7,
    collision: "walk",
  });
  const place = (draft: Scenario, x: number) =>
    applyFloorClick({ draft, tool: "duck", wallStart: null }, x, 0).draft.ducks.at(-1)!;

  it("puts a duck in its own half's team, facing the other goal", () => {
    const [home, away] = PITCH_TEAMS;
    const a = place(pitch(), -1);
    expect([a.team, a.brain, a.spawn[2]]).toEqual([home, "chase", 0]);
    const b = place({ ...pitch(), ducks: [a] }, 1);
    expect([b.team, b.brain, b.spawn[2]]).toEqual([away, "chase", Math.PI]);
  });

  it("follows the teams already on the pitch rather than assuming the default pair", () => {
    const sky = { ...place(pitch(), -1), team: "sky" as const };
    const mate = place({ ...pitch(), ducks: [sky] }, -1.2);
    expect(mate.team).toBe("sky");
    expect(place({ ...pitch(), ducks: [sky] }, 1.2).team).toBe(PITCH_TEAMS[1]);
  });

  it("leaves a duck teamless off a pitch", () => {
    const room = { ...pitch(), goal_width: 0 };
    const d = place(room, -1);
    expect(d.team).toBeUndefined();
    expect(d.brain).toBeUndefined();
  });
});

describe("makePitch / makeRoom: a pitch drawn in the editor is the lab's pitch", () => {
  // The lab's pitch builtins carry a 15 cm cove and 30 cm chamfered corners
  // (world_server.PITCH_COVE / PITCH_CORNER); a pitch toggled on in the
  // editor must match them or it plays a different game than /sim's own.
  const wall = (a: [number, number], b: [number, number]) => ({ from: a, to: b, height: 0.3, thickness: 0.02 });
  const room = (walls: Scenario["walls"]): Scenario => ({
    version: 1, name: "r", seed: 0, floor: { size: [3.9, 3.35] }, walls, boxes: [], balls: [], ducks: [], collision: "all",
  });
  const rect = room([
    wall([-1.7, -1.425], [1.7, -1.425]), wall([1.7, -1.425], [1.7, 1.425]),
    wall([1.7, 1.425], [-1.7, 1.425]), wall([-1.7, 1.425], [-1.7, -1.425]),
  ]);
  it("gives a rectangular room the cove and chamfered corners, and takes them back", () => {
    const p = makePitch(rect);
    expect(p.goal_width).toBe(0.7);
    expect(p.cove).toBe(0.15);
    expect(p.walls).toHaveLength(8);
    for (let i = 0; i < 8; i++) expect(p.walls[i].to).toEqual(p.walls[(i + 1) % 8].from);
    expect(p.walls[0].from).toEqual([-1.4, -1.425]);          // 0.3 in from the corner, as make_pitch(corner=0.3)
    expect(p.walls[1].to).toEqual([1.7, -1.125]);
    expect(p.balls).toHaveLength(1);
    const r = makeRoom(p);
    expect(r.goal_width).toBe(0);
    expect(r.cove).toBe(0);
    expect(r.walls).toHaveLength(4);
    expect(r.walls.map((w) => w.from)).toEqual([[-1.7, -1.425], [1.7, -1.425], [1.7, 1.425], [-1.7, 1.425]]);
  });
  it("leaves a room that is not a plain rectangle as drawn, cove and all", () => {
    const ell = room([...rect.walls, wall([0, -1.425], [0, 0])]);   // an interior wall: the user's layout
    const p = makePitch(ell);
    expect(p.cove).toBe(0.15);
    expect(p.walls).toEqual(ell.walls);
    expect(makeRoom(p).walls).toEqual(ell.walls);
  });
  it("draws the sides in either direction and still finds the rectangle", () => {
    const flipped = room(rect.walls.map((w) => wall(w.to, w.from)).reverse());
    expect(makePitch(flipped).walls).toHaveLength(8);
  });
});

describe("goalDefenders: whose end is which", () => {
  // The stage paints a goal frame in the colours of the team that KEEPS it,
  // and the mouth keys are the World's (`right` is the mouth at +x), so the
  // team spawned at −x facing +x defends `left`. Getting this backwards paints
  // both ends the wrong colour, which is worse than painting neither.
  const pitch = (ducks: Scenario["ducks"], attacks?: Scenario["attacks"]): Scenario => ({
    version: 1, name: "p", seed: 0, floor: { size: [4, 3] }, walls: [], boxes: [],
    balls: [{ pos: [0, 0], radius: 0.035, mass: 0.015 }], ducks, goal_width: 0.7,
    attacks, collision: "walk",
  });
  const duck = (id: string, x: number, yaw: number, team: string): Scenario["ducks"][number] =>
    ({ id, spawn: [x, 0, yaw], policy: null, tof: null, team: team as never });
  const [home, away] = PITCH_TEAMS;

  it("reads the spawn headings when the scenario declares nothing", () => {
    const s = pitch([duck("d0", -1, 0, home), duck("d1", 1, Math.PI, away)]);
    expect(goalDefenders(s)).toEqual({ left: home, right: away });
  });

  it("takes the declaration over the heading — a defender faces its OWN goal", () => {
    const s = pitch([duck("d0", -1, Math.PI, home), duck("d1", 1, Math.PI, away)],
                    { [home]: "right", [away]: "left" } as Scenario["attacks"]);
    expect(goalDefenders(s)).toEqual({ left: home, right: away });
  });

  it("leaves the far end unpainted on a one-team pitch, and both off a pitch", () => {
    expect(goalDefenders(pitch([duck("d0", -1, 0, home)]))).toEqual({ left: home, right: null });
    expect(goalDefenders({ ...pitch([duck("d0", -1, 0, home)]), goal_width: 0 }))
      .toEqual({ left: null, right: null });
    expect(goalDefenders(null)).toEqual({ left: null, right: null });
  });

  it("leaves an end unpainted when two teams attack it", () => {
    const s = pitch([duck("d0", -1, 0, home), duck("d1", 1, Math.PI, away), duck("d2", 1, Math.PI, "sky")]);
    expect(goalDefenders(s)).toEqual({ left: home, right: null });
  });
});
