// The brain menu files learned runs under their use case; this is the
// arithmetic behind the headings (lib/sim.ts groupLearned).

import { describe, expect, it } from "vitest";

import { applyFloorClick } from "@/components/SimEditor";
import { groupLearned, LEARNED_GROUPS, PITCH_TEAMS, type LearnedInfo, type Scenario } from "./sim";

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
